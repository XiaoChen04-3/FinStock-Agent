"""Plan-and-Execute agent with automatic fallback to ReAct on repeated failures."""
from __future__ import annotations

import json
import operator
from typing import Annotated, Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from fin_stock_agent.agents.react_agent import build_react_agent, extract_last_ai_text
from fin_stock_agent.core.llm import get_llm
from fin_stock_agent.prompts.plan_prompt import (
    FINALIZE_PROMPT,
    PLANNER_PROMPT,
    REPLANNER_PROMPT,
)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class PlanExecuteState(TypedDict):
    question: str
    plan: list[str]
    past_steps: Annotated[list[tuple[str, str]], operator.add]
    response: str | None
    error_count: int
    fallback_triggered: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAX_ERRORS = 2  # after this many executor errors, fallback to ReAct


def _parse_json_list(text: str) -> list[str]:
    """Extract a JSON array from an LLM response, returning [] on failure."""
    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        result = json.loads(text[start: end + 1])
        if isinstance(result, list):
            return [str(s) for s in result]
    except json.JSONDecodeError:
        pass
    return []


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def _planner_node(state: PlanExecuteState) -> dict:
    llm = get_llm()
    prompt = PLANNER_PROMPT.format(question=state["question"])
    response = llm.invoke([HumanMessage(content=prompt)])
    plan = _parse_json_list(response.content if isinstance(response.content, str) else "")
    if not plan:
        # Fallback: treat entire question as single step
        plan = [state["question"]]
    return {"plan": plan, "past_steps": [], "error_count": 0, "fallback_triggered": False}


def _executor_node(state: PlanExecuteState) -> dict:
    """Execute the next un-finished step using the ReAct agent."""
    completed = len(state.get("past_steps", []))
    plan = state.get("plan", [])
    if completed >= len(plan):
        return {}

    current_step = plan[completed]
    react = build_react_agent()
    try:
        result = react.invoke({"messages": [HumanMessage(content=current_step)]})
        step_result = extract_last_ai_text(result.get("messages", [])) or "（步骤无返回文本）"
        return {"past_steps": [(current_step, step_result)]}
    except Exception as e:
        error_count = state.get("error_count", 0) + 1
        return {
            "past_steps": [(current_step, f"[ERROR] {e}")],
            "error_count": error_count,
        }


def _replan_node(state: PlanExecuteState) -> dict:
    """After a step error, ask LLM to replan remaining steps."""
    past = state.get("past_steps", [])
    original_plan = "\n".join(f"{i+1}. {s}" for i, s in enumerate(state["plan"]))
    completed_text = "\n".join(
        f"步骤「{s}」→ {r}" for s, r in past
    )
    # Find the last error reason
    error_reason = next(
        (r for _, r in reversed(past) if r.startswith("[ERROR]")),
        "未知错误",
    )
    prompt = REPLANNER_PROMPT.format(
        original_plan=original_plan,
        completed_steps=completed_text,
        error_reason=error_reason,
    )
    llm = get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    new_remaining = _parse_json_list(
        response.content if isinstance(response.content, str) else ""
    )
    # Replace plan from current position onwards
    completed_count = len(past)
    new_plan = state["plan"][:completed_count] + new_remaining
    return {"plan": new_plan, "error_count": 0}


def _fallback_node(state: PlanExecuteState) -> dict:
    """Fall back to a single ReAct call on the original question."""
    react = build_react_agent()
    try:
        result = react.invoke({"messages": [HumanMessage(content=state["question"])]})
        answer = extract_last_ai_text(result.get("messages", [])) or "（无返回文本）"
    except Exception as e:
        answer = f"降级 ReAct 也失败了：{e}"
    return {"response": f"[已降级至 ReAct 模式]\n\n{answer}", "fallback_triggered": True}


def _finalize_node(state: PlanExecuteState) -> dict:
    """Summarise all step results into a coherent final answer."""
    step_results = "\n\n".join(
        f"**步骤{i+1}「{s}」**\n{r}" for i, (s, r) in enumerate(state.get("past_steps", []))
    )
    prompt = FINALIZE_PROMPT.format(
        question=state["question"],
        step_results=step_results,
    )
    llm = get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    answer = response.content if isinstance(response.content, str) else ""
    return {"response": answer}


# ---------------------------------------------------------------------------
# Routing conditions
# ---------------------------------------------------------------------------

def _after_executor(state: PlanExecuteState) -> Literal["finalize", "executor", "replan", "fallback"]:
    plan = state.get("plan", [])
    past = state.get("past_steps", [])
    error_count = state.get("error_count", 0)

    if error_count >= _MAX_ERRORS:
        return "fallback"

    # Check if last step errored
    if past and past[-1][1].startswith("[ERROR]"):
        return "replan"

    if len(past) >= len(plan):
        return "finalize"

    return "executor"


def _after_replan(state: PlanExecuteState) -> Literal["executor", "finalize", "fallback"]:
    plan = state.get("plan", [])
    past = state.get("past_steps", [])
    error_count = state.get("error_count", 0)

    if error_count >= _MAX_ERRORS:
        return "fallback"
    if len(past) >= len(plan):
        return "finalize"
    return "executor"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_plan_execute_agent():
    """Build and compile the Plan-and-Execute StateGraph."""
    g = StateGraph(PlanExecuteState)

    g.add_node("planner", _planner_node)
    g.add_node("executor", _executor_node)
    g.add_node("replan", _replan_node)
    g.add_node("fallback", _fallback_node)
    g.add_node("finalize", _finalize_node)

    g.set_entry_point("planner")
    g.add_edge("planner", "executor")
    g.add_conditional_edges(
        "executor",
        _after_executor,
        {
            "executor": "executor",
            "replan": "replan",
            "fallback": "fallback",
            "finalize": "finalize",
        },
    )
    g.add_conditional_edges(
        "replan",
        _after_replan,
        {"executor": "executor", "finalize": "finalize", "fallback": "fallback"},
    )
    g.add_edge("fallback", END)
    g.add_edge("finalize", END)

    return g.compile()
