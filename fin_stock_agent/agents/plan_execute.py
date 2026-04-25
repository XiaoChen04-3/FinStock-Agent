from __future__ import annotations

import operator
from typing import Annotated, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from fin_stock_agent.agents.react_agent import build_react_agent, extract_last_ai_text
from fin_stock_agent.core.config import get_config
from fin_stock_agent.core.llm import invoke_json, invoke_text
from fin_stock_agent.prompts.plan_prompt import FINALIZE_PROMPT, PLANNER_PROMPT, REPLANNER_PROMPT


class PlanExecuteState(TypedDict):
    question: str
    context: str
    plan: list[str]
    past_steps: Annotated[list[tuple[str, str]], operator.add]
    response: str | None
    error_count: int
    fallback_triggered: bool
    similar_plans: list[dict]


def _prepare_node(state: PlanExecuteState) -> dict:
    return {
        "context": state.get("context", ""),
        "similar_plans": state.get("similar_plans", []),
    }


def _planner_node(state: PlanExecuteState) -> dict:
    cfg = get_config().plan_execute
    parsed = invoke_json(
        "planner",
        [
            HumanMessage(
                content=PLANNER_PROMPT.format(
                    question=state["question"],
                    context=state.get("context", ""),
                    similar_plans=_format_similar_plans(state.get("similar_plans", [])),
                    min_steps=cfg.min_plan_steps,
                    max_steps=cfg.max_plan_steps,
                )
            )
        ],
    )
    plan = [str(item) for item in parsed] if isinstance(parsed, list) else []
    return {
        "plan": plan or [state["question"]],
        "past_steps": [],
        "error_count": 0,
        "fallback_triggered": False,
    }


def _executor_node(state: PlanExecuteState, config: RunnableConfig | None = None) -> dict:
    completed = len(state.get("past_steps", []))
    if completed >= len(state.get("plan", [])):
        return {}
    current_step = state["plan"][completed]
    agent = build_react_agent("executor")
    messages = [HumanMessage(content=current_step)]
    if state.get("context"):
        messages.insert(0, SystemMessage(content=state["context"]))
    try:
        result = agent.invoke({"messages": messages}, config=config)
        answer = extract_last_ai_text(result.get("messages", [])) or "(no answer)"
        return {"past_steps": [(current_step, answer)]}
    except Exception as exc:
        return {
            "past_steps": [(current_step, f"[ERROR] {exc}")],
            "error_count": state.get("error_count", 0) + 1,
        }


def _replan_node(state: PlanExecuteState) -> dict:
    original_plan = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(state["plan"]))
    completed_steps = "\n".join(f"{step} => {result}" for step, result in state["past_steps"])
    error_reason = next(
        (result for _, result in reversed(state["past_steps"]) if result.startswith("[ERROR]")),
        "unknown error",
    )
    parsed = invoke_json(
        "replan",
        [
            HumanMessage(
                content=REPLANNER_PROMPT.format(
                    context=state.get("context", ""),
                    original_plan=original_plan,
                    completed_steps=completed_steps,
                    error_reason=error_reason,
                )
            )
        ],
    )
    new_remaining = [str(item) for item in parsed] if isinstance(parsed, list) else []
    completed_count = len(state["past_steps"])
    return {"plan": state["plan"][:completed_count] + new_remaining, "error_count": 0}


def _fallback_node(state: PlanExecuteState, config: RunnableConfig | None = None) -> dict:
    react = build_react_agent("executor")
    messages = [HumanMessage(content=state["question"])]
    if state.get("context"):
        messages.insert(0, SystemMessage(content=state["context"]))
    try:
        result = react.invoke({"messages": messages}, config=config)
        answer = extract_last_ai_text(result.get("messages", [])) or "(no answer)"
    except Exception as exc:
        answer = f"fallback failed: {exc}"
    return {"response": answer, "fallback_triggered": True}


def _finalize_node(state: PlanExecuteState) -> dict:
    step_results = "\n\n".join(f"{step}\n{result}" for step, result in state.get("past_steps", []))
    answer = invoke_text(
        "finalize",
        [
            HumanMessage(
                content=FINALIZE_PROMPT.format(
                    question=state["question"],
                    context=state.get("context", ""),
                    step_results=step_results,
                )
            )
        ],
    )
    return {"response": answer}


def _after_prepare(state: PlanExecuteState) -> Literal["planner", "executor"]:
    if state.get("plan"):
        return "executor"
    return "planner"


def _after_executor(state: PlanExecuteState) -> Literal["finalize", "executor", "replan", "fallback"]:
    if state.get("error_count", 0) >= get_config().plan_execute.max_errors_before_fallback:
        return "fallback"
    past = state.get("past_steps", [])
    if past and past[-1][1].startswith("[ERROR]"):
        return "replan"
    if len(past) >= len(state.get("plan", [])):
        return "finalize"
    return "executor"


def _after_replan(state: PlanExecuteState) -> Literal["executor", "finalize", "fallback"]:
    if state.get("error_count", 0) >= get_config().plan_execute.max_errors_before_fallback:
        return "fallback"
    if len(state.get("past_steps", [])) >= len(state.get("plan", [])):
        return "finalize"
    return "executor"


def _format_similar_plans(items: list[dict]) -> str:
    if not items:
        return "None."
    lines: list[str] = []
    for index, item in enumerate(items, 1):
        lines.append(
            f"{index}. similarity={item.get('similarity', 0):.2f}, "
            f"quality={item.get('quality_score', 0):.2f}, "
            f"question={item.get('question_text', '')}"
        )
        for step in item.get("plan_steps", []):
            lines.append(f"   - {step}")
    return "\n".join(lines)


def build_plan_execute_agent():
    graph = StateGraph(PlanExecuteState)
    graph.add_node("prepare", _prepare_node)
    graph.add_node("planner", _planner_node)
    graph.add_node("executor", _executor_node)
    graph.add_node("replan", _replan_node)
    graph.add_node("fallback", _fallback_node)
    graph.add_node("finalize", _finalize_node)
    graph.set_entry_point("prepare")
    graph.add_conditional_edges("prepare", _after_prepare, {"planner": "planner", "executor": "executor"})
    graph.add_edge("planner", "executor")
    graph.add_conditional_edges(
        "executor",
        _after_executor,
        {"executor": "executor", "replan": "replan", "fallback": "fallback", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "replan",
        _after_replan,
        {"executor": "executor", "finalize": "finalize", "fallback": "fallback"},
    )
    graph.add_edge("fallback", END)
    graph.add_edge("finalize", END)
    return graph.compile()
