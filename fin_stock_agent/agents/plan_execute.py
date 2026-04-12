from __future__ import annotations

import operator
from typing import Annotated, Literal

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from fin_stock_agent.agents.react_agent import build_react_agent, extract_last_ai_text
from fin_stock_agent.core.llm import invoke_json, invoke_text
from fin_stock_agent.prompts.plan_prompt import FINALIZE_PROMPT, PLANNER_PROMPT, REPLANNER_PROMPT


class PlanExecuteState(TypedDict):
    question: str
    plan: list[str]
    past_steps: Annotated[list[tuple[str, str]], operator.add]
    response: str | None
    error_count: int
    fallback_triggered: bool


_MAX_ERRORS = 3


def _planner_node(state: PlanExecuteState) -> dict:
    parsed = invoke_json(
        "planner",
        [HumanMessage(content=PLANNER_PROMPT.format(question=state["question"]))],
    )
    plan = [str(item) for item in parsed] if isinstance(parsed, list) else []
    return {"plan": plan or [state["question"]], "past_steps": [], "error_count": 0, "fallback_triggered": False}


def _executor_node(state: PlanExecuteState, config: RunnableConfig | None = None) -> dict:
    completed = len(state.get("past_steps", []))
    if completed >= len(state.get("plan", [])):
        return {}
    current_step = state["plan"][completed]
    agent = build_react_agent("executor")
    try:
        result = agent.invoke({"messages": [HumanMessage(content=current_step)]}, config=config)
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
    try:
        result = react.invoke({"messages": [HumanMessage(content=state["question"])]}, config=config)
        answer = extract_last_ai_text(result.get("messages", [])) or "(no answer)"
    except Exception as exc:
        answer = f"fallback failed: {exc}"
    return {"response": answer, "fallback_triggered": True}


def _finalize_node(state: PlanExecuteState) -> dict:
    step_results = "\n\n".join(f"{step}\n{result}" for step, result in state.get("past_steps", []))
    answer = invoke_text(
        "finalize",
        [HumanMessage(content=FINALIZE_PROMPT.format(question=state["question"], step_results=step_results))],
    )
    return {"response": answer}


def _after_executor(state: PlanExecuteState) -> Literal["finalize", "executor", "replan", "fallback"]:
    if state.get("error_count", 0) >= _MAX_ERRORS:
        return "fallback"
    past = state.get("past_steps", [])
    if past and past[-1][1].startswith("[ERROR]"):
        return "replan"
    if len(past) >= len(state.get("plan", [])):
        return "finalize"
    return "executor"


def _after_replan(state: PlanExecuteState) -> Literal["executor", "finalize", "fallback"]:
    if state.get("error_count", 0) >= _MAX_ERRORS:
        return "fallback"
    if len(state.get("past_steps", [])) >= len(state.get("plan", [])):
        return "finalize"
    return "executor"


def build_plan_execute_agent():
    graph = StateGraph(PlanExecuteState)
    graph.add_node("planner", _planner_node)
    graph.add_node("executor", _executor_node)
    graph.add_node("replan", _replan_node)
    graph.add_node("fallback", _fallback_node)
    graph.add_node("finalize", _finalize_node)
    graph.set_entry_point("planner")
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
