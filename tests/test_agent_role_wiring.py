from __future__ import annotations

from types import SimpleNamespace

from fin_stock_agent.agents import plan_execute, react_agent
from fin_stock_agent.core import query_enhancer


class _FakeLLM:
    def __init__(self, content: str) -> None:
        self.content = content

    def invoke(self, messages, config=None):  # noqa: ANN001
        return SimpleNamespace(content=self.content)


def test_build_react_agent_uses_requested_role(monkeypatch) -> None:
    seen: list[str] = []

    monkeypatch.setattr(react_agent, "get_llm", lambda role: seen.append(role) or f"llm:{role}")
    monkeypatch.setattr(react_agent, "_all_tools", lambda: ["tool-a"])
    monkeypatch.setattr(
        react_agent,
        "create_react_agent",
        lambda llm, tools, prompt=None: {"llm": llm, "tools": tools, "prompt": prompt},
    )

    built = react_agent.build_react_agent("executor")

    assert built["llm"] == "llm:executor"
    assert built["tools"] == ["tool-a"]
    assert seen == ["executor"]


def test_query_enhancer_uses_query_enhancer_role(monkeypatch) -> None:
    seen: list[str] = []

    class _Resolver:
        def search(self, question: str, top_k: int = 5):  # noqa: ARG002
            return [{"name": "沪深300", "ts_code": "000300.SH"}]

    monkeypatch.setattr(
        query_enhancer,
        "get_llm",
        lambda role, temperature=None: seen.append(role)
        or _FakeLLM(
            '{"intent":"index_price","rewritten":"沪深300行情","resolved_codes":{"沪深300":"000300.SH"},'
            '"sub_queries":[],"suggested_tools":["index"],"keywords":["沪深300"],"complexity":"simple"}'
        ),
    )

    enhanced = query_enhancer.enhance_query("沪深300怎么样", resolver=_Resolver())

    assert enhanced.intent.value == "index_price"
    assert enhanced.rewritten == "沪深300行情"
    assert seen == ["query_enhancer"]


def test_plan_execute_nodes_use_configured_roles(monkeypatch) -> None:
    seen: list[str] = []

    def fake_invoke_json(role: str, messages, config=None):  # noqa: ARG001
        seen.append(role)
        if role == "planner":
            return ["step one"]
        if role == "replan":
            return ["step two"]
        raise AssertionError(f"unexpected role {role}")

    def fake_invoke_text(role: str, messages, config=None):  # noqa: ARG001
        seen.append(role)
        if role == "finalize":
            return "final answer"
        raise AssertionError(f"unexpected role {role}")

    monkeypatch.setattr(plan_execute, "invoke_json", fake_invoke_json)
    monkeypatch.setattr(plan_execute, "invoke_text", fake_invoke_text)

    planned = plan_execute._planner_node(
        {
            "question": "analyze",
            "plan": [],
            "past_steps": [],
            "response": None,
            "error_count": 0,
            "fallback_triggered": False,
        }
    )
    replanned = plan_execute._replan_node(
        {
            "question": "analyze",
            "plan": ["step one"],
            "past_steps": [("step one", "[ERROR] bad tool")],
            "response": None,
            "error_count": 1,
            "fallback_triggered": False,
        }
    )
    finalized = plan_execute._finalize_node(
        {
            "question": "analyze",
            "plan": ["step one"],
            "past_steps": [("step one", "done")],
            "response": None,
            "error_count": 0,
            "fallback_triggered": False,
        }
    )

    assert planned["plan"] == ["step one"]
    assert replanned["plan"] == ["step one", "step two"]
    assert finalized["response"] == "final answer"
    assert seen == ["planner", "replan", "finalize"]
