from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from fin_stock_agent.core import llm


class _StreamingLLM:
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    def stream(self, messages, config=None):  # noqa: ANN001
        for chunk in self._chunks:
            yield SimpleNamespace(content=chunk)


class _InvokeLLM:
    def __init__(self, content: str) -> None:
        self._content = content

    def invoke(self, messages, config=None):  # noqa: ANN001
        return SimpleNamespace(content=self._content)


class _BoundRepairer:
    def __init__(self, payload: str) -> None:
        self._payload = payload

    def invoke(self, messages, config=None):  # noqa: ANN001
        return SimpleNamespace(content=self._payload)


def test_invoke_text_streams_for_thinking_roles(monkeypatch) -> None:
    monkeypatch.setattr(llm, "get_llm", lambda role: _StreamingLLM(['["step', ' one"]']))

    text = llm.invoke_text("planner", [HumanMessage(content="plan this")])

    assert text == '["step one"]'


def test_invoke_json_repairs_non_standard_thinking_output(monkeypatch) -> None:
    monkeypatch.setattr(llm, "invoke_text", lambda role, messages, config=None: "答案如下：['a', 'b']")
    monkeypatch.setattr(
        llm,
        "get_llm",
        lambda role: SimpleNamespace(
            bind=lambda **kwargs: _BoundRepairer('{"payload":["a","b"]}')
        ),
    )

    data = llm.invoke_json("planner", [HumanMessage(content="plan this")])

    assert data == ["a", "b"]


def test_merge_token_usage_collects_message_metadata() -> None:
    usage = llm.merge_token_usage(
        SimpleNamespace(usage_metadata={"input_tokens": 12, "output_tokens": 8}),
        [SimpleNamespace(response_metadata={"token_usage": {"prompt_tokens": 3, "completion_tokens": 2}})],
    )

    assert usage == {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25}
