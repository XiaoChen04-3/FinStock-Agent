from __future__ import annotations

import json
import uuid
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import LLMResult

from fin_stock_agent.agents.plan_execute import build_plan_execute_agent
from fin_stock_agent.agents.react_agent import build_react_agent, extract_last_ai_text
from fin_stock_agent.core.exceptions import AgentRoutingError
from fin_stock_agent.core.identity import local_profile_id
from fin_stock_agent.core.query_enhancer import EnhancedQuery, enhance_query
from fin_stock_agent.core.time_utils import now_local
from fin_stock_agent.init.name_resolver import NameResolver
from fin_stock_agent.memory.portfolio_memory import PortfolioMemory, get_active, set_active
from fin_stock_agent.services.memory_manager import MemoryManager
from fin_stock_agent.services.portfolio_service import PortfolioService
from fin_stock_agent.stats.tracker import StatsTracker
from fin_stock_agent.tools.portfolio import set_tool_user_id

_POST_TURN_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="post-turn")


class _TokenCounter(BaseCallbackHandler):
    """Accumulates token usage from every LLM call in a request.

    Works for both streaming and non-streaming paths:
    - Non-streaming: llm_output["token_usage"] is populated directly.
    - Streaming with stream_usage=True: llm_output is None but
      generation.message.usage_metadata has the counts.
    """

    def __init__(self) -> None:
        super().__init__()
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        try:
            # Path 1: non-streaming — token_usage in llm_output
            usage = (response.llm_output or {}).get("token_usage", {})
            pt = usage.get("prompt_tokens", 0)
            ct = usage.get("completion_tokens", 0)

            # Path 2: streaming with stream_usage=True — usage in message metadata
            if pt == 0 and ct == 0:
                for gen_list in (response.generations or []):
                    for gen in gen_list:
                        msg = getattr(gen, "message", None)
                        if msg is not None:
                            um = getattr(msg, "usage_metadata", None) or {}
                            pt += um.get("input_tokens", 0)
                            ct += um.get("output_tokens", 0)

            self.prompt_tokens += pt
            self.completion_tokens += ct
        except Exception:
            pass


class _ToolCapture(BaseCallbackHandler):
    """Records the name of every tool invoked during a request.

    Works for both ReAct streaming and Plan-and-Execute paths as long as
    this callback is included in the config passed to every agent invocation.
    """

    def __init__(self) -> None:
        super().__init__()
        self.tools: list[str] = []

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs) -> None:
        name = (serialized or {}).get("name") or ""
        if name:
            self.tools.append(name)


def classify_mode(enhanced: EnhancedQuery) -> str:
    return "react" if enhanced.complexity == "simple" else "plan_execute"


def _tool_desc_map() -> dict[str, str]:
    from fin_stock_agent.agents.react_agent import _all_tools

    result = {}
    for tool in _all_tools():
        result[tool.name] = (tool.description or tool.name).strip().splitlines()[0][:80]
    return result


def _summarize_result(raw: str) -> str:
    try:
        data = json.loads(raw)
        if data.get("ok") is False:
            return str(data.get("error", "tool failed"))[:100]
        if "data" in data and isinstance(data["data"], list):
            return f"returned {len(data['data'])} rows"
        if "data" in data and isinstance(data["data"], dict):
            return "returned structured result"
        if "holdings" in data:
            return f"holdings: {len(data['holdings'])}"
        return "tool finished"
    except Exception:
        return str(raw)[:100].replace("\n", " ")


def _prep_session(
    question: str,
    *,
    user_id: str,
    session_id: str,
    memory: PortfolioMemory | None,
    history_messages: list | None,
    callbacks: list | None = None,
) -> tuple[str, list, EnhancedQuery, MemoryManager]:
    active_memory = memory
    if active_memory is None or active_memory.is_empty():
        persisted_memory = PortfolioService().build_memory(user_id)
        if len(persisted_memory) > 0 or active_memory is None:
            active_memory = persisted_memory
    if active_memory is not None:
        set_active(active_memory)
    else:
        get_active()
    set_tool_user_id(user_id)
    resolver = NameResolver()
    eq = enhance_query(question, resolver=resolver, callbacks=callbacks)
    mode = classify_mode(eq)
    memory_manager = MemoryManager(user_id=user_id, session_id=session_id)
    system_context = memory_manager.build_context_block()
    messages = [
        SystemMessage(
            content="\n\n".join(
                [
                    f"Current date: {now_local().strftime('%Y-%m-%d')}",
                    system_context,
                    eq.to_context_block(),
                ]
            )
        )
    ]
    messages.extend(list(history_messages or []))
    messages.append(HumanMessage(content=eq.rewritten))
    return mode, messages, eq, memory_manager


def _persist_turn_async(
    *,
    memory_manager: MemoryManager,
    history_messages: list | None,
    question: str,
    answer: str,
    tracker: StatsTracker,
    has_error: bool,
) -> None:
    turn_idx = (len(history_messages or []) // 2) + 1

    def _task() -> None:
        if answer:
            memory_manager.after_turn(turn_idx, question, answer)
        tracker.finish(has_error=has_error, background=False)

    _POST_TURN_EXECUTOR.submit(_task)


def run_agent(
    question: str,
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    memory: PortfolioMemory | None = None,
    history_messages: list | None = None,
) -> str:
    user_id = user_id or local_profile_id()
    session_id = session_id or str(uuid.uuid4())
    mode, messages, _, memory_manager = _prep_session(
        question,
        user_id=user_id,
        session_id=session_id,
        memory=memory,
        history_messages=history_messages,
    )
    if mode == "react":
        try:
            result = build_react_agent().invoke({"messages": messages})
            answer = extract_last_ai_text(result.get("messages", [])) or "(no answer)"
            memory_manager.after_turn((len(history_messages or []) // 2) + 1, question, answer)
            return answer
        except Exception as exc:
            raise AgentRoutingError(str(exc)) from exc
    result = build_plan_execute_agent().invoke(
        {
            "question": question,
            "plan": [],
            "past_steps": [],
            "response": None,
            "error_count": 0,
            "fallback_triggered": False,
        }
    )
    answer = result.get("response") or ""
    memory_manager.after_turn((len(history_messages or []) // 2) + 1, question, answer)
    return answer


def _stream_react(
    msgs: list,
    tracker: StatsTracker,
    token_counter: _TokenCounter,
    tool_capture: _ToolCapture,
) -> Generator[tuple[str, str], None, None]:
    """Stream a ReAct agent, yielding (event_type, content) pairs.

    Token counting and tool capture are handled via callbacks; do NOT also read
    chunk.usage_metadata or you will double-count tokens.
    Tool names are collected by tool_capture and transferred to the tracker by
    the caller after the generator is exhausted.
    """
    # pending_tools maps chunk-index → tool-name for UI event ordering only
    pending_tools: dict[int, str] = {}
    desc_map = _tool_desc_map()
    try:
        agent = build_react_agent()
        _cbs = [token_counter, tool_capture]
        for chunk, _meta in agent.stream(
            {"messages": msgs},
            stream_mode="messages",
            config={"callbacks": _cbs},
        ):
            if isinstance(chunk, AIMessageChunk):
                if chunk.tool_call_chunks:
                    for tc in chunk.tool_call_chunks:
                        name = tc.get("name") or ""
                        idx: int = tc.get("index") or 0
                        if name and idx not in pending_tools:
                            pending_tools[idx] = name
                            yield ("tool_start", name)
                elif chunk.content and isinstance(chunk.content, str):
                    yield ("token", chunk.content)
            elif isinstance(chunk, ToolMessage):
                # Pop the first pending tool (sequential ReAct always calls one at a time)
                if pending_tools:
                    first_idx = min(pending_tools)
                    tool_name = pending_tools.pop(first_idx)
                else:
                    tool_name = getattr(chunk, "name", None) or "unknown"
                yield (
                    "tool_interaction",
                    json.dumps(
                        {
                            "name": tool_name,
                            "description": desc_map.get(tool_name, tool_name),
                            "summary": _summarize_result(str(chunk.content)),
                        },
                        ensure_ascii=False,
                    ),
                )
    except Exception as exc:
        yield ("error", str(exc))


def stream_agent(
    question: str,
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    memory: PortfolioMemory | None = None,
    history_messages: list | None = None,
) -> Generator[tuple[str, str], None, None]:
    user_id = user_id or local_profile_id()
    session_id = session_id or str(uuid.uuid4())
    tracker = StatsTracker(session_id=session_id, query_text=question, user_id=user_id)

    master_counter = _TokenCounter()
    tool_capture = _ToolCapture()

    try:
        mode, messages, eq, memory_manager = _prep_session(
            question,
            user_id=user_id,
            session_id=session_id,
            memory=memory,
            history_messages=history_messages,
            callbacks=[master_counter],
        )
    except Exception as exc:
        yield ("error", f"Initialization failed: {exc}")
        tracker.finish(has_error=True)
        return

    tracker.set_intent(eq.intent.value)
    tracker.set_mode(mode)
    tracker.set_query_context(
        rewritten=eq.rewritten,
        complexity=eq.complexity,
        suggested_tools=eq.suggested_tools,
        resolved_codes=eq.resolved_codes,
        history_message_count=len(history_messages or []),
    )

    yield ("mode", "ReAct" if mode == "react" else "Plan-and-Execute")
    yield ("thinking", f"意图识别: {eq.intent_label()} ({eq.intent.value})")

    # Show query rewrite when it differs from the original
    if eq.rewritten and eq.rewritten.strip() != question.strip():
        yield ("thinking", f"查询改写: {eq.rewritten}")

    if eq.resolved_codes:
        yield ("thinking", "名称映射: " + json.dumps(eq.resolved_codes, ensure_ascii=False))
    if eq.suggested_tools:
        yield ("thinking", "推荐工具: " + ", ".join(eq.suggested_tools))

    if mode == "react":
        answer_parts: list[str] = []
        had_error = False
        for event_type, content in _stream_react(
            messages, tracker, master_counter, tool_capture
        ):
            if event_type == "token":
                answer_parts.append(content)
            if event_type == "error":
                had_error = True
            yield (event_type, content)

        answer = "".join(answer_parts)
        tracker.data.prompt_tokens = master_counter.prompt_tokens
        tracker.data.completion_tokens = master_counter.completion_tokens
        for name in tool_capture.tools:
            tracker.add_tool(name)

        # Summary thinking line
        p, c = master_counter.prompt_tokens, master_counter.completion_tokens
        if p or c:
            yield ("thinking", f"Token 消耗: {p} prompt + {c} completion = {p + c} 合计")

        if answer:
            yield ("answer", answer)
        _persist_turn_async(
            memory_manager=memory_manager,
            history_messages=history_messages,
            question=question,
            answer=answer,
            tracker=tracker,
            has_error=had_error,
        )
        return

    try:
        graph = build_plan_execute_agent()
        final_answer = ""
        for event in graph.stream(
            {
                "question": question,
                "plan": [],
                "past_steps": [],
                "response": None,
                "error_count": 0,
                "fallback_triggered": False,
            },
            stream_mode="updates",
            config={"callbacks": [master_counter, tool_capture]},
        ):
            if "planner" in event:
                plan = event["planner"].get("plan") or []
                for idx, step in enumerate(plan, 1):
                    yield ("thinking", f"步骤 {idx}: {step}")
            elif "executor" in event:
                past_steps = event["executor"].get("past_steps") or []
                if past_steps:
                    step, result = past_steps[-1]
                    yield ("thinking", f"{step} => {str(result)[:120]}")
            elif "replan" in event:
                yield ("thinking", "正在重新规划剩余步骤")
            elif "fallback" in event:
                final_answer = event["fallback"].get("response") or ""
            elif "finalize" in event:
                final_answer = event["finalize"].get("response") or ""

        tracker.data.prompt_tokens = master_counter.prompt_tokens
        tracker.data.completion_tokens = master_counter.completion_tokens
        for name in tool_capture.tools:
            tracker.add_tool(name)

        p, c = master_counter.prompt_tokens, master_counter.completion_tokens
        if p or c:
            yield ("thinking", f"Token 消耗: {p} prompt + {c} completion = {p + c} 合计")

        if final_answer:
            yield ("answer", final_answer)
            _persist_turn_async(
                memory_manager=memory_manager,
                history_messages=history_messages,
                question=question,
                answer=final_answer,
                tracker=tracker,
                has_error=False,
            )
        else:
            yield ("error", "Plan-and-Execute returned no final answer")
            tracker.finish(has_error=True)
    except Exception as exc:
        tracker.data.prompt_tokens = master_counter.prompt_tokens
        tracker.data.completion_tokens = master_counter.completion_tokens
        for name in tool_capture.tools:
            tracker.add_tool(name)
        yield ("error", f"Plan-and-Execute failed: {exc}")
        tracker.finish(has_error=True)
