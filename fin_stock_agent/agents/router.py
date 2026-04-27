from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Generator
from concurrent.futures import Future, ThreadPoolExecutor, wait
from threading import Lock

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import LLMResult

from fin_stock_agent.agents.plan_execute import build_plan_execute_agent
from fin_stock_agent.agents.react_agent import build_react_agent, extract_last_ai_text
from fin_stock_agent.core.config import get_config
from fin_stock_agent.core.exceptions import AgentRoutingError
from fin_stock_agent.core.identity import local_profile_id
from fin_stock_agent.core.query_enhancer import EnhancedQuery, enhance_query
from fin_stock_agent.core.time_utils import now_local
from fin_stock_agent.init.name_resolver import NameResolver
from fin_stock_agent.memory.portfolio_memory import PortfolioMemory, get_active, set_active
from fin_stock_agent.services.memory_manager import MemoryManager
from fin_stock_agent.services.plan_library_service import PlanLibraryService
from fin_stock_agent.services.portfolio_service import PortfolioService
from fin_stock_agent.stats.tracker import StatsTracker
from fin_stock_agent.tools.portfolio import set_tool_user_id

logger = logging.getLogger(__name__)
_POST_TURN_EXECUTOR = ThreadPoolExecutor(
    max_workers=get_config().concurrency.post_turn_workers,
    thread_name_prefix="post-turn",
)
_POST_TURN_FUTURES: list[Future] = []
_POST_TURN_LOCK = Lock()


def _submit_post_turn_task(fn) -> Future:
    future = _POST_TURN_EXECUTOR.submit(fn)
    with _POST_TURN_LOCK:
        _POST_TURN_FUTURES[:] = [item for item in _POST_TURN_FUTURES if not item.done()]
        _POST_TURN_FUTURES.append(future)
    return future


def flush_post_turn_tasks(timeout: float | None = None) -> None:
    with _POST_TURN_LOCK:
        futures = list(_POST_TURN_FUTURES)
    if not futures:
        return
    wait(futures, timeout=timeout)
    with _POST_TURN_LOCK:
        _POST_TURN_FUTURES[:] = [item for item in _POST_TURN_FUTURES if not item.done()]


class _TokenCounter(BaseCallbackHandler):
    """Accumulates token usage from every LLM call in a request."""

    def __init__(self) -> None:
        super().__init__()
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        try:
            usage = (response.llm_output or {}).get("token_usage", {})
            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            if prompt_tokens == 0 and completion_tokens == 0:
                for gen_list in response.generations or []:
                    for gen in gen_list:
                        message = getattr(gen, "message", None)
                        metadata = getattr(message, "usage_metadata", None) or {}
                        prompt_tokens += int(metadata.get("input_tokens", 0) or 0)
                        completion_tokens += int(metadata.get("output_tokens", 0) or 0)
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
        except Exception as exc:
            logger.debug("TokenCounter: failed to parse LLM token usage: %s", exc)


class _ToolCapture(BaseCallbackHandler):
    """Records the name of every tool invoked during a request."""

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
    try:
        eq = enhance_query(question, resolver=resolver, callbacks=callbacks)
    except Exception as exc:
        logger.warning("Query enhancement failed, using raw question: %s", exc)
        eq = EnhancedQuery(original=question, rewritten=question)
    mode = classify_mode(eq)
    memory_manager = MemoryManager(user_id=user_id, session_id=session_id)
    system_context = memory_manager.build_context_block(eq.rewritten or question)
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
    messages.append(HumanMessage(content=eq.rewritten or question))
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

    _submit_post_turn_task(_task)


def _persist_memory_async(
    *,
    memory_manager: MemoryManager,
    history_messages: list | None,
    question: str,
    answer: str,
) -> None:
    turn_idx = (len(history_messages or []) // 2) + 1

    def _task() -> None:
        if answer:
            memory_manager.after_turn(turn_idx, question, answer)

    _submit_post_turn_task(_task)


def _build_plan_execute_seed(
    *,
    user_id: str,
    question: str,
    eq: EnhancedQuery,
    memory_manager: MemoryManager,
) -> dict:
    cfg = get_config()
    plan_service = PlanLibraryService()
    rewritten_question = eq.rewritten or question
    context_parts = [memory_manager.build_prompt_memory_block(rewritten_question), eq.to_context_block()]
    context_block = "\n\n".join(part for part in context_parts if part).strip()
    context_block = context_block[: cfg.plan_execute.context_max_chars]

    candidates = plan_service.search_plans(user_id, rewritten_question, top_k=3)
    similar_plans: list[dict] = []
    reusable_plan: list[str] = []
    if candidates:
        best = candidates[0]
        if best.similarity >= cfg.memory.plan_library.reuse_threshold and best.plan_steps:
            reusable_plan = list(best.plan_steps)
            similar_plans = [_candidate_to_dict(best)]
        elif best.similarity >= cfg.memory.plan_library.reference_threshold:
            similar_plans = [_candidate_to_dict(item) for item in candidates]

    return {
        "question": rewritten_question,
        "context": context_block,
        "plan": reusable_plan,
        "past_steps": [],
        "response": None,
        "error_count": 0,
        "fallback_triggered": False,
        "similar_plans": similar_plans,
    }


def _candidate_to_dict(candidate) -> dict:
    return {
        "plan_steps": list(candidate.plan_steps),
        "quality_score": candidate.quality_score,
        "similarity": candidate.similarity,
        "question_text": candidate.question_text,
    }


def _maybe_persist_plan_async(user_id: str, question_text: str, state: dict) -> None:
    if not question_text.strip():
        return

    def _task() -> None:
        score = _score_plan_quality(state)
        if score < get_config().memory.plan_library.min_quality_score:
            return
        try:
            PlanLibraryService().save_plan(user_id, question_text, list(state.get("plan") or []), score)
        except Exception as exc:
            logger.warning("Plan library save failed: %s", exc)

    _submit_post_turn_task(_task)


def _score_plan_quality(state: dict) -> float:
    response = str(state.get("response") or "").strip()
    if not response:
        return 0.0
    score = 1.0
    if state.get("fallback_triggered"):
        score -= 0.5
    error_steps = [item for item in state.get("past_steps", []) if str(item[1]).startswith("[ERROR]")]
    if len(error_steps) > 1:
        score -= 0.3
    if state.get("error_count", 0) > 0:
        score -= 0.2
    return max(0.0, min(1.0, score))


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
    mode, messages, eq, memory_manager = _prep_session(
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
            _persist_memory_async(
                memory_manager=memory_manager,
                history_messages=history_messages,
                question=question,
                answer=answer,
            )
            return answer
        except Exception as exc:
            raise AgentRoutingError(str(exc)) from exc

    seed = _build_plan_execute_seed(user_id=user_id, question=question, eq=eq, memory_manager=memory_manager)
    result = build_plan_execute_agent().invoke(seed)
    answer = result.get("response") or ""
    _maybe_persist_plan_async(user_id, seed["question"], result)
    _persist_memory_async(
        memory_manager=memory_manager,
        history_messages=history_messages,
        question=question,
        answer=answer,
    )
    return answer


def _stream_react(
    msgs: list,
    tracker: StatsTracker,
    token_counter: _TokenCounter,
    tool_capture: _ToolCapture,
) -> Generator[tuple[str, str], None, None]:
    pending_tools: dict[int, str] = {}
    desc_map = _tool_desc_map()
    try:
        agent = build_react_agent()
        callbacks = [token_counter, tool_capture]
        for chunk, _meta in agent.stream(
            {"messages": msgs},
            stream_mode="messages",
            config={"callbacks": callbacks},
        ):
            if isinstance(chunk, AIMessageChunk):
                if chunk.tool_call_chunks:
                    for tc in chunk.tool_call_chunks:
                        name = tc.get("name") or ""
                        idx = int(tc.get("index") or 0)
                        if name and idx not in pending_tools:
                            pending_tools[idx] = name
                            yield ("tool_start", name)
                elif chunk.content and isinstance(chunk.content, str):
                    yield ("token", chunk.content)
            elif isinstance(chunk, ToolMessage):
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
        yield ("error", f"初始化失败：{exc}")
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

    yield ("mode", "ReAct" if mode == "react" else "规划执行")
    yield ("thinking", f"意图识别：{eq.intent_label()}（{eq.intent.value}）")
    if eq.rewritten and eq.rewritten.strip() != question.strip():
        yield ("thinking", f"问题改写：{eq.rewritten}")
    if eq.resolved_codes:
        yield ("thinking", "代码解析：" + json.dumps(eq.resolved_codes, ensure_ascii=False))
    if eq.suggested_tools:
        yield ("thinking", "建议工具：" + "、".join(eq.suggested_tools))

    if mode == "react":
        answer_parts: list[str] = []
        had_error = False
        for event_type, content in _stream_react(messages, tracker, master_counter, tool_capture):
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
        if tracker.data.prompt_tokens or tracker.data.completion_tokens:
            total = tracker.data.prompt_tokens + tracker.data.completion_tokens
            yield (
                "thinking",
                f"Token 消耗：输入 {tracker.data.prompt_tokens} + 输出 {tracker.data.completion_tokens} = 合计 {total}",
            )
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
        seed = _build_plan_execute_seed(user_id=user_id, question=question, eq=eq, memory_manager=memory_manager)
        if seed["plan"]:
            yield ("thinking", "命中高相似历史计划，直接复用。")
            for index, step in enumerate(seed["plan"], 1):
                yield ("thinking", f"步骤 {index}：{step}")
        elif seed["similar_plans"]:
            yield ("thinking", "规划器已获取相似历史计划作为参考。")

        final_answer = ""
        final_state: dict = dict(seed)
        for event in graph.stream(seed, stream_mode="updates", config={"callbacks": [master_counter, tool_capture]}):
            for node_name, payload in event.items():
                if isinstance(payload, dict):
                    final_state.update(payload)
                if node_name == "planner":
                    plan = payload.get("plan") or []
                    for index, step in enumerate(plan, 1):
                        yield ("thinking", f"步骤 {index}：{step}")
                elif node_name == "executor":
                    past_steps = payload.get("past_steps") or []
                    if past_steps:
                        step, result = past_steps[-1]
                        yield ("thinking", f"{step} => {str(result)[:120]}")
                elif node_name == "replan":
                    yield ("thinking", "已重新规划剩余步骤。")
                elif node_name == "fallback":
                    final_answer = payload.get("response") or ""
                elif node_name == "finalize":
                    final_answer = payload.get("response") or ""

        tracker.data.prompt_tokens = master_counter.prompt_tokens
        tracker.data.completion_tokens = master_counter.completion_tokens
        for name in tool_capture.tools:
            tracker.add_tool(name)
        if tracker.data.prompt_tokens or tracker.data.completion_tokens:
            total = tracker.data.prompt_tokens + tracker.data.completion_tokens
            yield (
                "thinking",
                f"Token 消耗：输入 {tracker.data.prompt_tokens} + 输出 {tracker.data.completion_tokens} = 合计 {total}",
            )

        if final_answer:
            final_state["response"] = final_answer
            yield ("answer", final_answer)
            _maybe_persist_plan_async(user_id, seed["question"], final_state)
            _persist_turn_async(
                memory_manager=memory_manager,
                history_messages=history_messages,
                question=question,
                answer=final_answer,
                tracker=tracker,
                has_error=False,
            )
        else:
            yield ("error", "规划执行模式未生成最终答案")
            tracker.finish(has_error=True)
    except Exception as exc:
        tracker.data.prompt_tokens = master_counter.prompt_tokens
        tracker.data.completion_tokens = master_counter.completion_tokens
        for name in tool_capture.tools:
            tracker.add_tool(name)
        yield ("error", f"规划执行失败：{exc}")
        tracker.finish(has_error=True)
