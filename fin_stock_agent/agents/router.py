"""Router: classify question complexity and dispatch to the right agent."""
from __future__ import annotations

import json
import re
from collections.abc import Generator
from typing import Literal

from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage
from pydantic import BaseModel

from fin_stock_agent.agents.react_agent import build_react_agent, extract_last_ai_text
from fin_stock_agent.core.exceptions import AgentRoutingError
from fin_stock_agent.core.llm import get_llm
from fin_stock_agent.core.query_enhancer import EnhancedQuery, enhance_query
from fin_stock_agent.memory.portfolio_memory import (
    PortfolioMemory,
    TradeRecord,
    get_active,
    set_active,
)
from fin_stock_agent.prompts.extraction import TRADE_EXTRACTION_PROMPT

# ---------------------------------------------------------------------------
# Complexity classification
# ---------------------------------------------------------------------------

_COMPLEXITY_PROMPT = """你是一个问题复杂度分类器。

判断下面的用户问题应该用哪种 Agent 模式：
- "react"：单一意图、单步数据查询（如查一只股票行情、问指数今日涨跌、记录一笔交易）。
- "plan_execute"：多意图、需要多步骤拆解（如「分析我全部持仓的盈亏并对比大盘」「查多只股票并给出对比建议」「今天哪个板块强、我的持仓有没有相关股票」）。

只输出一个 JSON，格式：{{"mode": "react"}} 或 {{"mode": "plan_execute"}}
不要输出其他任何内容。

用户问题：{question}"""


class _ComplexityResult(BaseModel):
    mode: Literal["react", "plan_execute"]


def classify_complexity(question: str) -> Literal["react", "plan_execute"]:
    """Use a lightweight LLM call to determine the agent mode."""
    # Fast heuristic first: very short questions almost always → react
    if len(question.strip()) < 30 and "\n" not in question:
        return "react"

    try:
        llm = get_llm()
        prompt = _COMPLEXITY_PROMPT.format(question=question)
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content if isinstance(response.content, str) else ""
        # Extract JSON
        m = re.search(r'\{[^}]+\}', raw)
        if m:
            data = json.loads(m.group())
            result = _ComplexityResult(**data)
            return result.mode
    except Exception:
        pass
    # Default to react on any failure
    return "react"


# ---------------------------------------------------------------------------
# Trade extraction
# ---------------------------------------------------------------------------

def _extract_trades(message: str) -> list[TradeRecord]:
    """Run extraction prompt; silently returns [] on failure."""
    try:
        llm = get_llm()
        prompt = TRADE_EXTRACTION_PROMPT.format(message=message)
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content if isinstance(response.content, str) else ""
        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end == -1:
            return []
        records_raw = json.loads(raw[start: end + 1])
        results: list[TradeRecord] = []
        for r in records_raw:
            try:
                results.append(TradeRecord(**r))
            except Exception:
                pass
        return results
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_agent(
    question: str,
    memory: PortfolioMemory | None = None,
    history_messages: list | None = None,
) -> str:
    """
    Primary entry point called by app_streamlit.py.

    Steps:
    1. Activate memory, extract trades from the message.
    2. Run query enhancement (intent recognition + rewrite + multi-query expansion).
    3. Classify routing complexity based on the enhanced question.
    4. Dispatch to ReAct or Plan-and-Execute; return final text answer.
    """
    # 1–3. Shared pre-processing (memory, trade extraction, enhancement, routing)
    mode, msgs, _eq = _prep_session(question, memory, history_messages)

    if mode == "react":
        try:
            agent = build_react_agent()
            result = agent.invoke({"messages": msgs})
            return extract_last_ai_text(result.get("messages", [])) or "（模型未返回文本）"
        except Exception as e:
            raise AgentRoutingError(f"ReAct agent 调用失败: {e}") from e

    else:  # plan_execute
        try:
            from fin_stock_agent.agents.plan_execute import build_plan_execute_agent

            agent = build_plan_execute_agent()
            result = agent.invoke(
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
            if not answer:
                raise ValueError("P&E graph returned empty response")
            return answer
        except AgentRoutingError:
            raise
        except Exception as e:
            # Final safety net: fall back to ReAct
            try:
                agent = build_react_agent()
                result = agent.invoke({"messages": msgs})
                fallback_text = extract_last_ai_text(result.get("messages", [])) or "（无返回文本）"
                return f"[自动降级至 ReAct 模式]\n\n{fallback_text}"
            except Exception as e2:
                raise AgentRoutingError(
                    f"Plan-and-Execute 失败 ({e})，ReAct 降级也失败 ({e2})"
                ) from e2


# ---------------------------------------------------------------------------
# Streaming entry point
# ---------------------------------------------------------------------------

# Event type aliases for UI consumption:
#   "mode"        – which agent mode was chosen ("ReAct" | "Plan-and-Execute")
#   "tool_call"   – a tool is being called; content = "tool_name(arg_summary)"
#   "tool_result" – tool returned; content = first 200 chars of result
#   "thinking"    – PnE planning / execution progress line
#   "token"       – a streamed text token from the final ReAct answer
#   "answer"      – complete final answer (PnE or ReAct fallback)
#   "error"       – unrecoverable error message


def _prep_session(
    question: str,
    memory: PortfolioMemory | None,
    history_messages: list | None,
) -> tuple[str, list, EnhancedQuery]:
    """
    Shared pre-processing:
      1. Activate memory.
      2. Extract trades from the raw message.
      3. Run query enhancement (intent recognition + rewrite + multi-query expansion).
      4. Classify routing complexity based on the *enhanced* question.
      5. Build the messages list with the enriched context block.

    Returns (mode, msgs, enhanced_query).
    """
    if memory is not None:
        set_active(memory)
    active_mem = get_active()

    # Extract trades from the original message before any rewriting
    for t in _extract_trades(question):
        active_mem.add(t)

    # Query enhancement
    eq = enhance_query(question)

    # Use rewritten question for routing complexity classification
    mode = classify_complexity(eq.rewritten)

    # Build enriched human message: rewritten question + sub-query hints
    enriched_content = eq.to_context_block()
    msgs = list(history_messages or []) + [HumanMessage(content=enriched_content)]

    return mode, msgs, eq


def _stream_react(
    msgs: list,
) -> Generator[tuple[str, str], None, str]:
    """
    Stream events from the ReAct agent.
    Returns the complete answer text as the generator's return value.
    Callers that only iterate will receive all ("token", ...) events.
    """
    agent = build_react_agent()
    tokens: list[str] = []
    try:
        for chunk, _meta in agent.stream(
            {"messages": msgs}, stream_mode="messages"
        ):
            if isinstance(chunk, AIMessageChunk):
                # Tool-call decision: accumulate tool_call_chunks until complete
                if chunk.tool_call_chunks:
                    for tc in chunk.tool_call_chunks:
                        name = tc.get("name") or ""
                        if name:
                            args_raw = tc.get("args") or ""
                            # args may arrive in multiple chunks; show name immediately
                            yield ("tool_call", f"{name}({str(args_raw)[:80]}…)")
                elif chunk.content:
                    text = chunk.content
                    if isinstance(text, str) and text:
                        tokens.append(text)
                        yield ("token", text)
            elif isinstance(chunk, ToolMessage):
                summary = str(chunk.content)[:200]
                yield ("tool_result", summary)
    except Exception as e:
        yield ("error", str(e))

    return "".join(tokens)


def stream_agent(
    question: str,
    memory: PortfolioMemory | None = None,
    history_messages: list | None = None,
) -> Generator[tuple[str, str], None, None]:
    """
    Streaming entry point for app_streamlit.py.

    Yields (event_type, content) tuples so the UI can render thinking steps
    and the final answer incrementally.
    """
    try:
        mode_key, msgs, eq = _prep_session(question, memory, history_messages)
    except Exception as e:
        yield ("error", f"初始化失败: {e}")
        return

    mode_label = "ReAct" if mode_key == "react" else "Plan-and-Execute"
    yield ("mode", mode_label)

    # Emit intent recognition result for UI display
    yield ("thinking", f"意图识别：{eq.intent_label()}（{eq.intent.value}）")
    if eq.rewritten != question:
        yield ("thinking", f"查询重写：{eq.rewritten}")
    if eq.suggested_tools:
        yield ("thinking", "推荐工具：" + "、".join(eq.suggested_tools))
    if eq.sub_queries:
        yield ("thinking", f"多角度查询扩展（{len(eq.sub_queries)} 条）："
               + " | ".join(eq.sub_queries))

    # ---- ReAct path --------------------------------------------------------
    if mode_key == "react":
        tokens: list[str] = []
        try:
            agent = build_react_agent()
            for chunk, _meta in agent.stream(
                {"messages": msgs}, stream_mode="messages"
            ):
                if isinstance(chunk, AIMessageChunk):
                    if chunk.tool_call_chunks:
                        for tc in chunk.tool_call_chunks:
                            name = tc.get("name") or ""
                            if name:
                                args_raw = str(tc.get("args") or "")[:80]
                                yield ("tool_call", f"{name}({args_raw}…)")
                    elif chunk.content:
                        text = chunk.content
                        if isinstance(text, str) and text:
                            tokens.append(text)
                            yield ("token", text)
                elif isinstance(chunk, ToolMessage):
                    yield ("tool_result", str(chunk.content)[:200])
        except Exception as e:
            # Try non-streaming fallback via invoke
            try:
                agent2 = build_react_agent()
                result = agent2.invoke({"messages": msgs})
                fallback = extract_last_ai_text(result.get("messages", [])) or "（无返回文本）"
                yield ("answer", f"[流式失败，完整回答]\n\n{fallback}")
            except Exception as e2:
                yield ("error", f"ReAct 失败: {e}; 降级也失败: {e2}")
        return

    # ---- Plan-and-Execute path ---------------------------------------------
    from fin_stock_agent.agents.plan_execute import build_plan_execute_agent

    initial_state = {
        "question": question,
        "plan": [],
        "past_steps": [],
        "response": None,
        "error_count": 0,
        "fallback_triggered": False,
    }

    try:
        pne_agent = build_plan_execute_agent()
        final_answer: str = ""

        for event in pne_agent.stream(initial_state, stream_mode="updates"):
            # planner node: show the generated step list
            if "planner" in event:
                plan = event["planner"].get("plan") or []
                if plan:
                    yield ("thinking", f"已生成 {len(plan)} 步计划：")
                    for i, step in enumerate(plan, 1):
                        yield ("thinking", f"  步骤 {i}：{step}")

            # executor node: show latest completed step
            elif "executor" in event:
                past = event["executor"].get("past_steps") or []
                if past:
                    step, result_text = past[-1]
                    short = str(result_text)[:120].replace("\n", " ")
                    yield ("thinking", f"✓ {step}  →  {short}…")

            # replan node
            elif "replan" in event:
                new_plan = event["replan"].get("plan") or []
                yield ("thinking", f"重新规划，剩余 {len(new_plan)} 步")

            # fallback node
            elif "fallback" in event:
                resp = (event["fallback"] or {}).get("response") or ""
                if resp:
                    final_answer = resp
                    yield ("thinking", "[已降级至 ReAct]")

            # finalize node
            elif "finalize" in event:
                resp = (event["finalize"] or {}).get("response") or ""
                if resp:
                    final_answer = resp

        if final_answer:
            yield ("answer", final_answer)
        else:
            yield ("error", "Plan-and-Execute 未产生最终回答")

    except Exception as e:
        # PnE failed entirely – stream via ReAct as fallback
        yield ("thinking", f"[P&E 失败 ({e})，降级至 ReAct]")
        try:
            agent = build_react_agent()
            tokens: list[str] = []
            for chunk, _meta in agent.stream(
                {"messages": msgs}, stream_mode="messages"
            ):
                if isinstance(chunk, AIMessageChunk):
                    if chunk.tool_call_chunks:
                        for tc in chunk.tool_call_chunks:
                            name = tc.get("name") or ""
                            if name:
                                yield ("tool_call", f"{name}(…)")
                    elif chunk.content and isinstance(chunk.content, str):
                        tokens.append(chunk.content)
                        yield ("token", chunk.content)
                elif isinstance(chunk, ToolMessage):
                    yield ("tool_result", str(chunk.content)[:200])
        except Exception as e2:
            yield ("error", f"P&E 降级也失败: {e2}")
