"""Router: classify question complexity and dispatch to the right agent."""
from __future__ import annotations

import json
import re
from collections.abc import Generator
from datetime import datetime, timedelta
from functools import lru_cache
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
    tushare_score: int = 0,
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
            agent = build_react_agent(tushare_score)
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
                agent = build_react_agent(tushare_score)
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

    # Always inject current date so the LLM never has to guess relative times
    _now = datetime.now()
    _wd = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][_now.weekday()]
    _recent_7 = [(_now - timedelta(days=i)).strftime("%Y%m%d") for i in range(7)]
    _date_ctx = (
        f"[系统时间] 今日 {_now.strftime('%Y-%m-%d')} {_wd}，"
        f"YYYYMMDD={_now.strftime('%Y%m%d')}。"
        f"近7日日期列表（含今日）：{', '.join(_recent_7)}。"
        f"A股交易日为周一至周五（法定节假日除外），若某日无行情数据请向前取最近交易日。"
    )

    # Build enriched human message: date context + rewritten question + sub-query hints
    enriched_content = _date_ctx + "\n\n" + eq.to_context_block()
    msgs = list(history_messages or []) + [HumanMessage(content=enriched_content)]

    return mode, msgs, eq


# ---------------------------------------------------------------------------
# Tool description + result summary helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _tool_desc_map() -> dict[str, str]:
    """Lazily build {tool_name: first-sentence description} from all registered tools."""
    try:
        from fin_stock_agent.agents.react_agent import _all_tools  # local import avoids cycles
        result: dict[str, str] = {}
        for t in _all_tools():
            desc = (t.description or "").strip()
            for sep in ("。", "；", "（", "\n"):
                idx = desc.find(sep)
                if 0 < idx < 60:
                    desc = desc[:idx]
                    break
            result[t.name] = desc[:60]
        return result
    except Exception:
        return {}


def _summarize_result(raw: str) -> str:
    """Parse a tool's JSON output and return a concise Chinese summary."""
    try:
        data = json.loads(raw)
    except Exception:
        cleaned = raw.strip()[:120].replace("\n", " ")
        return cleaned or "(无返回内容)"

    # Explicit error
    if data.get("ok") is False:
        return f"❌ {str(data.get('error', '未知错误'))[:80]}"

    # get_current_datetime
    if "today_display" in data:
        return (
            f"今日 {data.get('today_display', '')} "
            f"{data.get('weekday', '')}  {data.get('time', '')}"
        )

    # get_major_indices_performance
    if "indices" in data:
        items = data["indices"]
        n = len(items)
        parts = [f"{x.get('name','')} {x.get('pct_chg_period','')}%" for x in items[:3] if x.get("name")]
        return f"{n} 个指数区间涨跌：" + "、".join(parts)

    # get_sw_industry_top_movers
    if "top_gainers" in data or "top_losers" in data:
        td = data.get("trade_date", "")
        g = data.get("top_gainers", [])
        lo = data.get("top_losers", [])
        g_names = [x.get("industry_name", x.get("ts_code", "")) for x in g[:3]]
        l_names = [x.get("industry_name", x.get("ts_code", "")) for x in lo[:3]]
        return (
            f"{td} 涨幅前3：{'、'.join(g_names)}；"
            f"跌幅前3：{'、'.join(l_names)}"
        )

    # calculate_portfolio_pnl
    if "positions" in data:
        n = len(data.get("positions", []))
        fl = data.get("floating_pnl_total", "?")
        rl = data.get("realized_pnl_total", "?")
        return f"共 {n} 个持仓，浮动盈亏 {fl}，已实现盈亏 {rl}"

    # get_portfolio_positions / memory_tools plain dicts
    if "trades" in data or "holdings" in data:
        n = len(data.get("holdings") or data.get("trades") or [])
        return f"持仓记录 {n} 条"

    # Standard _df_to_payload format
    rows = data.get("rows")
    note = data.get("note", "")
    if rows is not None:
        if rows == 0:
            return note or "无数据"
        trunc = "（部分）" if data.get("truncated") else ""
        summary = f"共 {rows} 条{trunc}"
        records = data.get("data") or []
        if records and isinstance(records, list):
            first = records[0]
            if "trade_date" in first:
                dates = sorted(
                    str(r["trade_date"]) for r in records if r.get("trade_date")
                )
                if dates:
                    span = f"{dates[0]}~{dates[-1]}" if dates[0] != dates[-1] else dates[0]
                    summary += f"，时间 {span}"
            elif "name" in first:
                names = [str(r["name"]) for r in records[:3] if r.get("name")]
                if names:
                    summary += f"，含 {'、'.join(names)}"
            elif "concept_name" in first:
                names = [str(r["concept_name"]) for r in records[:3] if r.get("concept_name")]
                if names:
                    summary += f"，含 {'、'.join(names)}"
        return summary

    # Generic success
    return "✅ 数据已返回"


# ---------------------------------------------------------------------------
# ReAct streaming helper
# ---------------------------------------------------------------------------

def _stream_react(msgs: list, score: int = 0) -> Generator[tuple[str, str], None, None]:
    """
    Stream a ReAct agent execution, yielding structured events.

    Event types emitted:
      "tool_start"       – tool call has been decided; content = tool name
      "tool_interaction" – tool finished; content = JSON {name, description, summary}
      "token"            – final answer text fragment
      "error"            – unrecoverable error message

    On streaming failure, yields ("error", message) and stops.
    Does NOT yield raw "tool_call" / "tool_result" events any more.
    """
    agent = build_react_agent(score)
    pending_tool_name: str = ""
    try:
        for chunk, _meta in agent.stream(
            {"messages": msgs},
            config={"recursion_limit": 50},
            stream_mode="messages",
        ):
            if isinstance(chunk, AIMessageChunk):
                if chunk.tool_call_chunks:
                    for tc in chunk.tool_call_chunks:
                        name = tc.get("name") or ""
                        # Only emit tool_start on the first chunk that carries the name
                        if name and not pending_tool_name:
                            pending_tool_name = name
                            yield ("tool_start", name)
                elif chunk.content:
                    text = chunk.content
                    if isinstance(text, str) and text:
                        yield ("token", text)
            elif isinstance(chunk, ToolMessage):
                # Prefer buffered name; fall back to ToolMessage.name attribute
                tool_name = pending_tool_name or (getattr(chunk, "name", None) or "unknown")
                desc = _tool_desc_map().get(tool_name, "")
                summary = _summarize_result(str(chunk.content))
                yield (
                    "tool_interaction",
                    json.dumps(
                        {"name": tool_name, "description": desc, "summary": summary},
                        ensure_ascii=False,
                    ),
                )
                pending_tool_name = ""
    except Exception as e:
        yield ("error", str(e))


def stream_agent(
    question: str,
    memory: PortfolioMemory | None = None,
    history_messages: list | None = None,
    tushare_score: int = 0,
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
        streaming_error: str | None = None
        for ev_type, content in _stream_react(msgs, score=tushare_score):
            if ev_type == "error":
                streaming_error = content
                break
            yield (ev_type, content)

        if streaming_error:
            # Non-streaming fallback
            try:
                agent2 = build_react_agent(tushare_score)
                result = agent2.invoke({"messages": msgs})
                fallback = extract_last_ai_text(result.get("messages", [])) or "（无返回文本）"
                yield ("answer", f"[流式失败，完整回答]\n\n{fallback}")
            except Exception as e2:
                yield ("error", f"ReAct 流式失败: {streaming_error}; 降级也失败: {e2}")
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
        # PnE failed entirely – fall back to ReAct (uses the same structured events)
        yield ("thinking", f"[P&E 失败 ({e})，降级至 ReAct]")
        for ev_type, content in _stream_react(msgs, score=tushare_score):
            yield (ev_type, content)
