"""ReAct agent – used for simple single-intent queries and as PnE fallback."""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.prebuilt import ToolNode, create_react_agent

from fin_stock_agent.core.llm import get_llm
from fin_stock_agent.core.tushare_permissions import get_available_tool_names
from fin_stock_agent.prompts.react_prompt import REACT_SYSTEM_PROMPT
from fin_stock_agent.tools.activity import get_activity_tools
from fin_stock_agent.tools.derivatives import get_derivatives_tools
from fin_stock_agent.tools.fund_etf import get_fund_etf_tools
from fin_stock_agent.tools.fundamentals import get_fundamentals_tools
from fin_stock_agent.tools.global_markets import get_global_market_tools
from fin_stock_agent.tools.macro import get_macro_tools
from fin_stock_agent.tools.market import get_market_tools
from fin_stock_agent.tools.memory_tools import get_memory_tools
from fin_stock_agent.tools.portfolio import get_portfolio_tools
from fin_stock_agent.tools.sector import get_sector_tools
from fin_stock_agent.utils.datetime_utils import get_datetime_tools


def _all_tools(score: int = 0) -> list:
    """
    Return the list of tools available at the given Tushare score.

    ``score=0`` (default) means unrestricted – all tools are returned,
    preserving backward-compatible behaviour when no score is configured.
    """
    candidates = [
        *get_datetime_tools(),        # time resolution (always first)
        *get_market_tools(),          # A-share market data + indices
        *get_fund_etf_tools(),        # fund / ETF search and daily bars
        *get_sector_tools(),          # concept boards and index members
        *get_macro_tools(),           # macro-economy (CPI / M2 / GDP / SHIBOR)
        *get_fundamentals_tools(),    # financial statements + stock screener
        *get_activity_tools(),        # money flow, limit list, dragon-tiger, northbound
        *get_global_market_tools(),   # HK stocks, US stocks, global indices
        *get_derivatives_tools(),     # convertible bonds, futures, options
        *get_portfolio_tools(),       # portfolio PnL calculation
        *get_memory_tools(),          # in-session trade memory
    ]
    if score <= 0:
        return candidates  # unrestricted
    allowed = get_available_tool_names(score)
    return [t for t in candidates if t.name in allowed]


# ---------------------------------------------------------------------------
# Adaptive tool guard: blocks tools after consecutive empty / error results
# ---------------------------------------------------------------------------

def _is_empty_result(content: str) -> bool:
    """Return True when a tool result represents empty data or a permission failure."""
    try:
        data = json.loads(content)
        return data.get("ok") is False or data.get("rows", -1) == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Per-tool fallback hints shown inside the blocked ToolMessage so the LLM
# always has a concrete next step instead of giving up.
# ---------------------------------------------------------------------------
_TOOL_FALLBACK_HINTS: dict[str, str] = {
    "get_global_index_daily": (
        "请立即改用以下路径获取数据：\n"
        "① 调用 search_sector_etf 搜索跟踪该指数的 A 股 ETF（如关键字「恒生科技」）\n"
        "② 取返回的 ts_code（格式 xxxxxx.SH/.SZ），调用 get_daily_bars 查询最近行情\n"
        "此路径不依赖海外权限，必有数据，必须立即执行，不要停止分析。"
    ),
    "get_fund_daily": (
        "请改用 get_daily_bars（使用场内 ETF 代码 xxxxxx.SH/.SZ）获取行情，"
        "若代码未知请先用 search_sector_etf 或 search_fund 搜索。"
    ),
    "get_hk_daily": (
        "请改用 search_sector_etf 搜索相关港股联接基金/ETF，"
        "再用 get_daily_bars（.SH/.SZ 代码）查询 A 股行情作为替代。"
    ),
    "get_us_daily": (
        "请改用 search_sector_etf 搜索相关 QDII/美股 ETF，"
        "再用 get_daily_bars（.SH/.SZ 代码）查询 A 股行情作为替代。"
    ),
    "get_moneyflow_hsgt": (
        "请改用 get_sw_industry_top_movers 查行业异动，"
        "结合 get_daily_bars 查大盘整体走势作为替代。"
    ),
    "get_hsgt_top10": (
        "请改用 get_sw_industry_top_movers 查近期行业强弱作为替代。"
    ),
    "get_top_list": (
        "请改用 get_limit_list 查涨跌停情况，或用 get_moneyflow 查个股资金流向。"
    ),
}

_DEFAULT_FALLBACK_HINT = (
    "请停止调用此工具，立即转向其他可用工具继续完成分析任务，不要放弃。"
)


class AdaptiveToolGuard:
    """
    Passed as ``wrap_tool_call`` to ``ToolNode``.

    Tracks consecutive empty / failed results per tool name within a single
    agent invocation.  After ``THRESHOLD`` consecutive empty results the tool
    call is skipped and a synthetic ToolMessage is returned that includes a
    concrete fallback hint, so the LLM always has a next step instead of
    stopping prematurely.

    A fresh instance is created for every ``build_react_agent()`` call so
    counts never leak between user turns.
    """

    THRESHOLD: int = 2

    def __init__(self) -> None:
        self._fail_counts: dict[str, int] = {}

    def __call__(
        self,
        request: Any,
        execute: Callable[[Any], ToolMessage | Any],
    ) -> ToolMessage | Any:
        tool_call = request.tool_call
        tool_name: str = tool_call.get("name", "")
        fail_count = self._fail_counts.get(tool_name, 0)

        if fail_count >= self.THRESHOLD:
            hint = _TOOL_FALLBACK_HINTS.get(tool_name, _DEFAULT_FALLBACK_HINT)
            return ToolMessage(
                content=(
                    f"[工具已被系统拦截] {tool_name} 已连续 {fail_count} 次返回空数据，"
                    f"判断为 Tushare 权限不足。本次调用已跳过，请勿再次调用此工具。\n"
                    f"{hint}"
                ),
                tool_call_id=tool_call["id"],
                name=tool_name,
            )

        # Execute normally
        result = execute(request)

        # Track the result
        if isinstance(result, ToolMessage):
            if _is_empty_result(str(result.content)):
                self._fail_counts[tool_name] = fail_count + 1
            else:
                self._fail_counts[tool_name] = 0  # reset on success

        return result


def build_react_agent(score: int = 0):
    """
    Compile and return a ReAct agent graph.

    Args:
        score: Tushare score. Tools requiring a higher score are excluded.
               0 (default) = unrestricted (all tools available).
    """
    guard = AdaptiveToolGuard()
    tool_node = ToolNode(_all_tools(score), wrap_tool_call=guard)
    return create_react_agent(get_llm(), tool_node, prompt=REACT_SYSTEM_PROMPT)


def extract_last_ai_text(messages: list[BaseMessage]) -> str:
    """Return the text content of the last AIMessage in a messages list."""
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            c = m.content
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                parts = [
                    block.get("text", "")
                    for block in c
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                return "\n".join(parts)
    return ""
