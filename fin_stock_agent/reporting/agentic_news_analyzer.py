"""AgenticNewsAnalyzer — LLM + tools: daily briefing + keyword news search + position sizing."""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from fin_stock_agent.core.llm import merge_token_usage, get_llm
from fin_stock_agent.news.models import NewsItem
from fin_stock_agent.news.news_reader import NewsReader

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
你是中国公募基金投顾助手。你必须结合三类信息做判断：
1) **当日市场要点**（工具 get_daily_briefing）
2) **与基金相关的概念/行业新闻**（工具 search_news，用基金名称里的主题词、行业词多次搜索）
3) **用户真实持仓规模**：份额、成本均价、当前市值、浮动盈亏 —— 已在用户消息中给出；\
仓位大或亏损/盈利显著时，风险态度应更谨慎，理由中要点明与金额相关的考量。

流程建议：
- 先 get_daily_briefing 了解大盘与主线
- 再用 search_news 搜索基金名称中的关键词（如「白酒」「科创」「债」「红利」等），必要时换同义词再搜
- 综合后输出 JSON，不要输出其他文字

【重要】最终回答必须且只能是以下 JSON：
{
  "action": "buy" | "hold" | "sell",
  "confidence": 0.0 到 1.0,
  "reasoning": "中文简要理由（不超过80字，需体现新闻+当日形势+仓位/盈亏若相关）",
  "relevant_titles": ["用到的相关新闻标题1", "标题2"]
}

数据不足时 action=\"hold\"，confidence=0.5，reasoning 说明数据不足。
"""


class AgenticNewsAnalyzer:
    def __init__(self) -> None:
        self._news_reader = NewsReader()
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def analyze(
        self,
        holdings: list[dict],
        all_news: list[NewsItem],  # noqa: ARG002
        *,
        daily_briefing: dict[str, Any] | None = None,
        name_keywords: list[str] | None = None,
    ) -> dict[str, dict]:
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if not holdings:
            return {}

        try:
            llm = get_llm("agentic_news")
        except Exception as exc:
            logger.warning("LLM unavailable for agentic news analysis: %s", exc)
            return {h["ts_code"]: _neutral() for h in holdings}

        briefing = daily_briefing or {}
        briefing_text = briefing.get("markdown_catalog") or briefing.get("summary") or "（无简报）"
        hint_keywords = ", ".join((name_keywords or [])[:8])

        news_reader = self._news_reader

        @tool
        def get_daily_briefing() -> str:
            """返回今日市场重点十条目录与摘要（已预生成）。"""
            return briefing_text[:4000]

        @tool
        def search_news(keyword: str) -> str:
            """按关键字搜索本地新闻缓存（标题+摘要匹配）。"""
            items = news_reader.get_cached_news(keywords=[keyword], limit=12)
            if not items:
                return "未找到与 " + keyword + " 相关的新闻。"
            lines = [
                "[" + it.get("source", "") + "] " + it["title"] + ": " + (it.get("summary") or "")[:120]
                for it in items
            ]
            return "\n".join(lines)

        try:
            agent = create_react_agent(llm, [get_daily_briefing, search_news], prompt=_SYSTEM_PROMPT)
        except Exception as exc:
            logger.warning("Failed to build ReAct agent: %s", exc)
            return {h["ts_code"]: _neutral() for h in holdings}

        results: dict[str, dict] = {}
        for holding in holdings:
            code = holding["ts_code"]
            name = holding.get("name", code)
            qty = holding.get("quantity", 0.0)
            avg = holding.get("avg_cost", 0.0)
            mv = holding.get("market_value")
            pnl = holding.get("unrealized_pnl")
            position_block = (
                f"基金：{name}（{code}）\n"
                f"持仓份额：{qty}\n成本均价：{avg}\n"
                f"当前市值：{mv if mv is not None else '未知'}\n"
                f"浮动盈亏：{pnl if pnl is not None else '未知'}\n"
                f"可优先搜索的概念词提示：{hint_keywords or '（无）'}"
            )
            try:
                rec, usage = self._analyze_one(agent, position_block)
                self.last_usage = merge_token_usage(self.last_usage, usage)
            except Exception as exc:
                logger.warning("Agentic analysis failed for %s: %s", code, exc)
                rec = _neutral()
            results[code] = rec

        return results

    def _analyze_one(self, agent: Any, position_block: str) -> tuple[dict, dict[str, int]]:
        prompt = position_block + "\n请调用工具并完成分析，最后只输出 JSON。"
        try:
            response = agent.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config={"recursion_limit": 14},
            )
        except Exception as exc:
            logger.warning("Agent invoke error: %s", exc)
            return _neutral(), {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        messages = response.get("messages", [])
        text = _last_ai_text(messages)
        return _parse_recommendation(text), merge_token_usage(messages)


def _last_ai_text(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return ""


def _parse_recommendation(text: str) -> dict:
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if 0 <= start < end:
            data = json.loads(text[start:end])
            action = data.get("action", "hold")
            if action not in {"buy", "hold", "sell"}:
                action = "hold"
            return {
                "action": action,
                "confidence": float(data.get("confidence", 0.5)),
                "reasoning": str(data.get("reasoning", ""))[:160],
                "relevant_titles": list(data.get("relevant_titles", [])),
            }
    except Exception:
        pass
    return _neutral()


def _neutral() -> dict:
    return {
        "action": "hold",
        "confidence": 0.5,
        "reasoning": "新闻数据不足，建议观望。",
        "relevant_titles": [],
    }
