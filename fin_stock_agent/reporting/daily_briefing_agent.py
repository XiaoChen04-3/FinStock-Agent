"""DailyBriefingAgent — pick the ~10 most market-moving headlines for the day.

Uses the LLM when available; otherwise falls back to recency ordering.
Output feeds the daily report ``market_context`` and the agentic analyzer.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage

from fin_stock_agent.core.llm import get_llm, merge_token_usage
from fin_stock_agent.news.models import NewsItem

logger = logging.getLogger(__name__)

def _feed_lines(items: list[NewsItem]) -> str:
    lines: list[str] = []
    for i, it in enumerate(items, 1):
        lines.append(f"{i}. [{it.source}] {it.title}")
    return "\n".join(lines)


class DailyBriefingAgent:
    def __init__(self) -> None:
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def run(self, news_items: list[NewsItem]) -> dict[str, Any]:
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if not news_items:
            return {
                "summary": "今日暂无抓取到新闻。",
                "top_10": [],
                "markdown_catalog": "",
            }

        try:
            llm = get_llm("daily_briefing", temperature=0.2)
            prompt = (
                "你是财经主编。下面是从财联社/东财/同花顺抓取的今日快讯列表（含标题与摘要）。\n"
                "请选出对**当日A股与基金市场**影响最大的10条，按影响从大到小排序。\n"
                "输出**仅**一个 JSON 对象，字段如下：\n"
                '{"summary":"用2-3句中文概括当日市场氛围与主线",'
                '"top_10":[{"rank":1,"title":"原文标题","impact":1-5,"reason":"一句话为何重要","source":"cls|eastmoney|ths"}'
                ",...]}\n\n"
                + _feed_lines(news_items)
            )
            resp = llm.invoke([HumanMessage(content=prompt)])
            self.last_usage = merge_token_usage(resp)
            raw = resp.content if isinstance(resp.content, str) else ""
            raw = re.sub(r"```(?:json)?", "", raw).strip()
            s, e = raw.find("{"), raw.rfind("}") + 1
            if s >= 0 and e > s:
                data = json.loads(raw[s:e])
                top = data.get("top_10") or []
                summary = str(data.get("summary") or "").strip() or "已生成当日要闻目录。"
                return {
                    "summary": summary,
                    "top_10": top,
                    "markdown_catalog": _to_markdown(summary, top),
                }
        except Exception as exc:
            logger.warning("DailyBriefingAgent LLM failed: %s", exc)

        return _fallback(news_items)


def _to_markdown(summary: str, top_10: list[dict]) -> str:
    parts = [f"**当日要点**  {summary}", "", "**重点十条**"]
    for row in top_10[:10]:
        r = row.get("rank", 0)
        t = row.get("title", "")
        imp = row.get("impact", "")
        why = row.get("reason", "")
        src = row.get("source", "")
        parts.append(f"{r}. **{t}**  [{src}]  影响:{imp}/5 — {why}")
    return "\n".join(parts)


def _fallback(items: list[NewsItem]) -> dict[str, Any]:
    top = []
    for i, it in enumerate(items[:10], 1):
        top.append(
            {
                "rank": i,
                "title": it.title,
                "impact": 3,
                "reason": "按抓取顺序（LLM 不可用）",
                "source": it.source,
            }
        )
    summary = f"共 {len(items)} 条快讯；以下为按时间优先的十条摘要。"
    return {
        "summary": summary,
        "top_10": top,
        "markdown_catalog": _to_markdown(summary, top),
    }
