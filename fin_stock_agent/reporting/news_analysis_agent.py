from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage

from fin_stock_agent.core.llm import get_llm, merge_token_usage
from fin_stock_agent.news.models import NewsItem

logger = logging.getLogger(__name__)


def _top_news(news_items: list[NewsItem], limit: int = 10) -> list[dict]:
    return [
        {
            "title": item.title,
            "source": item.source,
            "time": item.published_at.isoformat() if item.published_at else "",
        }
        for item in news_items[:limit]
    ]


def _keyword_matches(news_items: list[NewsItem], holdings_keywords: list[str]) -> dict[str, list[str]]:
    relevant: dict[str, list[str]] = {}
    for item in news_items[:10]:
        blob = f"{item.title} {item.summary}"
        matched = [keyword for keyword in holdings_keywords if keyword and keyword in blob]
        if matched:
            relevant[item.title] = matched
    return relevant


class NewsAnalysisAgent:
    def __init__(self) -> None:
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def run(
        self,
        news_items: list[NewsItem],
        holdings_keywords: list[str],
        *,
        daily_briefing: dict[str, Any] | None = None,
    ) -> dict:
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        top_news = _top_news(news_items)
        relevant = _keyword_matches(news_items, holdings_keywords)
        briefing = daily_briefing or {}
        summary_text = briefing.get("summary") or "News integrated from configured sources."
        catalog_md = briefing.get("markdown_catalog") or ""
        briefing_top10 = briefing.get("top_10") or []

        if not news_items:
            return self._fallback(
                summary_text=summary_text,
                catalog_md=catalog_md,
                top_news=top_news,
                relevant=relevant,
                briefing_top10=briefing_top10,
            )

        try:
            llm = get_llm("news_analysis")
            prompt = (
                "You are a Chinese financial news analysis agent.\n"
                "Summarize market sentiment for a user portfolio and return JSON only.\n"
                "JSON schema:\n"
                "{"
                '"sentiment_score": 0.0,'
                '"sentiment_label": "bullish|watch|neutral|cautious|bearish",'
                '"market_summary": "short Chinese summary",'
                '"risk_signals": ["signal1"],'
                '"holdings_relevant_news": {"title": ["keyword"]}'
                "}\n\n"
                f"Holdings keywords: {json.dumps(holdings_keywords[:12], ensure_ascii=False)}\n"
                f"Daily briefing summary: {summary_text}\n"
                f"Daily briefing catalog: {catalog_md[:1600]}\n"
                "Top news feed:\n"
                + "\n".join(
                    f"- [{item.source}] {item.title} | {(item.summary or '')[:180]}"
                    for item in news_items[:12]
                )
            )
            response = llm.invoke([HumanMessage(content=prompt)])
            self.last_usage = merge_token_usage(response)
            raw = response.content if isinstance(response.content, str) else ""
            raw = re.sub(r"```(?:json)?", "", raw).strip()
            start, end = raw.find("{"), raw.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(raw[start:end])
                sentiment_label = str(data.get("sentiment_label") or "neutral")
                return {
                    "sentiment_score": float(data.get("sentiment_score", 0.55)),
                    "sentiment_label": sentiment_label,
                    "topic_summary": {
                        "market": str(data.get("market_summary") or summary_text),
                        "catalog": catalog_md,
                    },
                    "top_news": top_news,
                    "daily_briefing_top10": briefing_top10,
                    "holdings_relevant_news": data.get("holdings_relevant_news") or relevant,
                    "risk_signals": data.get("risk_signals") or [],
                }
        except Exception as exc:
            logger.warning("NewsAnalysisAgent LLM failed: %s", exc)

        return self._fallback(
            summary_text=summary_text,
            catalog_md=catalog_md,
            top_news=top_news,
            relevant=relevant,
            briefing_top10=briefing_top10,
        )

    def _fallback(
        self,
        *,
        summary_text: str,
        catalog_md: str,
        top_news: list[dict],
        relevant: dict[str, list[str]],
        briefing_top10: list[dict],
    ) -> dict:
        return {
            "sentiment_score": 0.55,
            "sentiment_label": "watch" if relevant else "neutral",
            "topic_summary": {
                "market": summary_text,
                "catalog": catalog_md,
            },
            "top_news": top_news,
            "daily_briefing_top10": briefing_top10,
            "holdings_relevant_news": relevant,
            "risk_signals": [],
        }
