from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage

from fin_stock_agent.core.llm import get_llm, merge_token_usage
from fin_stock_agent.news.models import NewsItem
from fin_stock_agent.prompts.reporting_prompts import NEWS_BRIEFING_PROMPT
from fin_stock_agent.reporting.report_models import AgentResult, AgentTimer, ReportContext

logger = logging.getLogger(__name__)


def _feed_lines(items: list[NewsItem]) -> str:
    return "\n".join(f"{index}. [{item.source}] {item.title}" for index, item in enumerate(items, 1))


class NewsFilterAgent:
    def __init__(self) -> None:
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def run(self, ctx: ReportContext) -> AgentResult:
        timer = AgentTimer()
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if not ctx.news_items:
            return AgentResult(
                agent_name="news_filter",
                status="fallback",
                output={
                    "top_news": [],
                    "personalized_news": [],
                    "briefing_summary": "No news items were available for the selected date.",
                    "markdown_catalog": "",
                },
                elapsed_ms=timer.elapsed_ms(),
            )

        ranked = self._rank_news(ctx)
        top_news = [self._serialize_item(item) for item, _score in ranked[: ctx.config.briefing_top_n]]
        personalized = [self._serialize_item(item) for item, _score in ranked if self._relevance_score(ctx, item) > 0][
            : ctx.config.personalized_news_top_n
        ]

        summary = self._llm_summary(ctx.news_items) or self._fallback_summary(top_news)
        output = {
            "top_news": top_news,
            "personalized_news": personalized,
            "briefing_summary": summary,
            "markdown_catalog": self._to_markdown(summary, top_news),
        }
        return AgentResult(
            agent_name="news_filter",
            status="success",
            output=output,
            token_usage=dict(self.last_usage),
            elapsed_ms=timer.elapsed_ms(),
        )

    def _rank_news(self, ctx: ReportContext) -> list[tuple[NewsItem, float]]:
        items = ctx.news_items[: ctx.config.news_fetch_limit]
        ranked: list[tuple[NewsItem, float]] = []
        for index, item in enumerate(items):
            importance_score = max(0.0, (10 - min(index, 9)) / 10)
            relevance_score = self._relevance_score(ctx, item)
            total_score = 0.6 * importance_score + 0.4 * relevance_score
            ranked.append((item, total_score))
        ranked.sort(key=lambda pair: pair[1], reverse=True)
        return ranked

    def _relevance_score(self, ctx: ReportContext, item: NewsItem) -> float:
        keywords = list(ctx.user_profile.focus_themes)
        keywords.extend(holding.get("name", "") for holding in ctx.holdings)
        keywords.extend(ctx.user_profile.watchlist)
        blob = f"{item.title} {item.summary}".lower()
        matched = 0
        for keyword in keywords:
            normalized = str(keyword or "").strip().lower()
            if normalized and normalized in blob:
                matched += 1
        return min(1.0, matched * 0.25)

    def _llm_summary(self, items: list[NewsItem]) -> str:
        try:
            llm = get_llm("daily_briefing", temperature=0.2)
            prompt = NEWS_BRIEFING_PROMPT.format(news_feed=_feed_lines(items[:10]))
            response = llm.invoke([HumanMessage(content=prompt)])
            self.last_usage = merge_token_usage(response)
            text = getattr(response, "content", "")
            if isinstance(text, str) and text.strip():
                return text.strip()[:180]
        except Exception as exc:
            logger.warning("NewsFilterAgent 摘要生成失败，降级使用规则摘要: %s", exc)
        return ""

    @staticmethod
    def _fallback_summary(top_news: list[dict]) -> str:
        if not top_news:
            return "暂无重要市场新闻。"
        titles = "；".join(item["title"] for item in top_news[:3])
        return f"今日市场焦点：{titles}"

    @staticmethod
    def _serialize_item(item: NewsItem) -> dict:
        return {
            "title": item.title,
            "source": item.source,
            "time": item.published_at.isoformat() if item.published_at else "",
        }

    @staticmethod
    def _to_markdown(summary: str, top_news: list[dict]) -> str:
        lines = [summary, "", "重点新闻："]
        for index, item in enumerate(top_news, 1):
            meta = " | ".join(part for part in [item.get("source", ""), item.get("time", "")] if part)
            if meta:
                lines.append(f"{index}. {item['title']} [{meta}]")
            else:
                lines.append(f"{index}. {item['title']}")
        return "\n".join(lines)


DailyBriefingAgent = NewsFilterAgent
