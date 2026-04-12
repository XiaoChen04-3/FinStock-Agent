from __future__ import annotations

import json
import logging
import re
from datetime import datetime

from langchain_core.messages import HumanMessage

from fin_stock_agent.core.llm import get_llm, merge_token_usage
from fin_stock_agent.core.time_utils import now_local
from fin_stock_agent.reporting.models import DailyReport, FundDailyStatus, MarketFundIdea

logger = logging.getLogger(__name__)

_ACTION_LABEL = {
    "buy": "建议加仓",
    "hold": "建议持有",
    "sell": "建议卖出",
}


class ReportSynthesisAgent:
    def __init__(self) -> None:
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def run(
        self,
        *,
        user_id: str,
        report_date: str,
        recent_trading_days: list[str],
        holdings: list[dict],
        news_ctx: dict,
        fund_ctx: dict,
        market_fund_ideas: list[dict] | None = None,
        holding_recommendations: dict | None = None,
        elapsed_ms: float,
    ) -> DailyReport:
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        recs = holding_recommendations or {}
        market_fund_ideas = market_fund_ideas or []
        statuses: list[FundDailyStatus] = []
        total_market_value = 0.0
        total_unrealized = 0.0
        ranked_news = news_ctx.get("daily_briefing_top10") or news_ctx.get("top_news", [])

        for holding in holdings:
            code = holding["ts_code"]
            total_market_value += holding.get("market_value") or 0.0
            total_unrealized += holding.get("unrealized_pnl") or 0.0

            fund_analysis = fund_ctx.get("analyses", {}).get(code, {})
            rec = recs.get(code, {})
            action = rec.get("action", "hold")
            confidence = float(rec.get("confidence", 0.6))
            reason = self._merge_reasoning(rec.get("reasoning"), fund_analysis.get("analysis"))
            agentic_titles: list[str] = rec.get("relevant_titles", [])
            fallback_titles: list[str] = [item.get("title", "") for item in ranked_news[:2]]
            related = (agentic_titles or fallback_titles)[:5]
            three_year_return = fund_analysis.get("metrics", {}).get("return_3y")

            statuses.append(
                FundDailyStatus(
                    ts_code=code,
                    name=holding["name"],
                    quantity=holding["quantity"],
                    avg_cost=holding["avg_cost"],
                    nav_today=holding.get("last_price"),
                    market_value=holding.get("market_value"),
                    unrealized_pnl=holding.get("unrealized_pnl"),
                    unrealized_pnl_pct=(
                        (holding.get("unrealized_pnl") or 0.0)
                        / max(holding["avg_cost"] * holding["quantity"], 1e-6)
                    ),
                    today_change_pct=0.0,
                    action=action,
                    confidence=confidence,
                    reason=reason,
                    trend=str(fund_analysis.get("trend") or "insufficient_data"),
                    analysis_summary=str(fund_analysis.get("analysis") or ""),
                    three_year_return_pct=float(three_year_return) if three_year_return is not None else None,
                    key_risks=list(news_ctx.get("risk_signals") or [])[:3],
                    related_news=related,
                )
            )

        pct = total_unrealized / total_market_value if total_market_value else 0.0
        overall_summary = self._fallback_summary(statuses, market_fund_ideas)

        try:
            llm = get_llm("report_synthesis")
            prompt = (
                "You are a Chinese portfolio report synthesis agent.\n"
                "Return JSON only with keys overall_summary and market_context.\n"
                "Keep overall_summary under 180 Chinese characters.\n\n"
                f"Status rows: {json.dumps([item.model_dump() for item in statuses], ensure_ascii=False, default=str)}\n"
                f"Market fund ideas: {json.dumps(market_fund_ideas, ensure_ascii=False, default=str)}\n"
                f"News context: {json.dumps(news_ctx, ensure_ascii=False, default=str)}\n"
            )
            response = llm.invoke([HumanMessage(content=prompt)])
            self.last_usage = merge_token_usage(response)
            raw = response.content if isinstance(response.content, str) else ""
            raw = re.sub(r"```(?:json)?", "", raw).strip()
            start, end = raw.find("{"), raw.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(raw[start:end])
                overall_summary = str(data.get("overall_summary") or overall_summary)
                market_ctx = str(data.get("market_context") or "").strip()
            else:
                market_ctx = ""
        except Exception as exc:
            logger.warning("ReportSynthesisAgent LLM failed: %s", exc)
            market_ctx = ""

        ts = news_ctx.get("topic_summary") or {}
        default_market_line = ts.get("market") or "No market context."
        catalog = (ts.get("catalog") or "").strip()
        fallback_market_ctx = default_market_line if not catalog else default_market_line + "\n\n" + catalog
        market_context = market_ctx or fallback_market_ctx

        return DailyReport(
            user_id=user_id,
            report_date=report_date,
            generated_at=now_local(),
            recent_trading_days=recent_trading_days,
            total_market_value=round(total_market_value, 4),
            total_unrealized_pnl=round(total_unrealized, 4),
            total_unrealized_pnl_pct=round(pct, 4),
            today_portfolio_change_pct=0.0,
            fund_statuses=statuses,
            overall_summary=overall_summary,
            market_context=market_context,
            news_sentiment_label=news_ctx.get("sentiment_label", "unknown"),
            top_news=ranked_news,
            market_fund_ideas=[MarketFundIdea.model_validate(item) for item in market_fund_ideas],
            total_elapsed_ms=elapsed_ms,
        )

    def _fallback_summary(self, statuses: list[FundDailyStatus], market_fund_ideas: list[dict]) -> str:
        action_counts: dict[str, int] = {"buy": 0, "hold": 0, "sell": 0}
        for status in statuses:
            action_counts[status.action] = action_counts.get(status.action, 0) + 1
        summary_parts = []
        for action_key, label in _ACTION_LABEL.items():
            if action_counts.get(action_key, 0):
                names = [status.name for status in statuses if status.action == action_key]
                summary_parts.append(f"{label}: {', '.join(names)}")
        if market_fund_ideas:
            focus_names = [row.get("fund_name", "") for row in market_fund_ideas[:3] if row.get("fund_name")]
            if focus_names:
                summary_parts.append(f"可关注基金: {', '.join(focus_names)}")
        if summary_parts:
            return "；".join(summary_parts)
        return "基于最新新闻分析，当前适合先关注市场主线与主题基金，再择机分批布局。"

    def _merge_reasoning(self, agentic_reasoning: str | None, fund_analysis: str | None) -> str:
        parts: list[str] = []
        for text in (agentic_reasoning, fund_analysis):
            clean = (text or "").strip()
            if clean and clean not in parts:
                parts.append(clean)
        return "；".join(parts)[:220]
