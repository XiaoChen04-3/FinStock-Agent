from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage

from fin_stock_agent.core.config import get_config
from fin_stock_agent.core.llm import get_llm, merge_token_usage
from fin_stock_agent.core.time_utils import now_local
from fin_stock_agent.memory.profile_memory import UserProfileMemory
from fin_stock_agent.prompts.reporting_prompts import REPORT_SYNTHESIS_PROMPT
from fin_stock_agent.reporting.models import DailyReport, FundDailyStatus
from fin_stock_agent.reporting.report_models import AgentResult, AgentTimer, ReportContext

logger = logging.getLogger(__name__)

_ACTION_LABEL = {"buy": "建议加仓", "hold": "建议持有", "sell": "建议卖出"}


class ReportGenerationAgent:
    def __init__(self) -> None:
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def run(self, *args, **kwargs) -> DailyReport:
        if args and isinstance(args[0], ReportContext):
            ctx = args[0]
            results = args[1] if len(args) > 1 else kwargs.get("results", {})
            elapsed_ms = float(kwargs.get("elapsed_ms", 0.0))
            return self._run_with_context(ctx, results, elapsed_ms=elapsed_ms)
        return self._run_legacy(**kwargs)

    def _run_with_context(self, ctx: ReportContext, results: dict[str, AgentResult], *, elapsed_ms: float) -> DailyReport:
        news_filter = results.get("news_filter") or AgentResult(agent_name="news_filter", status="fallback")
        sentiment = results.get("sentiment") or AgentResult(agent_name="sentiment_analysis", status="fallback")
        fund_trend = results.get("fund_trend") or AgentResult(agent_name="fund_trend", status="fallback")
        holding_corr = results.get("holding_corr") or AgentResult(agent_name="holding_correlation", status="fallback")
        statuses = self._build_statuses(ctx, fund_trend.output.get("analyses", {}), holding_corr.output.get("recommendations", {}), news_filter.output.get("top_news", []), sentiment.output.get("risk_signals", []))
        overall_summary = self._fallback_summary(statuses, sentiment.output)
        market_context = sentiment.output.get("market_summary") or news_filter.output.get("briefing_summary") or "暂无市场概述。"
        top_news = news_filter.output.get("top_news") or []

        try:
            llm = get_llm("report_synthesis")
            prompt = REPORT_SYNTHESIS_PROMPT.format(
                statuses=json.dumps([item.model_dump() for item in statuses], ensure_ascii=False, default=str),
                sentiment=json.dumps(sentiment.output, ensure_ascii=False, default=str),
                top_news=json.dumps(top_news, ensure_ascii=False, default=str),
            )
            response = llm.invoke([HumanMessage(content=prompt)])
            self.last_usage = merge_token_usage(response)
            raw = response.content if isinstance(response.content, str) else ""
            raw = re.sub(r"```(?:json)?", "", raw).strip()
            start, end = raw.find("{"), raw.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(raw[start:end])
                overall_summary = str(data.get("overall_summary") or overall_summary)
                market_context = str(data.get("market_context") or market_context)
        except Exception as exc:
            logger.warning("ReportGenerationAgent LLM 调用失败，使用规则摘要: %s", exc)

        total_market_value = sum(item.market_value or 0.0 for item in statuses)
        total_unrealized = sum(item.unrealized_pnl or 0.0 for item in statuses)
        total_unrealized_pct = total_unrealized / total_market_value if total_market_value else 0.0
        report = DailyReport(
            user_id=ctx.user_id,
            report_date=ctx.report_date,
            generated_at=now_local(),
            recent_trading_days=ctx.recent_trading_days,
            total_market_value=round(total_market_value, 4),
            total_unrealized_pnl=round(total_unrealized, 4),
            total_unrealized_pnl_pct=round(total_unrealized_pct, 4),
            today_portfolio_change_pct=0.0,
            fund_statuses=statuses,
            overall_summary=overall_summary[: ctx.config.report_summary_max_chars],
            market_context=market_context,
            news_sentiment_label=str(sentiment.output.get("sentiment_label") or "neutral"),
            top_news=top_news,
            total_elapsed_ms=elapsed_ms,
        )
        return ReportValidationAgent().validate(report)

    def _run_legacy(
        self,
        *,
        user_id: str,
        report_date: str,
        recent_trading_days: list[str],
        holdings: list[dict],
        news_ctx: dict,
        fund_ctx: dict,
        holding_recommendations: dict | None = None,
        elapsed_ms: float,
    ) -> DailyReport:
        fake_ctx = ReportContext(
            user_id=user_id,
            report_date=report_date,
            holdings=holdings,
            user_profile=UserProfileMemory(),
            news_items=[],
            raw_nav_history={},
            recent_trading_days=recent_trading_days,
            config=get_config().daily_report,
        )
        results = {
            "news_filter": AgentResult(agent_name="news_filter", status="success", output={"top_news": news_ctx.get("daily_briefing_top10") or news_ctx.get("top_news") or [], "briefing_summary": (news_ctx.get("topic_summary") or {}).get("market", "")}),
            "sentiment": AgentResult(agent_name="sentiment_analysis", status="success", output={"sentiment_label": news_ctx.get("sentiment_label", "neutral"), "market_summary": (news_ctx.get("topic_summary") or {}).get("market", ""), "risk_signals": news_ctx.get("risk_signals", [])}),
            "fund_trend": AgentResult(agent_name="fund_trend", status="success", output=fund_ctx),
            "holding_corr": AgentResult(agent_name="holding_correlation", status="success", output={"recommendations": holding_recommendations or {}}),
        }
        return self._run_with_context(fake_ctx, results, elapsed_ms=elapsed_ms)

    @staticmethod
    def _build_statuses(
        ctx: ReportContext,
        analyses: dict[str, dict],
        recommendations: dict[str, dict],
        top_news: list[dict],
        risk_signals: list[str],
    ) -> list[FundDailyStatus]:
        statuses: list[FundDailyStatus] = []
        fallback_titles = [item.get("title", "") for item in top_news[:3] if item.get("title")]
        for holding in ctx.holdings:
            code = holding["ts_code"]
            analysis = analyses.get(code, {})
            recommendation = recommendations.get(code, {})
            reasoning = str(recommendation.get("reasoning") or "").strip()
            analysis_text = str(analysis.get("analysis") or "").strip()
            if reasoning and analysis_text and analysis_text not in reasoning:
                reason = f"{analysis_text} {reasoning}"
            else:
                reason = reasoning or analysis_text
            market_value = holding.get("market_value")
            unrealized = holding.get("unrealized_pnl")
            cost_basis = max(float(holding.get("avg_cost") or 0.0) * float(holding.get("quantity") or 0.0), 1e-6)
            statuses.append(
                FundDailyStatus(
                    ts_code=code,
                    name=holding.get("name", code),
                    quantity=float(holding.get("quantity") or 0.0),
                    avg_cost=float(holding.get("avg_cost") or 0.0),
                    nav_today=holding.get("last_price"),
                    market_value=market_value,
                    unrealized_pnl=unrealized,
                    unrealized_pnl_pct=(unrealized or 0.0) / cost_basis if market_value is not None else None,
                    today_change_pct=0.0,
                    action=str(recommendation.get("action") or "hold"),
                    confidence=float(recommendation.get("confidence", 0.5)),
                    reason=reason,
                    trend=str(analysis.get("trend") or "insufficient_data"),
                    analysis_summary=str(analysis.get("analysis") or ""),
                    three_year_return_pct=analysis.get("metrics", {}).get("return_3y"),
                    key_risks=list(risk_signals or [])[:3],
                    related_news=list(recommendation.get("relevant_titles") or fallback_titles),
                )
            )
        return statuses

    @staticmethod
    def _fallback_summary(statuses: list[FundDailyStatus], sentiment_output: dict) -> str:
        if statuses:
            parts = []
            for action_key, label in _ACTION_LABEL.items():
                names = [status.name for status in statuses if status.action == action_key]
                if names:
                    parts.append(f"{label}: {', '.join(names)}")
            if parts:
                return "；".join(parts)
        return str(sentiment_output.get("market_summary") or "综合当日新闻与持仓状况，市场情绪整体中性，请关注持仓动态。")


class ReportValidationAgent:
    def validate(self, report: DailyReport) -> DailyReport:
        valid_actions = {"buy", "hold", "sell"}
        valid_labels = {"bullish", "watch", "neutral", "cautious", "bearish"}
        for status in report.fund_statuses:
            if status.action not in valid_actions:
                status.action = "hold"
            if not 0.0 <= status.confidence <= 1.0:
                status.confidence = max(0.0, min(1.0, status.confidence))
            if not status.reason:
                status.reason = "数据不足，建议持续观察"
        if report.news_sentiment_label not in valid_labels:
            report.news_sentiment_label = "neutral"
        if not report.overall_summary:
            report.overall_summary = ReportGenerationAgent._fallback_summary(report.fund_statuses, {"market_summary": report.market_context})
        return report


ReportSynthesisAgent = ReportGenerationAgent
