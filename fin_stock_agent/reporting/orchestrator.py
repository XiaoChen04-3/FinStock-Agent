from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

from fin_stock_agent.core.config import get_config
from fin_stock_agent.core.llm import merge_token_usage
from fin_stock_agent.core.time_utils import today_local_str
from fin_stock_agent.init.trade_calendar import TradingCalendar
from fin_stock_agent.news.news_reader import NewsReader
from fin_stock_agent.reporting.agentic_news_analyzer import HoldingCorrelationAgent
from fin_stock_agent.reporting.daily_briefing_agent import NewsFilterAgent
from fin_stock_agent.reporting.fund_analysis_agent import FundTrendAgent
from fin_stock_agent.reporting.fund_fetcher import TushareFundFetcher
from fin_stock_agent.reporting.models import DailyReport
from fin_stock_agent.reporting.news_analysis_agent import SentimentAnalysisAgent
from fin_stock_agent.reporting.report_models import AgentResult, ReportContext
from fin_stock_agent.reporting.report_synthesis_agent import ReportGenerationAgent
from fin_stock_agent.services.portfolio_service import PortfolioService
from fin_stock_agent.services.user_memory_service import UserMemoryService


class OrchestratorAgent:
    def __init__(self) -> None:
        self.portfolio_service = PortfolioService()
        self.user_memory_service = UserMemoryService()
        self.news_reader = NewsReader()
        self.fund_fetcher = TushareFundFetcher()
        self.trade_calendar = TradingCalendar()
        self.news_filter_agent = NewsFilterAgent()
        self.sentiment_agent = SentimentAnalysisAgent()
        self.fund_trend_agent = FundTrendAgent()
        self.holding_corr_agent = HoldingCorrelationAgent()
        self.report_generation_agent = ReportGenerationAgent()

    def run(self, user_id: str, date: str | None = None, force: bool = False) -> DailyReport:
        _ = force
        started = perf_counter()
        cfg = get_config().daily_report
        report_date = date or today_local_str()
        holdings = self.portfolio_service.get_holdings(user_id)
        user_profile = self.user_memory_service.get_profile(user_id)
        news_result = self.news_reader.fetch_today_sync()
        codes = [item["ts_code"] for item in holdings]
        raw_nav_history = self.fund_fetcher.fetch_history(codes, years=cfg.nav_history_years) if codes else {}
        recent_trading_days = self.trade_calendar.get_recent_trading_days(cfg.recent_trading_days)
        ctx = ReportContext(
            user_id=user_id,
            report_date=report_date,
            holdings=holdings,
            user_profile=user_profile,
            news_items=news_result.items[: cfg.news_fetch_limit],
            raw_nav_history=raw_nav_history,
            recent_trading_days=recent_trading_days,
            config=cfg,
        )

        results: dict[str, AgentResult] = {}
        with ThreadPoolExecutor(max_workers=get_config().concurrency.daily_report_workers) as executor:
            futures = {"news_filter": executor.submit(self.news_filter_agent.run, ctx)}
            if holdings:
                futures["fund_trend"] = executor.submit(self.fund_trend_agent.run, ctx)
            for name, future in futures.items():
                try:
                    results[name] = future.result()
                except Exception as exc:
                    results[name] = AgentResult(agent_name=name, status="failed", error=str(exc))

        if results.get("news_filter", AgentResult(agent_name="news_filter", status="failed")).status == "failed":
            results["sentiment"] = AgentResult(agent_name="sentiment_analysis", status="failed", error="news_filter_failed")
        else:
            results["sentiment"] = self.sentiment_agent.run(ctx, results["news_filter"])

        if holdings:
            sentiment_result = results.get("sentiment")
            if sentiment_result is None or sentiment_result.status == "failed":
                sentiment_result = AgentResult(
                    agent_name="sentiment_analysis",
                    status="fallback",
                    output={
                        "sentiment_score": 0.5,
                        "sentiment_label": "neutral",
                        "market_summary": "Sentiment analysis degraded to neutral due to missing upstream output.",
                        "risk_signals": [],
                    },
                )
            results["holding_corr"] = self.holding_corr_agent.run(
                ctx,
                sentiment_result,
                results.get("fund_trend"),
                results.get("news_filter"),
            )
        else:
            results["holding_corr"] = AgentResult(agent_name="holding_correlation", status="fallback", output={"recommendations": {}})

        report = self.report_generation_agent.run(
            ctx,
            results,
            elapsed_ms=(perf_counter() - started) * 1000,
        )
        report.stage1_tokens = merge_token_usage(
            results.get("news_filter", AgentResult(agent_name="news_filter", status="fallback")).token_usage,
            results.get("fund_trend", AgentResult(agent_name="fund_trend", status="fallback")).token_usage,
        )["total_tokens"]
        report.stage2_tokens = merge_token_usage(
            results.get("sentiment", AgentResult(agent_name="sentiment", status="fallback")).token_usage,
            results.get("holding_corr", AgentResult(agent_name="holding_corr", status="fallback")).token_usage,
        )["total_tokens"]
        report.stage3_tokens = self.report_generation_agent.last_usage["total_tokens"]
        report.total_elapsed_ms = (perf_counter() - started) * 1000
        return report
