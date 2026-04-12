from __future__ import annotations

import time

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from fin_stock_agent.core.llm import merge_token_usage
from fin_stock_agent.core.time_utils import today_local_str
from fin_stock_agent.init.name_resolver import NameResolver
from fin_stock_agent.init.trade_calendar import TradingCalendar
from fin_stock_agent.news.news_reader import NewsReader
from fin_stock_agent.reporting.agentic_news_analyzer import AgenticNewsAnalyzer
from fin_stock_agent.reporting.daily_briefing_agent import DailyBriefingAgent
from fin_stock_agent.reporting.fund_analysis_agent import FundAnalysisAgent
from fin_stock_agent.reporting.fund_fetcher import TushareFundFetcher
from fin_stock_agent.reporting.models import DailyReport
from fin_stock_agent.reporting.news_analysis_agent import NewsAnalysisAgent
from fin_stock_agent.reporting.report_synthesis_agent import ReportSynthesisAgent
from fin_stock_agent.services.daily_report_digest_service import DailyReportDigestService
from fin_stock_agent.services.portfolio_service import PortfolioService
from fin_stock_agent.stats.tracker import write_stats_event
from fin_stock_agent.storage.cache import get_cache
from fin_stock_agent.storage.database import get_session
from fin_stock_agent.storage.models import DailyReportORM


class DailyReporter:
    def __init__(self) -> None:
        self.cache = get_cache()
        self.portfolio_service = PortfolioService()
        self.trade_calendar = TradingCalendar()
        self.news_reader = NewsReader()
        self.fund_fetcher = TushareFundFetcher()
        self.news_agent = NewsAnalysisAgent()
        self.briefing_agent = DailyBriefingAgent()
        self.agentic_analyzer = AgenticNewsAnalyzer()
        self.fund_agent = FundAnalysisAgent()
        self.synthesis_agent = ReportSynthesisAgent()
        self.name_resolver = NameResolver()
        self.digest_service = DailyReportDigestService()

    def resolve_report_date(self, date: str | None = None) -> str:
        return date or today_local_str()

    def get_existing_report(self, user_id: str, date: str | None = None) -> DailyReport | None:
        report_date = self.resolve_report_date(date)
        cache_key = f"daily_report:{user_id}:{report_date}"
        cached = self.cache.get(cache_key)
        if cached:
            return DailyReport.model_validate_json(cached)

        with get_session() as session:
            existing = session.execute(
                select(DailyReportORM).where(
                    DailyReportORM.user_id == user_id,
                    DailyReportORM.report_date == report_date,
                )
            ).scalar_one_or_none()
            if existing is None:
                return None
            self.cache.setex(cache_key, 12 * 3600, existing.report_json)
            return DailyReport.model_validate_json(existing.report_json)

    def generate(self, user_id: str, date: str | None = None, force: bool = False) -> DailyReport:
        report_date = self.resolve_report_date(date)
        cache_key = f"daily_report:{user_id}:{report_date}"
        if not force:
            existing = self.get_existing_report(user_id=user_id, date=report_date)
            if existing is not None:
                write_stats_event(
                    "daily_report_cache_hit",
                    user_id=user_id,
                    report_date=report_date,
                    has_holdings=bool(existing.fund_statuses),
                    fund_status_count=len(existing.fund_statuses),
                    top_news_count=len(existing.top_news),
                    generated_at=existing.generated_at.isoformat(),
                )
                return existing

        started = time.perf_counter()
        stage_started = started
        stage_ms: dict[str, float] = {}

        try:
            holdings = self.portfolio_service.get_holdings(user_id)
            codes = [item["ts_code"] for item in holdings]
            keywords = self.name_resolver.get_keywords_for_holdings(codes)
            recent_trading_days = self.trade_calendar.get_recent_trading_days(3)
            stage_ms["load_portfolio"] = (time.perf_counter() - stage_started) * 1000

            stage_started = time.perf_counter()
            news_result = self.news_reader.fetch_today_sync()
            daily_briefing = self.briefing_agent.run(news_result.items)
            news_ctx = self.news_agent.run(news_result.items, keywords, daily_briefing=daily_briefing)
            stage_ms["analyze_news"] = (time.perf_counter() - stage_started) * 1000
            news_usage = {
                "daily_briefing": dict(self.briefing_agent.last_usage),
                "news_analysis": dict(self.news_agent.last_usage),
            }

            stage_started = time.perf_counter()
            if holdings:
                nav_history = self.fund_fetcher.fetch_history(codes, years=3)
                fund_ctx = self.fund_agent.run(holdings, nav_history, recent_trading_days)
                holding_recs = self.agentic_analyzer.analyze(
                    holdings,
                    news_result.items,
                    daily_briefing=daily_briefing,
                    name_keywords=keywords,
                )
            else:
                fund_ctx = {"analyses": {}}
                holding_recs = {}
            stage_ms["analyze_holdings"] = (time.perf_counter() - stage_started) * 1000
            holdings_usage = {
                "fund_analysis": dict(self.fund_agent.last_usage),
                "agentic_news": dict(self.agentic_analyzer.last_usage),
            }

            stage_started = time.perf_counter()
            report = self.synthesis_agent.run(
                user_id=user_id,
                report_date=report_date,
                recent_trading_days=recent_trading_days,
                holdings=holdings,
                news_ctx=news_ctx,
                fund_ctx=fund_ctx,
                holding_recommendations=holding_recs,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
            stage_ms["synthesize_report"] = (time.perf_counter() - stage_started) * 1000
            synthesis_usage = dict(self.synthesis_agent.last_usage)

            stage1_usage = merge_token_usage(*news_usage.values())
            stage2_usage = merge_token_usage(*holdings_usage.values())
            stage3_usage = merge_token_usage(synthesis_usage)
            report_usage = {
                "daily_briefing": news_usage["daily_briefing"],
                "news_analysis": news_usage["news_analysis"],
                "fund_analysis": holdings_usage["fund_analysis"],
                "agentic_news": holdings_usage["agentic_news"],
                "report_synthesis": synthesis_usage,
                "stage1": stage1_usage,
                "stage2": stage2_usage,
                "stage3": stage3_usage,
                "total": merge_token_usage(stage1_usage, stage2_usage, stage3_usage),
            }
            report = report.model_copy(
                update={
                    "stage1_tokens": stage1_usage["total_tokens"],
                    "stage2_tokens": stage2_usage["total_tokens"],
                    "stage3_tokens": stage3_usage["total_tokens"],
                }
            )

            payload = report.model_dump_json()
            self.cache.setex(cache_key, 12 * 3600, payload)
            self._save_report(
                user_id=user_id,
                report_date=report_date,
                payload=payload,
                elapsed_ms=report.total_elapsed_ms,
            )
            write_stats_event(
                "daily_report_generated",
                user_id=user_id,
                report_date=report_date,
                holdings_count=len(holdings),
                holdings_codes=codes,
                keyword_count=len(keywords),
                recent_trading_days=recent_trading_days,
                news_count=len(news_result.items),
                fetched_sources=news_result.fetched_sources,
                briefing_top10_count=len((daily_briefing or {}).get("top_10") or []),
                fund_status_count=len(report.fund_statuses),
                top_news_count=len(report.top_news),
                total_elapsed_ms=report.total_elapsed_ms,
                stage_elapsed_ms=stage_ms,
                report_llm_usage=report_usage,
                prompt_tokens=report_usage["total"]["prompt_tokens"],
                completion_tokens=report_usage["total"]["completion_tokens"],
                total_tokens=report_usage["total"]["total_tokens"],
                stage1_tokens=report.stage1_tokens,
                stage2_tokens=report.stage2_tokens,
                stage3_tokens=report.stage3_tokens,
            )
            return report
        except Exception as exc:
            write_stats_event(
                "daily_report_failed",
                user_id=user_id,
                report_date=report_date,
                has_error=True,
                error=str(exc),
                total_elapsed_ms=(time.perf_counter() - started) * 1000,
                stage_elapsed_ms=stage_ms,
            )
            raise

    def _save_report(self, *, user_id: str, report_date: str, payload: str, elapsed_ms: float) -> None:
        report = DailyReport.model_validate_json(payload)
        try:
            with get_session() as session:
                existing = session.execute(
                    select(DailyReportORM).where(
                        DailyReportORM.user_id == user_id,
                        DailyReportORM.report_date == report_date,
                    )
                ).scalar_one_or_none()
                if existing is None:
                    session.add(
                        DailyReportORM(
                            user_id=user_id,
                            report_date=report_date,
                            report_json=payload,
                            model_name="deterministic-skeleton",
                            stage1_tokens=report.stage1_tokens,
                            stage2_tokens=report.stage2_tokens,
                            stage3_tokens=report.stage3_tokens,
                            elapsed_ms=elapsed_ms,
                        )
                    )
                else:
                    existing.report_json = payload
                    existing.model_name = "deterministic-skeleton"
                    existing.stage1_tokens = report.stage1_tokens
                    existing.stage2_tokens = report.stage2_tokens
                    existing.stage3_tokens = report.stage3_tokens
                    existing.elapsed_ms = elapsed_ms
        except IntegrityError:
            with get_session() as session:
                existing = session.execute(
                    select(DailyReportORM).where(
                        DailyReportORM.user_id == user_id,
                        DailyReportORM.report_date == report_date,
                    )
                ).scalar_one_or_none()
                if existing is not None:
                    existing.report_json = payload
                    existing.model_name = "deterministic-skeleton"
                    existing.stage1_tokens = report.stage1_tokens
                    existing.stage2_tokens = report.stage2_tokens
                    existing.stage3_tokens = report.stage3_tokens
                    existing.elapsed_ms = elapsed_ms

        try:
            self.digest_service.write_digest(report)
        except Exception:
            pass
