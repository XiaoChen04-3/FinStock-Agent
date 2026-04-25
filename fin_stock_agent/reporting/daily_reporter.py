from __future__ import annotations

import time

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from fin_stock_agent.core.config import get_config
from fin_stock_agent.core.time_utils import today_local_str
from fin_stock_agent.reporting.models import DailyReport
from fin_stock_agent.reporting.orchestrator import OrchestratorAgent
from fin_stock_agent.services.daily_report_digest_service import DailyReportDigestService
from fin_stock_agent.stats.tracker import write_stats_event
from fin_stock_agent.storage.cache import get_cache
from fin_stock_agent.storage.database import get_session
from fin_stock_agent.storage.models import DailyReportORM


class DailyReporter:
    def __init__(self) -> None:
        self.cache = get_cache()
        self.orchestrator = OrchestratorAgent()
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
            ttl = get_config().daily_report.cache_ttl_hours * 3600
            self.cache.setex(cache_key, ttl, existing.report_json)
            return DailyReport.model_validate_json(existing.report_json)

    def generate(self, user_id: str, date: str | None = None, force: bool = False) -> DailyReport:
        report_date = self.resolve_report_date(date)
        cache_key = f"daily_report:{user_id}:{report_date}"
        ttl = get_config().daily_report.cache_ttl_hours * 3600
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
        try:
            report = self.orchestrator.run(user_id=user_id, date=report_date, force=force)
            payload = report.model_dump_json()
            self.cache.setex(cache_key, ttl, payload)
            self._save_report(user_id=user_id, report_date=report_date, payload=payload, elapsed_ms=report.total_elapsed_ms)
            self.digest_service.write_digest(report)
            write_stats_event(
                "daily_report_generated",
                user_id=user_id,
                report_date=report_date,
                holdings_count=len(report.fund_statuses),
                fund_status_count=len(report.fund_statuses),
                top_news_count=len(report.top_news),
                total_elapsed_ms=report.total_elapsed_ms,
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
                            model_name="orchestrator",
                            stage1_tokens=report.stage1_tokens,
                            stage2_tokens=report.stage2_tokens,
                            stage3_tokens=report.stage3_tokens,
                            elapsed_ms=elapsed_ms,
                        )
                    )
                else:
                    existing.report_json = payload
                    existing.model_name = "orchestrator"
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
                    existing.model_name = "orchestrator"
                    existing.stage1_tokens = report.stage1_tokens
                    existing.stage2_tokens = report.stage2_tokens
                    existing.stage3_tokens = report.stage3_tokens
                    existing.elapsed_ms = elapsed_ms
