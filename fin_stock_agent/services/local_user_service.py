from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from sqlalchemy import select

from fin_stock_agent.core.settings import settings
from fin_stock_agent.memory.user_profile_file import get_user_profile_file_service
from fin_stock_agent.stats.tracker import write_stats_event
from fin_stock_agent.storage.database import get_session
from fin_stock_agent.storage.models import (
    ConversationSummaryORM,
    DailyReportORM,
    TradeRecordORM,
    UserMemoryEventORM,
    UserMemoryProfileORM,
)


class LocalUserService:
    def canonical_user_id(self) -> str:
        return settings.user_id_seed or "local-user"

    def consolidate_legacy_data(self, target_user_id: str | None = None) -> dict[str, int]:
        canonical_user_id = target_user_id or self.canonical_user_id()
        summary = {
            "legacy_user_count": 0,
            "trade_rows_migrated": 0,
            "conversation_rows_migrated": 0,
            "memory_event_rows_migrated": 0,
            "memory_profile_rows_migrated": 0,
            "report_rows_migrated": 0,
            "report_rows_deleted": 0,
        }

        with get_session() as session:
            legacy_user_ids = self._load_legacy_user_ids(session, canonical_user_id)
            summary["legacy_user_count"] = len(legacy_user_ids)
            if not legacy_user_ids:
                return summary

            trade_rows = session.execute(
                select(TradeRecordORM).where(TradeRecordORM.user_id.in_(legacy_user_ids))
            ).scalars().all()
            for row in trade_rows:
                row.user_id = canonical_user_id
            summary["trade_rows_migrated"] = len(trade_rows)

            conversation_rows = session.execute(
                select(ConversationSummaryORM).where(ConversationSummaryORM.user_id.in_(legacy_user_ids))
            ).scalars().all()
            for row in conversation_rows:
                row.user_id = canonical_user_id
            summary["conversation_rows_migrated"] = len(conversation_rows)

            memory_event_rows = session.execute(
                select(UserMemoryEventORM).where(UserMemoryEventORM.user_id.in_(legacy_user_ids))
            ).scalars().all()
            summary["memory_event_rows_migrated"] = len(memory_event_rows)

            memory_profile_rows = session.execute(
                select(UserMemoryProfileORM).where(UserMemoryProfileORM.user_id.in_(legacy_user_ids + [canonical_user_id]))
            ).scalars().all()
            if memory_profile_rows:
                winner = self._pick_memory_profile_winner(memory_profile_rows, canonical_user_id)
                self._export_memory_profile_if_needed(winner)
                summary["memory_profile_rows_migrated"] = len(memory_profile_rows)

            report_rows = session.execute(
                select(DailyReportORM).where(
                    DailyReportORM.user_id.in_(legacy_user_ids + [canonical_user_id])
                )
            ).scalars().all()
            grouped: dict[str, list[DailyReportORM]] = defaultdict(list)
            for row in report_rows:
                grouped[row.report_date].append(row)

            for report_date, rows in grouped.items():
                winner = self._pick_report_winner(rows, canonical_user_id)
                if winner.user_id != canonical_user_id:
                    winner.user_id = canonical_user_id
                    summary["report_rows_migrated"] += 1

                for row in rows:
                    if row is winner:
                        continue
                    if row.user_id != canonical_user_id:
                        summary["report_rows_migrated"] += 1
                    session.delete(row)
                    summary["report_rows_deleted"] += 1

        if any(summary[key] for key in summary if key != "legacy_user_count") or summary["legacy_user_count"] > 0:
            write_stats_event(
                "single_user_migration",
                user_id=canonical_user_id,
                migrated_to=canonical_user_id,
                **summary,
            )
        return summary

    def _load_legacy_user_ids(self, session, canonical_user_id: str) -> list[str]:
        ids: set[str] = set()
        for model in (
            TradeRecordORM,
            ConversationSummaryORM,
            DailyReportORM,
            UserMemoryEventORM,
            UserMemoryProfileORM,
        ):
            rows = session.execute(select(model.user_id)).scalars().all()
            for user_id in rows:
                if user_id and user_id != canonical_user_id:
                    ids.add(user_id)
        return sorted(ids)

    def _pick_report_winner(self, rows: list[DailyReportORM], canonical_user_id: str) -> DailyReportORM:
        canonical_rows = [row for row in rows if row.user_id == canonical_user_id]
        if canonical_rows:
            if len(canonical_rows) == 1:
                return canonical_rows[0]
            return max(canonical_rows, key=self._report_sort_key)
        return max(rows, key=self._report_sort_key)

    def _report_sort_key(self, row: DailyReportORM) -> tuple[datetime, int]:
        return (row.created_at or datetime.min, row.id or 0)

    def _pick_memory_profile_winner(
        self,
        rows: list[UserMemoryProfileORM],
        canonical_user_id: str,
    ) -> UserMemoryProfileORM:
        canonical_rows = [row for row in rows if row.user_id == canonical_user_id]
        if canonical_rows:
            return max(canonical_rows, key=lambda row: row.updated_at or datetime.min)
        return max(rows, key=lambda row: row.updated_at or datetime.min)

    def _export_memory_profile_if_needed(self, row: UserMemoryProfileORM) -> None:
        try:
            from fin_stock_agent.services.user_memory_service import UserMemoryService

            service = get_user_profile_file_service()
            service.initialize(initial_text=UserMemoryService()._legacy_row_to_markdown(row))
        except Exception:
            return
