from __future__ import annotations

import json
import logging
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from fin_stock_agent.core.config import get_config
from fin_stock_agent.memory.vector_store import get_vector_store
from fin_stock_agent.reporting.models import DailyReport
from fin_stock_agent.storage.database import get_session
from fin_stock_agent.storage.models import DailyReportDigestORM

logger = logging.getLogger(__name__)


def _sentiment_marker(label: str) -> str:
    return {
        "bullish": "[bullish]",
        "watch": "[watch]",
        "neutral": "[neutral]",
        "cautious": "[cautious]",
        "bearish": "[bearish]",
    }.get(label, "")


class DailyReportDigestService:
    """Extract and inject lightweight daily-report harnesses for memory context."""

    def write_digest(self, report: DailyReport) -> None:
        holdings_digest = []
        buy_count = hold_count = sell_count = 0
        vec_id = f"{report.user_id}_{report.report_date}"

        for status in report.fund_statuses:
            action = status.action.lower()
            if action == "buy":
                buy_count += 1
            elif action == "sell":
                sell_count += 1
            else:
                hold_count += 1
            holdings_digest.append(
                {
                    "ts_code": status.ts_code,
                    "name": status.name,
                    "action": action,
                    "confidence": round(status.confidence, 2),
                    "trend": status.trend,
                    "key_risks": status.key_risks[:2],
                }
            )

        digest = DailyReportDigestORM(
            user_id=report.user_id,
            report_date=report.report_date,
            sentiment_label=report.news_sentiment_label or "neutral",
            overall_summary=(report.overall_summary or "")[:400],
            market_context=(report.market_context or "")[:600],
            holdings_digest_json=json.dumps(holdings_digest, ensure_ascii=False),
            buy_count=buy_count,
            hold_count=hold_count,
            sell_count=sell_count,
            total_pnl_pct=report.total_unrealized_pnl_pct if report.total_unrealized_pnl_pct else None,
            vec_id=vec_id,
            created_at=datetime.utcnow(),
        )

        try:
            with get_session() as session:
                existing = session.execute(
                    select(DailyReportDigestORM).where(
                        DailyReportDigestORM.user_id == report.user_id,
                        DailyReportDigestORM.report_date == report.report_date,
                    )
                ).scalar_one_or_none()
                if existing is None:
                    session.add(digest)
                else:
                    existing.sentiment_label = digest.sentiment_label
                    existing.overall_summary = digest.overall_summary
                    existing.market_context = digest.market_context
                    existing.holdings_digest_json = digest.holdings_digest_json
                    existing.buy_count = digest.buy_count
                    existing.hold_count = digest.hold_count
                    existing.sell_count = digest.sell_count
                    existing.total_pnl_pct = digest.total_pnl_pct
                    existing.vec_id = digest.vec_id
        except IntegrityError:
            logger.debug(
                "DailyReportDigest upsert: IntegrityError for user=%s date=%s (concurrent write or duplicate); skipping insert",
                report.user_id, report.report_date,
            )

        try:
            get_vector_store().upsert(
                self._collection_name(report.user_id),
                vec_id,
                f"{report.overall_summary} {report.market_context}".strip(),
                {"user_id": report.user_id, "report_date": report.report_date},
            )
        except Exception as exc:
            logger.warning("Daily report digest vector upsert failed: %s", exc)

    def get_recent_digests(self, user_id: str, limit: int = 5) -> list[DailyReportDigestORM]:
        """Load digest rows. Expunge before returning so callers can read attributes after
        the session context closes (avoids "not bound to a Session" on lazy/expired state).
        """
        with get_session() as session:
            rows = list(
                session.execute(
                    select(DailyReportDigestORM)
                    .where(DailyReportDigestORM.user_id == user_id)
                    .order_by(DailyReportDigestORM.report_date.desc())
                    .limit(limit)
                ).scalars()
            )
            for row in rows:
                session.expunge(row)
            return rows

    def build_digest_context(self, user_id: str, limit: int = 5) -> str:
        digests = self.get_recent_digests(user_id=user_id, limit=limit)
        if not digests:
            return "## Recent daily report digests\nNo daily reports generated yet."

        lines = [f"## Recent daily report digests (last {len(digests)} days)"]
        for row in digests:
            marker = _sentiment_marker(row.sentiment_label or "")
            sentiment = f"{marker}{row.sentiment_label}" if row.sentiment_label else ""
            summary_text = (row.overall_summary or "").replace("\n", " ")
            line = f"- {row.report_date} {sentiment} {summary_text}".strip()
            if row.buy_count or row.hold_count or row.sell_count:
                line += f" (buy:{row.buy_count} hold:{row.hold_count} sell:{row.sell_count})"

            holdings_detail: list[str] = []
            if row.holdings_digest_json:
                try:
                    items: list[dict] = json.loads(row.holdings_digest_json)
                    for item in items:
                        name = item.get("name") or item.get("ts_code", "")
                        action = item.get("action", "hold")
                        conf = item.get("confidence", 0.5)
                        holdings_detail.append(f"{name}->{action}({conf})")
                except (json.JSONDecodeError, TypeError):
                    pass
            if holdings_detail:
                line += "\n  " + ", ".join(holdings_detail)
            lines.append(line)
        return "\n".join(lines)

    def search_relevant_digests(self, user_id: str, question: str) -> list[str]:
        cfg = get_config().memory.semantic_search
        try:
            results = get_vector_store().search(
                self._collection_name(user_id),
                question,
                top_k=cfg.digest_top_k,
                threshold=cfg.similarity_threshold,
            )
            if results:
                return [item.text for item in results]
        except Exception as exc:
            logger.warning("Digest semantic search failed, falling back: %s", exc)
        rows = self.get_recent_digests(user_id=user_id, limit=cfg.time_fallback_limit)
        return [
            " ".join(part for part in [row.overall_summary or "", row.market_context or ""] if part).strip()
            for row in rows
        ]

    def clear_user(self, user_id: str) -> None:
        with get_session() as session:
            rows = session.execute(
                select(DailyReportDigestORM).where(DailyReportDigestORM.user_id == user_id)
            ).scalars().all()
            for row in rows:
                session.delete(row)
        try:
            get_vector_store().delete_collection(self._collection_name(user_id))
        except Exception:
            logger.debug("Ignoring digest collection delete failure for %s", user_id)

    @staticmethod
    def _collection_name(user_id: str) -> str:
        return f"{user_id}_digests"
