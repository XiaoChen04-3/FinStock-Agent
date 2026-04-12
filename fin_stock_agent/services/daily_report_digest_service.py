from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from fin_stock_agent.reporting.models import DailyReport
from fin_stock_agent.storage.database import get_session
from fin_stock_agent.storage.models import DailyReportDigestORM


def _sentiment_emoji(label: str) -> str:
    return {
        "bullish": "📈",
        "watch": "👀",
        "neutral": "➖",
        "cautious": "⚠️",
        "bearish": "📉",
    }.get(label, "")


class DailyReportDigestService:
    """Extract and inject lightweight daily-report harnesses for memory context."""

    def write_digest(self, report: DailyReport) -> None:
        """Extract a digest from a freshly generated DailyReport and persist it."""
        holdings_digest = []
        buy_count = hold_count = sell_count = 0

        for fs in report.fund_statuses:
            action = fs.action.lower()
            if action == "buy":
                buy_count += 1
            elif action == "sell":
                sell_count += 1
            else:
                hold_count += 1

            holdings_digest.append(
                {
                    "ts_code": fs.ts_code,
                    "name": fs.name,
                    "action": action,
                    "confidence": round(fs.confidence, 2),
                    "trend": fs.trend,
                    "key_risks": fs.key_risks[:2],
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
        except IntegrityError:
            pass

    def get_recent_digests(self, user_id: str, limit: int = 5) -> list[DailyReportDigestORM]:
        with get_session() as session:
            rows = session.execute(
                select(DailyReportDigestORM)
                .where(DailyReportDigestORM.user_id == user_id)
                .order_by(DailyReportDigestORM.report_date.desc())
                .limit(limit)
            ).scalars()
            return list(rows)

    def build_digest_context(self, user_id: str, limit: int = 5) -> str:
        digests = self.get_recent_digests(user_id=user_id, limit=limit)
        if not digests:
            return "## Recent daily report digests\nNo daily reports generated yet."

        lines = [f"## Recent daily report digests (last {len(digests)} days)"]
        for row in digests:
            emoji = _sentiment_emoji(row.sentiment_label or "")
            sentiment = f"{emoji}[{row.sentiment_label}]" if row.sentiment_label else ""
            summary_text = (row.overall_summary or "").replace("\n", " ")
            line = f"- {row.report_date} {sentiment} {summary_text}"
            if row.buy_count or row.hold_count or row.sell_count:
                line += f"  (buy:{row.buy_count} hold:{row.hold_count} sell:{row.sell_count})"

            holdings_detail: list[str] = []
            if row.holdings_digest_json:
                try:
                    items: list[dict] = json.loads(row.holdings_digest_json)
                    for item in items:
                        name = item.get("name") or item.get("ts_code", "")
                        action = item.get("action", "hold")
                        conf = item.get("confidence", 0.5)
                        holdings_detail.append(f"{name}→{action}({conf})")
                except (json.JSONDecodeError, TypeError):
                    pass

            if holdings_detail:
                line += "\n  " + ", ".join(holdings_detail)

            lines.append(line)

        return "\n".join(lines)
