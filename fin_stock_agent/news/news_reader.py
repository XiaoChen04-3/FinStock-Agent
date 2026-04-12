"""NewsReader — fetch, deduplicate, persist, and query financial news.

News is sourced directly from CLS, Eastmoney, and THS via
`china_finance_fetcher` (no external HTTP bridge required).
Fetched items are stored in `NewsCacheORM` and can be queried with optional
keyword filtering.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime
from zoneinfo import ZoneInfo

from sqlalchemy import select

from fin_stock_agent.core.settings import settings
from fin_stock_agent.init.trade_calendar import TradingCalendar
from fin_stock_agent.news.china_finance_fetcher import fetch_all_sources
from fin_stock_agent.news.models import NewsFetchResult, NewsItem
from fin_stock_agent.storage.database import get_session
from fin_stock_agent.storage.models import NewsCacheORM

logger = logging.getLogger(__name__)


class NewsReader:
    def __init__(self) -> None:
        self.tz = ZoneInfo(settings.app_timezone)
        self.trade_calendar = TradingCalendar()

    def _resolve_since(self, since: datetime | None) -> datetime:
        if since is None:
            return datetime.now(self.tz).replace(hour=0, minute=0, second=0, microsecond=0)
        if since.tzinfo is None:
            return since.replace(tzinfo=self.tz)
        return since.astimezone(self.tz)

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch_today_sync(self, since: datetime | None = None) -> NewsFetchResult:
        """Sync wrapper that defaults to today's news, or accepts a custom threshold."""
        try:
            return asyncio.run(self.fetch_today(since=since))
        except RuntimeError:
            # Already inside a running event loop (e.g. Streamlit with async support)
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self.fetch_today(since=since))
            finally:
                loop.close()

    async def fetch_today(self, since: datetime | None = None) -> NewsFetchResult:
        """Fetch news newer than or equal to *since*, defaulting to today's midnight."""
        if not settings.enable_news_fetch:
            return NewsFetchResult(items=[], fetched_sources=[], degraded=True, message="news fetching disabled")

        threshold = self._resolve_since(since)
        try:
            raw_items = await fetch_all_sources(since=threshold)
        except Exception as exc:
            logger.warning("fetch_all_sources failed: %s", exc)
            return NewsFetchResult(items=[], fetched_sources=[], degraded=True, message=str(exc))

        seen_urls: set[str] = set()
        kept: list[NewsItem] = []
        sources_seen: set[str] = set()

        for item in raw_items:
            if not item.url or item.url in seen_urls:
                continue
            # Filter to the requested threshold only.
            if item.published_at:
                local_pub = item.published_at.astimezone(self.tz)
                if local_pub < threshold:
                    continue
            seen_urls.add(item.url)
            sources_seen.add(item.source)
            kept.append(item)
            self._save_item(item)

        self.prune_cache(retain_trading_days=3)

        return NewsFetchResult(
            items=kept[:200],
            fetched_sources=sorted(sources_seen),
        )

    def get_cached_news(self, keywords: list[str] | None = None, limit: int = 20) -> list[dict]:
        """Return recently cached news from DB, optionally filtered by keywords."""
        keywords = [kw for kw in (keywords or []) if kw]
        self.prune_cache(retain_trading_days=3)
        with get_session() as session:
            rows = session.execute(
                select(NewsCacheORM).order_by(NewsCacheORM.published_at.desc()).limit(200)
            ).scalars()
            items: list[dict] = []
            for row in rows:
                blob = f"{row.title} {row.summary or ''}"
                if keywords and not any(kw in blob for kw in keywords):
                    continue
                items.append(
                    {
                        "title": row.title,
                        "summary": row.summary or "",
                        "url": row.url,
                        "source": row.source or "",
                        "published_at": row.published_at.isoformat() if row.published_at else None,
                    }
                )
                if len(items) >= limit:
                    break
        return items

    def prune_cache(self, retain_trading_days: int = 3) -> int:
        recent_days = self.trade_calendar.get_recent_trading_days(retain_trading_days)
        if not recent_days:
            return 0
        cutoff_date = datetime.strptime(recent_days[-1], "%Y%m%d").date()
        removed = 0
        with get_session() as session:
            rows = session.execute(select(NewsCacheORM)).scalars().all()
            for row in rows:
                anchor = row.published_at or row.fetched_at
                if self._local_date(anchor) < cutoff_date:
                    session.delete(row)
                    removed += 1
        if removed:
            logger.info("News cache pruned: removed=%s cutoff=%s", removed, cutoff_date.isoformat())
        return removed

    # ── Internal ──────────────────────────────────────────────────────────────

    def _save_item(self, item: NewsItem) -> None:
        if not item.url:
            return
        try:
            with get_session() as session:
                existing = session.execute(
                    select(NewsCacheORM).where(NewsCacheORM.url == item.url)
                ).scalar_one_or_none()
                if existing is not None:
                    return
                session.add(
                    NewsCacheORM(
                        url=item.url,
                        title=item.title[:500],
                        summary=item.summary,
                        source=item.source,
                        published_at=item.published_at,
                    )
                )
        except Exception as exc:
            logger.debug("Failed to persist news item %s: %s", item.url, exc)

    def _local_date(self, value: datetime | None) -> date:
        if value is None:
            return datetime.now(self.tz).date()
        if value.tzinfo is None:
            return value.replace(tzinfo=self.tz).date()
        return value.astimezone(self.tz).date()
