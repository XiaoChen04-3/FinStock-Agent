from __future__ import annotations

import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

from fin_stock_agent.core.settings import settings
from fin_stock_agent.news import china_finance_fetcher as fetcher
from fin_stock_agent.news import news_reader
from fin_stock_agent.news.models import NewsItem
from fin_stock_agent.news.news_reader import NewsReader


def _to_epoch(dt: datetime) -> int:
    return int(dt.timestamp())


def test_fetch_cls_telegraph_paginates_until_since(monkeypatch) -> None:
    tz = ZoneInfo(settings.app_timezone)
    dt_1100 = datetime(2026, 4, 12, 11, 0, tzinfo=tz)
    dt_1000 = datetime(2026, 4, 12, 10, 0, tzinfo=tz)
    dt_0930 = datetime(2026, 4, 12, 9, 30, tzinfo=tz)
    dt_0850 = datetime(2026, 4, 12, 8, 50, tzinfo=tz)
    since = datetime(2026, 4, 12, 9, 0, tzinfo=tz)
    start_cursor = _to_epoch(dt_1100)

    class FakeDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return dt_1100 if tz is None else dt_1100.astimezone(tz)

    monkeypatch.setattr(fetcher, "datetime", FakeDateTime)

    pages = {
        start_cursor: [
            {"id": 301, "ctime": _to_epoch(dt_1100), "brief": "11:00"},
            {"id": 300, "ctime": _to_epoch(dt_1000), "brief": "10:00"},
        ],
        _to_epoch(dt_1000): [
            {"id": 299, "ctime": _to_epoch(dt_0930), "brief": "09:30"},
            {"id": 298, "ctime": _to_epoch(dt_0850), "brief": "08:50"},
        ],
    }
    calls: list[int] = []

    async def fake_fetch_cls_page(session, last_time: int, page_size: int = 50) -> list[dict]:
        calls.append(last_time)
        return pages.get(last_time, [])

    monkeypatch.setattr(fetcher, "_fetch_cls_page", fake_fetch_cls_page)

    items = asyncio.run(fetcher.fetch_cls_telegraph(session=None, max_pages=5, since=since))

    assert calls == [start_cursor, _to_epoch(dt_1000)]
    assert [item.title for item in items] == ["11:00", "10:00", "09:30"]
    assert all(item.published_at is not None and item.published_at >= since for item in items)


def test_cls_sign_matches_sorted_query_hash() -> None:
    params = {
        "app": "CailianpressWeb",
        "category": "all",
        "last_time": 1775973389,
        "os": "web",
        "refresh_type": 1,
        "rn": 20,
        "sv": "8.4.6",
    }

    assert fetcher._cls_sign(params) == "b9b6e121093bce74fb3fbabdb8ad1406"


def test_news_reader_fetch_today_uses_custom_since(monkeypatch) -> None:
    tz = ZoneInfo(settings.app_timezone)
    since = datetime(2026, 4, 12, 9, 30, tzinfo=tz)
    captured_since: list[datetime] = []

    async def fake_fetch_all_sources(
        max_pages: int = 20,
        since: datetime | None = None,
    ) -> list[NewsItem]:
        captured_since.append(since)
        return [
            NewsItem(
                title="older",
                summary="",
                url="https://example.com/old",
                source="cls",
                published_at=datetime(2026, 4, 12, 9, 0, tzinfo=tz),
            ),
            NewsItem(
                title="kept-cls",
                summary="",
                url="https://example.com/new",
                source="cls",
                published_at=datetime(2026, 4, 12, 9, 35, tzinfo=tz),
            ),
            NewsItem(
                title="dup-cls",
                summary="",
                url="https://example.com/new",
                source="cls",
                published_at=datetime(2026, 4, 12, 9, 36, tzinfo=tz),
            ),
            NewsItem(
                title="kept-ths",
                summary="",
                url="https://example.com/ths",
                source="ths",
                published_at=datetime(2026, 4, 12, 10, 0, tzinfo=tz),
            ),
        ]

    monkeypatch.setattr(news_reader, "fetch_all_sources", fake_fetch_all_sources)
    monkeypatch.setattr(NewsReader, "_save_item", lambda self, item: None)

    result = asyncio.run(NewsReader().fetch_today(since=since))

    assert captured_since == [since]
    assert [item.title for item in result.items] == ["kept-cls", "kept-ths"]
    assert result.fetched_sources == ["cls", "ths"]
