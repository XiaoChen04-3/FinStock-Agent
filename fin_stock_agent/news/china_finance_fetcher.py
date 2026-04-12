"""Direct fetchers for Chinese financial news APIs.

Calls the upstream JSON APIs from CLS, Eastmoney, and THS directly inside the
application process and normalizes them into ``NewsItem`` objects.
"""
from __future__ import annotations

import asyncio
import json
import hashlib
import logging
import re
from urllib.parse import urlencode
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import aiohttp

from fin_stock_agent.core.settings import settings
from fin_stock_agent.news.models import NewsItem

logger = logging.getLogger(__name__)

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
_TIMEOUT = aiohttp.ClientTimeout(total=12)
_TZ = ZoneInfo(settings.app_timezone)


def _today_start_local() -> datetime:
    """Midnight of today in the app timezone."""
    local_now = datetime.now(_TZ)
    return local_now.replace(hour=0, minute=0, second=0, microsecond=0)


def _normalize_since(since: datetime | None) -> datetime:
    """Return a timezone-aware local datetime, defaulting to today's midnight."""
    if since is None:
        return _today_start_local()
    if since.tzinfo is None:
        return since.replace(tzinfo=_TZ)
    return since.astimezone(_TZ)


def _is_before_threshold(dt: datetime | None, threshold: datetime) -> bool:
    """Return True when *dt* is older than the inclusive threshold."""
    if dt is None:
        return False
    return dt.astimezone(timezone.utc) < threshold.astimezone(timezone.utc)


def _from_epoch_to_local(ts: int | str | None) -> datetime | None:
    """Normalize second-level epoch timestamps to the configured app timezone."""
    try:
        value = int(ts or 0)
    except Exception:
        return None
    if value <= 0:
        return None
    return datetime.fromtimestamp(value, tz=_TZ)


def _build_cls_news_item(article: dict) -> NewsItem | None:
    if not isinstance(article, dict):
        return None
    article_id = int(article.get("id") or 0)
    if article_id <= 0:
        return None
    dt = _from_epoch_to_local(article.get("ctime"))
    if dt is None:
        return None
    title = article.get("title") or article.get("brief") or article.get("content") or ""
    summary = article.get("content") or article.get("brief") or ""
    return NewsItem(
        title=str(title)[:200],
        summary=str(summary),
        url=f"https://www.cls.cn/detail/{article_id}",
        source="cls",
        published_at=dt,
    )


def _cls_sign(params: dict[str, str | int]) -> str:
    """Mirror CLS frontend signing: sorted query string -> SHA1 -> MD5."""
    items = sorted((key, str(value)) for key, value in params.items())
    query = urlencode(items)
    sha1_digest = hashlib.sha1(query.encode("utf-8")).hexdigest()
    return hashlib.md5(sha1_digest.encode("utf-8")).hexdigest()


async def _fetch_cls_page(
    session: aiohttp.ClientSession,
    last_time: int,
    page_size: int = 50,
) -> list[dict]:
    """Fetch one CLS telegraph page using the signed frontend API."""
    params: dict[str, str | int] = {
        "app": "CailianpressWeb",
        "category": "all",
        "last_time": last_time,
        "os": "web",
        "refresh_type": 1,
        "rn": page_size,
        "sv": "8.4.6",
    }
    params["sign"] = _cls_sign(params)
    try:
        async with session.get(
            "https://www.cls.cn/v1/roll/get_roll_list",
            headers={
                "User-Agent": _UA,
                "Referer": "https://www.cls.cn/telegraph",
                "Accept": "application/json, text/plain, */*",
                "X-Requested-With": "XMLHttpRequest",
            },
            params=params,
            timeout=_TIMEOUT,
        ) as resp:
            if resp.status != 200:
                return []
            payload = await resp.json(content_type=None)
    except Exception as exc:
        logger.warning("CLS telegraph page fetch failed at cursor %s: %s", last_time, exc)
        return []

    if payload.get("errno") != 0:
        logger.warning(
            "CLS telegraph API returned errno=%s msg=%s",
            payload.get("errno"),
            payload.get("msg"),
        )
        return []

    return payload.get("data", {}).get("roll_data", []) or []


async def fetch_cls_telegraph(
    session: aiohttp.ClientSession,
    max_pages: int = 20,
    since: datetime | None = None,
) -> list[NewsItem]:
    """Fetch CLS telegraphs newer than or equal to *since*.

    The stable pagination strategy is to walk the official ``last_time`` cursor
    until the page reaches the requested threshold.
    """
    threshold = _normalize_since(since)
    cursor = int(datetime.now(_TZ).timestamp())
    by_id: dict[int, NewsItem] = {}

    for _ in range(max(max_pages, 1)):
        page = await _fetch_cls_page(session, last_time=cursor)
        if not page:
            break

        oldest_ctime: int | None = None
        reached_threshold = False

        for article in page:
            ctime = int(article.get("ctime") or 0)
            if ctime > 0 and (oldest_ctime is None or ctime < oldest_ctime):
                oldest_ctime = ctime

            item = _build_cls_news_item(article)
            article_id = int(article.get("id") or 0)
            if item is None or article_id <= 0:
                continue
            if _is_before_threshold(item.published_at, threshold):
                reached_threshold = True
                continue
            by_id.setdefault(article_id, item)

        if oldest_ctime is None:
            break
        if reached_threshold:
            break
        if cursor and oldest_ctime >= cursor:
            logger.warning("CLS telegraph cursor stalled at %s", cursor)
            break
        cursor = oldest_ctime

    return sorted(
        by_id.values(),
        key=lambda item: (item.published_at or datetime.min.replace(tzinfo=timezone.utc), item.url),
        reverse=True,
    )


async def fetch_eastmoney_kuaixun(
    session: aiohttp.ClientSession,
    max_pages: int = 5,
    since: datetime | None = None,
) -> list[NewsItem]:
    """Eastmoney 7x24 news via simple page-number pagination."""
    threshold = _normalize_since(since)
    pagesize = 50
    all_items: list[NewsItem] = []
    headers = {"User-Agent": _UA, "Referer": "https://kuaixun.eastmoney.com/"}

    for page in range(1, max_pages + 1):
        url = (
            f"https://newsapi.eastmoney.com/kuaixun/v1/"
            f"getlist_102_ajaxResult_{pagesize}_{page}_.html"
        )
        try:
            async with session.get(url, headers=headers, timeout=_TIMEOUT) as resp:
                text = await resp.text()
        except Exception as exc:
            logger.warning("Eastmoney kuaixun page %d fetch failed: %s", page, exc)
            break

        match = re.search(r"var ajaxResult=(\{.*?\});?\s*$", text, re.DOTALL)
        if not match:
            break
        try:
            result = json.loads(match.group(1))
        except Exception:
            break

        live_list = result.get("LivesList", [])
        if not live_list:
            break

        all_before_threshold = True
        for entry in live_list:
            showtime = entry.get("showtime", "")
            try:
                dt: datetime | None = datetime.strptime(showtime, "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=_TZ
                )
            except Exception:
                dt = None

            if _is_before_threshold(dt, threshold):
                continue

            all_before_threshold = False
            all_items.append(
                NewsItem(
                    title=entry.get("title", ""),
                    summary=entry.get("digest", ""),
                    url=f"https://kuaixun.eastmoney.com/a/{entry.get('newsid', '')}",
                    source="eastmoney",
                    published_at=dt,
                )
            )

        if all_before_threshold:
            break

    return all_items


async def fetch_ths_kuaixun(
    session: aiohttp.ClientSession,
    max_pages: int = 5,
    since: datetime | None = None,
) -> list[NewsItem]:
    """THS 7x24 news via simple page-number pagination."""
    threshold = _normalize_since(since)
    pagesize = 50
    all_items: list[NewsItem] = []
    headers = {"User-Agent": _UA, "Referer": "https://news.10jqka.com.cn/"}

    for page in range(1, max_pages + 1):
        url = (
            f"https://news.10jqka.com.cn/tapp/news/push/stock/"
            f"?page={page}&tag=&track=website&pagesize={pagesize}"
        )
        try:
            async with session.get(url, headers=headers, timeout=_TIMEOUT) as resp:
                data = await resp.json(content_type=None)
        except Exception as exc:
            logger.warning("THS kuaixun page %d fetch failed: %s", page, exc)
            break

        entries = data.get("data", {}).get("list", [])
        if not entries:
            break

        all_before_threshold = True
        for entry in entries:
            dt = _from_epoch_to_local(entry.get("ctime", 0))
            if _is_before_threshold(dt, threshold):
                continue

            all_before_threshold = False
            all_items.append(
                NewsItem(
                    title=entry.get("title", ""),
                    summary=entry.get("digest", entry.get("remark", "")),
                    url=f"https://news.10jqka.com.cn/{entry.get('seq', '')}",
                    source="ths",
                    published_at=dt,
                )
            )

        if all_before_threshold:
            break

    return all_items


async def fetch_all_sources(
    max_pages: int = 20,
    since: datetime | None = None,
) -> list[NewsItem]:
    """Fetch from CLS, Eastmoney, and THS concurrently and merge results."""
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            fetch_cls_telegraph(session, max_pages, since=since),
            fetch_eastmoney_kuaixun(session, max_pages=max_pages, since=since),
            fetch_ths_kuaixun(session, max_pages=max_pages, since=since),
            return_exceptions=True,
        )
    items: list[NewsItem] = []
    for result in results:
        if isinstance(result, list):
            items.extend(result)
    return items
