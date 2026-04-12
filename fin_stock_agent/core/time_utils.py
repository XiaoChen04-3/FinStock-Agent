from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from fin_stock_agent.core.settings import settings


def app_tz() -> ZoneInfo:
    return ZoneInfo(settings.app_timezone)


def now_local() -> datetime:
    return datetime.now(app_tz())


def local_now_iso() -> str:
    return now_local().isoformat()


def today_local_str() -> str:
    return now_local().strftime("%Y-%m-%d")
