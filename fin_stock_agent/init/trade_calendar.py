from __future__ import annotations

import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import select

from fin_stock_agent.core.settings import settings
from fin_stock_agent.storage.cache import get_cache
from fin_stock_agent.storage.database import get_session
from fin_stock_agent.storage.models import TradeCalendarRecord
class TradingCalendar:
    def __init__(self) -> None:
        self.cache = get_cache()
        self.tz = ZoneInfo(settings.app_timezone)

    def get_recent_trading_days(self, n: int = 3) -> list[str]:
        days = self._load_recent_days_from_db(max(n, 10))
        if len(days) < n:
            self.refresh()
            days = self._load_recent_days_from_db(max(n, 10))
        days = days[:n]
        key = f"trade_cal:recent{n}"
        self.cache.setex(key, 3600, json.dumps(days, ensure_ascii=False))
        if days:
            self.cache.setex("trade_cal:latest", 3600, days[0])
        return days

    def get_latest_trading_day(self) -> str:
        days = self.get_recent_trading_days(1)
        return days[0] if days else datetime.now(self.tz).strftime("%Y%m%d")

    def is_trading_day(self, date_str: str) -> bool:
        normalized = (date_str or "").replace("-", "")
        with get_session() as session:
            record = session.get(TradeCalendarRecord, normalized)
            if record is not None:
                return bool(record.is_open)
        self.refresh()
        with get_session() as session:
            record = session.get(TradeCalendarRecord, normalized)
            return bool(record.is_open) if record is not None else False

    def refresh(self) -> None:
        from fin_stock_agent.utils.tushare_client import get_client

        client = get_client()
        today = datetime.now(self.tz).strftime("%Y%m%d")
        start = (datetime.now(self.tz) - timedelta(days=120)).strftime("%Y%m%d")
        df = client.call(
            "trade_cal",
            exchange="SSE",
            start_date=start,
            end_date=today,
            fields="cal_date,is_open,exchange",
        )
        with get_session() as session:
            for item in df.to_dict(orient="records"):
                cal_date = str(item.get("cal_date", "")).replace("-", "")
                existing = session.get(TradeCalendarRecord, cal_date)
                if existing is None:
                    existing = TradeCalendarRecord(cal_date=cal_date)
                    session.add(existing)
                existing.is_open = str(item.get("is_open", "0")) == "1"
                existing.exchange = item.get("exchange") or "SSE"
        self.get_recent_trading_days(3)

    def _load_recent_days_from_db(self, n: int) -> list[str]:
        with get_session() as session:
            rows = session.execute(
                select(TradeCalendarRecord.cal_date)
                .where(TradeCalendarRecord.is_open.is_(True))
                .order_by(TradeCalendarRecord.cal_date.desc())
                .limit(n)
            ).scalars()
            return list(rows)
