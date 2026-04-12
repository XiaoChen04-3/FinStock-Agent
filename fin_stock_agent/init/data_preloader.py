from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import delete

from fin_stock_agent.storage.database import get_session
from fin_stock_agent.storage.models import FundLookupRecord, IndexLookupRecord, TradeCalendarRecord
from fin_stock_agent.utils.tushare_client import get_client


class DataPreloader:
    def preload(self) -> dict[str, int]:
        return {
            "trade_calendar": self.preload_trade_calendar(),
            "index_lookup": self.preload_index_lookup(),
            "fund_lookup": self.preload_fund_lookup(),
        }

    def preload_trade_calendar(self) -> int:
        client = get_client()
        today = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=90)).strftime("%Y%m%d")
        df = client.call(
            "trade_cal",
            exchange="SSE",
            start_date=start,
            end_date=today,
            fields="cal_date,is_open,exchange",
        )
        rows = [
            TradeCalendarRecord(
                cal_date=str(item.get("cal_date", "")).replace("-", ""),
                is_open=str(item.get("is_open", "0")) == "1",
                exchange=item.get("exchange") or "SSE",
            )
            for item in df.to_dict(orient="records")
        ]
        with get_session() as session:
            session.execute(delete(TradeCalendarRecord))
            session.add_all(rows)
        return len(rows)

    def preload_index_lookup(self) -> int:
        client = get_client()
        frames = [client.call("index_basic", market=market) for market in ("SSE", "SZSE")]
        df = frames[0]
        for part in frames[1:]:
            df = df._append(part, ignore_index=True)
        rows = [
            IndexLookupRecord(
                ts_code=item.get("ts_code", ""),
                name=item.get("name", ""),
                market=item.get("market"),
                category=item.get("category"),
            )
            for item in df.to_dict(orient="records")
        ]
        with get_session() as session:
            session.execute(delete(IndexLookupRecord))
            session.add_all(rows)
        return len(rows)

    def preload_fund_lookup(self) -> int:
        client = get_client()
        frames = [client.call("fund_basic", market=market) for market in ("E", "O")]
        df = frames[0]
        for part in frames[1:]:
            df = df._append(part, ignore_index=True)
        rows = [
            FundLookupRecord(
                ts_code=item.get("ts_code", ""),
                name=item.get("name", ""),
                fund_type=item.get("fund_type"),
                status=item.get("status"),
                market=item.get("market"),
            )
            for item in df.to_dict(orient="records")
        ]
        with get_session() as session:
            session.execute(delete(FundLookupRecord))
            session.add_all(rows)
        return len(rows)
