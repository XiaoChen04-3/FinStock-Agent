from __future__ import annotations

import json
from typing import Annotated

from langchain_core.tools import tool

from fin_stock_agent.core.exceptions import TushareRequestError
from fin_stock_agent.init.trade_calendar import TradingCalendar
from fin_stock_agent.utils.tushare_client import get_client


def _payload(ok: bool, **kwargs) -> str:
    return json.dumps({"ok": ok, **kwargs}, ensure_ascii=False, default=str)


@tool
def get_index_price(
    ts_code: Annotated[str, "index code"],
    start_date: Annotated[str, "YYYYMMDD"] = "",
    end_date: Annotated[str, "YYYYMMDD"] = "",
) -> str:
    """Fetch historical daily index prices between the given dates."""
    try:
        calendar = TradingCalendar()
        end = (end_date or calendar.get_latest_trading_day()).replace("-", "")
        start = (start_date or end).replace("-", "")
        df = get_client().call("index_daily", ts_code=ts_code.strip().upper(), start_date=start, end_date=end)
        if df is None or df.empty:
            return _payload(False, error=f"No index data for {ts_code}")
        return _payload(True, rows=len(df), data=df.head(200).to_dict(orient="records"))
    except TushareRequestError as exc:
        return _payload(False, error=str(exc))


@tool
def get_index_today(ts_code: Annotated[str, "index code"]) -> str:
    """Return the most recent available daily snapshot for an index."""
    try:
        calendar = TradingCalendar()
        for day in calendar.get_recent_trading_days(5):
            df = get_client().call("index_daily", ts_code=ts_code.strip().upper(), start_date=day, end_date=day)
            if df is not None and not df.empty:
                return _payload(True, data=df.head(1).to_dict(orient="records")[0])
        return _payload(False, error=f"No recent index data for {ts_code}")
    except TushareRequestError as exc:
        return _payload(False, error=str(exc))


@tool
def get_global_overview() -> str:
    """Return a compact overview of several major global market indices."""
    try:
        codes = ["HSI", "SPX", "IXIC", "DJI", "N225"]
        items = []
        client = get_client()
        for code in codes:
            df = client.call("index_global", ts_code=code)
            if df is None or df.empty:
                continue
            row = df.sort_values("trade_date").iloc[-1].to_dict()
            items.append(row)
        if not items:
            return _payload(False, error="No global overview data available")
        return _payload(True, data=items)
    except TushareRequestError as exc:
        return _payload(False, error=str(exc))


def get_index_tools() -> list:
    return [get_index_price, get_index_today, get_global_overview]
