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
def search_fund(keyword: Annotated[str, "fund keyword"]) -> str:
    """Search mutual funds and ETFs by fuzzy name match via Tushare fund_basic."""
    try:
        client = get_client()
        df = client.call("fund_basic", market="E")
        df2 = client.call("fund_basic", market="O")
        df = df._append(df2, ignore_index=True)
        filtered = df[df["name"].str.contains(keyword, case=False, na=False)]
        return _payload(True, rows=len(filtered), data=filtered.head(20).to_dict(orient="records"))
    except TushareRequestError as exc:
        return _payload(False, error=str(exc))


@tool
def get_fund_info(ts_code: Annotated[str, "fund code"]) -> str:
    """Return one fund's basic metadata such as name, type, market, and status."""
    try:
        client = get_client()
        market = "E" if ts_code.upper().endswith((".SH", ".SZ")) else "O"
        df = client.call("fund_basic", market=market)
        filtered = df[df["ts_code"].str.upper() == ts_code.upper()]
        if filtered.empty:
            return _payload(False, error=f"No fund matched {ts_code}")
        return _payload(True, data=filtered.head(1).to_dict(orient="records")[0])
    except TushareRequestError as exc:
        return _payload(False, error=str(exc))


@tool
def get_fund_nav(
    ts_code: Annotated[str, "fund code"],
    start_date: Annotated[str, "YYYYMMDD"] = "",
    end_date: Annotated[str, "YYYYMMDD"] = "",
) -> str:
    """Fetch historical fund NAV data between the given date range."""
    try:
        client = get_client()
        end = (end_date or TradingCalendar().get_latest_trading_day()).replace("-", "")
        start = (start_date or end).replace("-", "")
        df = client.call("fund_nav", ts_code=ts_code.strip().upper(), start_date=start, end_date=end)
        if df is None or df.empty:
            return _payload(False, error=f"No nav data for {ts_code}")
        return _payload(True, rows=len(df), data=df.head(300).to_dict(orient="records"))
    except TushareRequestError as exc:
        return _payload(False, error=str(exc))


@tool
def get_fund_nav_today(ts_code: Annotated[str, "fund code"]) -> str:
    """Fetch the latest available recent NAV snapshot for a fund."""
    try:
        calendar = TradingCalendar()
        days = calendar.get_recent_trading_days(5)
        client = get_client()
        for day in days:
            df = client.call("fund_nav", ts_code=ts_code.strip().upper(), end_date=day, start_date=day)
            if df is not None and not df.empty:
                return _payload(True, data=df.head(1).to_dict(orient="records")[0])
        return _payload(False, error=f"No recent nav data for {ts_code}")
    except TushareRequestError as exc:
        return _payload(False, error=str(exc))


def get_fund_tools() -> list:
    return [search_fund, get_fund_info, get_fund_nav, get_fund_nav_today]
