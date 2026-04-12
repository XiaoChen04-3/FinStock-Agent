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
def get_stock_info(keyword_or_code: Annotated[str, "stock code or keyword"]) -> str:
    """Search one A-share stock and enrich it with a recent daily_basic snapshot."""
    try:
        client = get_client()
        keyword = (keyword_or_code or "").strip()
        df = client.call(
            "stock_basic",
            list_status="L",
            fields="ts_code,symbol,name,area,industry,market,exchange,list_date",
        )
        if "." in keyword.upper():
            filtered = df[df["ts_code"].str.upper() == keyword.upper()]
        else:
            filtered = df[
                df["name"].str.contains(keyword, case=False, na=False)
                | df["ts_code"].str.contains(keyword, case=False, na=False)
            ]
        if filtered.empty:
            return _payload(False, error=f"No stock matched {keyword}")
        item = filtered.head(1).to_dict(orient="records")[0]
        latest_day = TradingCalendar().get_latest_trading_day()
        daily_basic = client.call("daily_basic", ts_code=item["ts_code"], trade_date=latest_day)
        if daily_basic is not None and not daily_basic.empty:
            basic = daily_basic.head(1).to_dict(orient="records")[0]
            item.update(
                {
                    "pe": basic.get("pe"),
                    "pb": basic.get("pb"),
                    "total_mv": basic.get("total_mv"),
                }
            )
        return _payload(True, data=item)
    except TushareRequestError as exc:
        return _payload(False, error=str(exc))


@tool
def get_stock_price(
    ts_code: Annotated[str, "tushare stock code"],
    start_date: Annotated[str, "YYYYMMDD"] = "",
    end_date: Annotated[str, "YYYYMMDD"] = "",
) -> str:
    """Fetch historical daily bars for a stock within the requested date range."""
    try:
        client = get_client()
        calendar = TradingCalendar()
        end = (end_date or calendar.get_latest_trading_day()).replace("-", "")
        start = (start_date or end).replace("-", "")
        df = client.call("daily", ts_code=ts_code.strip().upper(), start_date=start, end_date=end)
        if df is None or df.empty:
            return _payload(False, error=f"No daily data for {ts_code}")
        return _payload(True, rows=len(df), data=df.head(200).to_dict(orient="records"))
    except TushareRequestError as exc:
        return _payload(False, error=str(exc))


@tool
def get_realtime_quote(ts_code: Annotated[str, "tushare stock code"]) -> str:
    """Return the latest available recent daily quote as a lightweight realtime proxy."""
    try:
        client = get_client()
        calendar = TradingCalendar()
        latest_day = calendar.get_latest_trading_day()
        df = client.call("daily", ts_code=ts_code.strip().upper(), start_date=latest_day, end_date=latest_day)
        if df is None or df.empty:
            df = client.call(
                "daily",
                ts_code=ts_code.strip().upper(),
                start_date=calendar.get_recent_trading_days(3)[-1],
                end_date=latest_day,
            )
        if df is None or df.empty:
            return _payload(False, error=f"No recent quote data for {ts_code}")
        row = df.sort_values("trade_date").iloc[-1].to_dict()
        return _payload(True, data=row)
    except TushareRequestError as exc:
        return _payload(False, error=str(exc))


@tool
def get_stock_fundamentals(ts_code: Annotated[str, "tushare stock code"]) -> str:
    """Return recent stock valuation and daily_basic fundamentals for one symbol."""
    try:
        client = get_client()
        latest_day = TradingCalendar().get_latest_trading_day()
        df = client.call("daily_basic", ts_code=ts_code.strip().upper(), trade_date=latest_day)
        if df is None or df.empty:
            return _payload(False, error=f"No fundamentals for {ts_code} on {latest_day}")
        return _payload(True, data=df.head(1).to_dict(orient="records")[0])
    except TushareRequestError as exc:
        return _payload(False, error=str(exc))


def get_market_tools() -> list:
    return [get_stock_info, get_stock_price, get_realtime_quote, get_stock_fundamentals]
