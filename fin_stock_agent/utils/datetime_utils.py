from __future__ import annotations

import json

from langchain_core.tools import tool

from fin_stock_agent.init.trade_calendar import TradingCalendar


@tool
def get_current_datetime() -> str:
    """Return the latest trading day and a small recent trading-day window."""
    calendar = TradingCalendar()
    recent = calendar.get_recent_trading_days(7)
    return json.dumps(
        {
            "today": recent[0] if recent else "",
            "latest_trading_day": calendar.get_latest_trading_day(),
            "recent_trading_days": recent,
        },
        ensure_ascii=False,
    )


def get_datetime_tools() -> list:
    return [get_current_datetime]
