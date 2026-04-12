from __future__ import annotations

import json
from typing import Annotated

import pandas as pd
from langchain_core.tools import tool

from fin_stock_agent.core.exceptions import TushareRequestError
from fin_stock_agent.init.trade_calendar import TradingCalendar
from fin_stock_agent.utils.tushare_client import get_client


def _payload(ok: bool, **kwargs) -> str:
    return json.dumps({"ok": ok, **kwargs}, ensure_ascii=False, default=str)


def _load_bars(ts_code: str) -> pd.DataFrame:
    calendar = TradingCalendar()
    end = calendar.get_latest_trading_day()
    start = calendar.get_recent_trading_days(10)[-1]
    df = get_client().call("daily", ts_code=ts_code.strip().upper(), start_date=start, end_date=end)
    return df.sort_values("trade_date").reset_index(drop=True)


@tool
def get_technical_indicators(ts_code: Annotated[str, "stock code"]) -> str:
    """Calculate lightweight local technical indicators from recent daily bars."""
    try:
        df = _load_bars(ts_code)
        if df is None or df.empty:
            return _payload(False, error=f"No bars for {ts_code}")
        close = df["close"].astype(float)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(6).mean()
        loss = (-delta.clip(upper=0)).rolling(6).mean()
        rs = gain / loss.replace(0, pd.NA)
        rsi = 100 - 100 / (1 + rs.fillna(0))
        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=9, adjust=False).mean()
        ma5 = close.rolling(5).mean()
        ma10 = close.rolling(10).mean()
        return _payload(
            True,
            data={
                "trade_date": str(df.iloc[-1]["trade_date"]),
                "close": float(close.iloc[-1]),
                "ma5": float(ma5.iloc[-1]) if not pd.isna(ma5.iloc[-1]) else None,
                "ma10": float(ma10.iloc[-1]) if not pd.isna(ma10.iloc[-1]) else None,
                "macd": float(macd.iloc[-1]),
                "signal": float(signal.iloc[-1]),
                "rsi6": float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None,
            },
        )
    except TushareRequestError as exc:
        return _payload(False, error=str(exc))


@tool
def get_trend_signal(ts_code: Annotated[str, "stock code"]) -> str:
    """Generate a simple bullish, bearish, or range trend signal from local indicators."""
    raw = json.loads(get_technical_indicators.invoke({"ts_code": ts_code}))
    if not raw.get("ok"):
        return json.dumps(raw, ensure_ascii=False)
    data = raw["data"]
    signal = "range"
    if data.get("ma5") and data.get("ma10"):
        if data["ma5"] > data["ma10"] and (data.get("macd") or 0) >= (data.get("signal") or 0):
            signal = "bullish"
        elif data["ma5"] < data["ma10"]:
            signal = "bearish"
    return _payload(True, data={"signal": signal, "snapshot": data})


def get_technical_tools() -> list:
    return [get_technical_indicators, get_trend_signal]
