"""Pure PnL calculation functions – no LangChain, no Streamlit dependencies."""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from fin_stock_agent.core.exceptions import TushareRequestError
from fin_stock_agent.utils.tushare_client import get_client


def compute_pnl_from_trades(
    df: pd.DataFrame,
    last_closes: dict[str, float],
) -> dict:
    """Weighted-average cost method; realised PnL settled at average cost on sell."""
    by_symbol: dict[str, dict] = {}

    for _, row in df.iterrows():
        code = row["ts_code"]
        side = str(row["direction"]).lower().strip()
        qty = float(row["quantity"])
        price = float(row["price"])
        fee = float(row.get("fee", 0) or 0)
        if qty <= 0:
            continue
        st = by_symbol.setdefault(code, {"qty": 0.0, "cost_basis": 0.0, "realized": 0.0})
        avg = st["cost_basis"] / st["qty"] if st["qty"] > 0 else 0.0
        if side == "buy":
            st["cost_basis"] += qty * price + fee
            st["qty"] += qty
        elif side == "sell":
            sell_qty = min(qty, st["qty"])
            if sell_qty <= 0:
                continue
            st["realized"] += (price - avg) * sell_qty - fee * (sell_qty / qty)
            st["cost_basis"] -= avg * sell_qty
            st["qty"] -= sell_qty

    realized_total = sum(s["realized"] for s in by_symbol.values())
    positions: list[dict] = []
    floating_total = 0.0

    for code, st in by_symbol.items():
        qty = st["qty"]
        cb = st["cost_basis"]
        avg = cb / qty if qty > 0 else 0.0
        last = last_closes.get(code)
        mkt = (last * qty) if (last is not None and qty > 0) else None
        fl = (mkt - cb) if mkt is not None else None
        if fl is not None:
            floating_total += fl
        positions.append(
            {
                "ts_code": code,
                "quantity": round(qty, 6),
                "avg_cost": round(avg, 6) if qty > 0 else 0.0,
                "cost_basis": round(cb, 4),
                "last_close": last,
                "market_value": round(mkt, 4) if mkt is not None else None,
                "floating_pnl": round(fl, 4) if fl is not None else None,
                "realized_pnl": round(st["realized"], 4),
            }
        )

    return {
        "method": "加权平均成本，卖出按当时均价计算实现盈亏",
        "realized_pnl_total": round(realized_total, 4),
        "floating_pnl_total": round(floating_total, 4),
        "positions": positions,
    }


def fetch_last_closes(
    ts_codes: list[str], end_hint: str | None = None
) -> tuple[dict[str, float], str]:
    """Fetch the most-recent closing price for each ts_code via daily bars."""
    c = get_client()
    end = (end_hint or datetime.now().strftime("%Y%m%d")).replace("-", "")
    start = (datetime.strptime(end, "%Y%m%d") - pd.Timedelta(days=120)).strftime("%Y%m%d")
    out: dict[str, float] = {}
    last_dates: list[str] = []
    for code in ts_codes:
        try:
            df = c.call("daily", ts_code=code, start_date=start, end_date=end, use_cache=True)
        except TushareRequestError:
            continue
        if df is None or df.empty:
            continue
        row = df.sort_values("trade_date").iloc[-1]
        out[code] = float(row["close"])
        last_dates.append(str(row["trade_date"]))
    as_of = max(last_dates) if last_dates else end
    return out, as_of
