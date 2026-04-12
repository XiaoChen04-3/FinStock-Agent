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


def _latest_fund_nav(c, code: str, end: str, start: str) -> tuple[float | None, str | None]:
    """Latest unit/adj NAV for an open-end fund (.OF) via ``fund_nav``."""
    try:
        df = c.call("fund_nav", ts_code=code, start_date=start, end_date=end)
    except TushareRequestError:
        return None, None
    if df is None or df.empty:
        return None, None
    sort_col = "nav_date" if "nav_date" in df.columns else "ann_date"
    if sort_col not in df.columns:
        sort_col = df.columns[0]
    row = df.sort_values(sort_col).iloc[-1]
    nav = row.get("adj_nav")
    if nav is None or (isinstance(nav, float) and nav != nav):
        nav = row.get("unit_nav")
    if nav is None or (isinstance(nav, float) and nav != nav):
        nav = row.get("accum_nav")
    if nav is None:
        return None, None
    nav_date = str(row.get("nav_date") or row.get("ann_date") or row.get(sort_col) or "")
    return float(nav), nav_date


def fetch_last_closes(
    ts_codes: list[str], end_hint: str | None = None
) -> tuple[dict[str, float], str]:
    """Latest tradable price per symbol.

    - ``*.OF`` open-end funds: Tushare ``fund_nav`` (unit / adj NAV).
    - Stocks / ETFs: ``daily`` K-line ``close``; if empty, try ``fund_nav`` as fallback.
    """
    c = get_client()
    end = (end_hint or datetime.now().strftime("%Y%m%d")).replace("-", "")
    start = (datetime.strptime(end, "%Y%m%d") - pd.Timedelta(days=120)).strftime("%Y%m%d")
    out: dict[str, float] = {}
    last_dates: list[str] = []
    for raw in ts_codes:
        code = (raw or "").strip().upper()
        if not code:
            continue
        try:
            if code.endswith(".OF"):
                nav, nav_date = _latest_fund_nav(c, code, end, start)
                if nav is not None:
                    out[code] = nav
                    if nav_date:
                        last_dates.append(nav_date.replace("-", "")[:8])
                continue
            df = c.call("daily", ts_code=code, start_date=start, end_date=end, use_cache=True)
            if df is not None and not df.empty:
                row = df.sort_values("trade_date").iloc[-1]
                out[code] = float(row["close"])
                last_dates.append(str(row["trade_date"]))
                continue
            nav, nav_date = _latest_fund_nav(c, code, end, start)
            if nav is not None:
                out[code] = nav
                if nav_date:
                    last_dates.append(nav_date.replace("-", "")[:8])
        except TushareRequestError:
            continue
    as_of = max(last_dates) if last_dates else end
    return out, as_of
