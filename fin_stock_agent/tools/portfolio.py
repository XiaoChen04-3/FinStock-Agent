from __future__ import annotations

import io
import json
from datetime import datetime
from typing import Annotated

import pandas as pd
from langchain_core.tools import tool

from finstock_agent.errors import TushareRequestError
from finstock_agent.tushare_client import get_client


def _normalize_ts_code(code: str) -> str:
    c = (code or "").strip().upper()
    return c


def _parse_trades_csv(csv_content: str) -> pd.DataFrame:
    raw = (csv_content or "").strip()
    if not raw:
        raise ValueError("CSV 内容为空")
    df = pd.read_csv(io.StringIO(raw))
    cols = {c.strip().lower(): c for c in df.columns}
    required = ["ts_code", "trade_date", "direction", "quantity", "price"]
    for r in required:
        if r not in cols:
            raise ValueError(f"缺少列: {r}，需要列: {', '.join(required)}，可选 fee")
    rename = {cols[r]: r for r in required}
    if "fee" in cols:
        rename[cols["fee"]] = "fee"
    df = df.rename(columns=rename)
    df["ts_code"] = df["ts_code"].map(_normalize_ts_code)
    df["direction"] = df["direction"].astype(str).str.lower().str.strip()
    df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "", regex=False)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    if "fee" not in df.columns:
        df["fee"] = 0.0
    else:
        df["fee"] = pd.to_numeric(df["fee"], errors="coerce").fillna(0.0)
    bad = df["direction"].isin(["buy", "sell"]) & df["quantity"].notna() & df["price"].notna()
    df = df.loc[bad].copy()
    if df.empty:
        raise ValueError("没有有效成交行（direction 需为 buy/sell）")
    df = df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    return df


def compute_pnl_from_trades(
    df: pd.DataFrame,
    last_closes: dict[str, float],
) -> dict:
    """加权平均成本；卖出按当前均价结转实现盈亏。"""
    by_symbol: dict[str, dict] = {}

    for _, row in df.iterrows():
        code = row["ts_code"]
        side = row["direction"]
        qty = float(row["quantity"])
        price = float(row["price"])
        fee = float(row["fee"])
        if qty <= 0:
            continue
        st = by_symbol.setdefault(
            code,
            {
                "qty": 0.0,
                "cost_basis": 0.0,
                "realized": 0.0,
            },
        )
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
        else:
            continue

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


def _fetch_last_closes(ts_codes: list[str], end_hint: str | None) -> tuple[dict[str, float], str]:
    c = get_client()
    end = (end_hint or datetime.now().strftime("%Y%m%d")).replace("-", "")
    start = (datetime.strptime(end, "%Y%m%d") - pd.Timedelta(days=120)).strftime("%Y%m%d")
    out: dict[str, float] = {}
    last_dates: list[str] = []
    for code in ts_codes:
        df = c.call("daily", ts_code=code, start_date=start, end_date=end, use_cache=True)
        if df is None or df.empty:
            continue
        d = df.sort_values("trade_date").iloc[-1]
        out[code] = float(d["close"])
        last_dates.append(str(d["trade_date"]))
    as_of = max(last_dates) if last_dates else end
    return out, as_of


@tool
def calculate_portfolio_pnl(
    csv_content: Annotated[str, "持仓流水 CSV 文本，列: ts_code,trade_date,direction,quantity,price 可选 fee"],
    as_of_date: Annotated[str, "用于取收盘价的日期 YYYYMMDD，可留空则取区间内最近收盘"] = "",
) -> str:
    """根据买卖流水计算加权成本、实现盈亏，并对当前持仓用最近日线收盘价估算浮动盈亏。"""
    try:
        df = _parse_trades_csv(csv_content)
        codes = sorted(df["ts_code"].unique().tolist())
        hint = as_of_date.strip().replace("-", "") or None
        last_closes, as_of = _fetch_last_closes(codes, hint)
        summary = compute_pnl_from_trades(df, last_closes)
        summary["as_of_trade_date"] = as_of
        summary["ok"] = True
        return json.dumps(summary, ensure_ascii=False, default=str)
    except ValueError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


def get_portfolio_tools():
    return [calculate_portfolio_pnl]


PORTFOLIO_CSV_TEMPLATE = """ts_code,trade_date,direction,quantity,price,fee
600519.SH,20240102,buy,100,1688.0,5.0
600519.SH,20240201,sell,30,1750.0,5.0
"""
