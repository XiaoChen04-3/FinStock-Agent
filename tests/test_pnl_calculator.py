"""Unit tests for the pure PnL calculator (no network calls)."""
import pandas as pd
import pytest

from fin_stock_agent.utils.pnl_calculator import compute_pnl_from_trades


def _df(rows):
    return pd.DataFrame(rows)


def test_weighted_average_basic():
    df = _df([
        {"ts_code": "600519.SH", "trade_date": "20240102", "direction": "buy",  "quantity": 100, "price": 1000, "fee": 0},
        {"ts_code": "600519.SH", "trade_date": "20240103", "direction": "buy",  "quantity": 100, "price": 1100, "fee": 0},
        {"ts_code": "600519.SH", "trade_date": "20240104", "direction": "sell", "quantity": 100, "price": 1200, "fee": 0},
    ])
    last = {"600519.SH": 1300.0}
    out = compute_pnl_from_trades(df, last)
    pos = {p["ts_code"]: p for p in out["positions"]}

    assert pos["600519.SH"]["quantity"] == 100
    assert abs(pos["600519.SH"]["avg_cost"] - 1050.0) < 1e-6
    # realized = (1200 - 1050) * 100 = 15000
    assert abs(out["realized_pnl_total"] - 15000.0) < 1e-6
    # floating = (1300 - 1050) * 100 = 25000
    assert abs(pos["600519.SH"]["floating_pnl"] - 25000.0) < 1e-6


def test_full_sell_no_floating():
    df = _df([
        {"ts_code": "000001.SZ", "trade_date": "20240101", "direction": "buy",  "quantity": 10, "price": 10, "fee": 0},
        {"ts_code": "000001.SZ", "trade_date": "20240102", "direction": "sell", "quantity": 10, "price": 12, "fee": 0},
    ])
    out = compute_pnl_from_trades(df, {"000001.SZ": 15.0})
    assert out["floating_pnl_total"] == 0.0
    assert abs(out["realized_pnl_total"] - 20.0) < 1e-6


def test_fee_reduces_pnl():
    df = _df([
        {"ts_code": "000001.SZ", "trade_date": "20240101", "direction": "buy",  "quantity": 10, "price": 10, "fee": 5},
        {"ts_code": "000001.SZ", "trade_date": "20240102", "direction": "sell", "quantity": 10, "price": 12, "fee": 5},
    ])
    out = compute_pnl_from_trades(df, {})
    # realized = (12-10.5)*10 - 5 = 10
    # avg_cost = (100+5)/10 = 10.5
    # realized = (12-10.5)*10 - 5*(10/10) = 15 - 5 = 10
    assert abs(out["realized_pnl_total"] - 10.0) < 1e-6
