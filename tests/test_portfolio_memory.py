"""Unit tests for PortfolioMemory."""
import pytest

from fin_stock_agent.memory.portfolio_memory import PortfolioMemory, TradeRecord


def _mem_with_trades():
    m = PortfolioMemory()
    m.add(TradeRecord(ts_code="600519.SH", direction="buy", quantity=100, price=1688.0, fee=5.0, trade_date="20240102", name="贵州茅台"))
    m.add(TradeRecord(ts_code="600519.SH", direction="sell", quantity=30, price=1750.0, fee=5.0, trade_date="20240201"))
    return m


def test_add_and_all_trades():
    m = _mem_with_trades()
    trades = m.all_trades()
    assert len(trades) == 2
    assert trades[0].ts_code == "600519.SH"
    assert trades[0].direction == "buy"


def test_clear():
    m = _mem_with_trades()
    assert len(m) == 2
    m.clear()
    assert m.is_empty()
    assert len(m) == 0


def test_to_csv():
    m = _mem_with_trades()
    csv = m.to_csv()
    assert "600519.SH" in csv
    assert "buy" in csv
    assert "sell" in csv


def test_to_trades_df_sorted():
    m = _mem_with_trades()
    df = m.to_trades_df()
    assert list(df.columns) == ["ts_code", "trade_date", "direction", "quantity", "price", "fee"]
    assert df.iloc[0]["trade_date"] <= df.iloc[1]["trade_date"]


def test_to_dataframe_display():
    m = _mem_with_trades()
    df = m.to_dataframe()
    assert "代码" in df.columns
    assert "方向" in df.columns
    assert len(df) == 2


def test_normalise_code():
    r = TradeRecord(ts_code="600519.sh", direction="buy", quantity=10, price=100)
    assert r.ts_code == "600519.SH"


def test_normalise_date_unknown():
    r = TradeRecord(ts_code="000001.SZ", direction="buy", quantity=10, price=10, trade_date="")
    assert r.trade_date == "unknown"


def test_empty_memory():
    m = PortfolioMemory()
    assert m.is_empty()
    assert m.to_csv() == ""
    assert m.to_dataframe().empty
