from __future__ import annotations

import io
from datetime import datetime
from typing import Literal

import pandas as pd
from pydantic import BaseModel, field_validator


class TradeRecord(BaseModel):
    """A single buy/sell transaction extracted from conversation."""

    ts_code: str
    direction: Literal["buy", "sell"]
    quantity: float
    price: float
    fee: float = 0.0
    trade_date: str = "unknown"  # YYYYMMDD or "unknown"
    name: str | None = None

    @field_validator("ts_code")
    @classmethod
    def normalise_code(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("trade_date")
    @classmethod
    def normalise_date(cls, v: str) -> str:
        v = v.strip().replace("-", "")
        if v == "" or v.lower() == "unknown":
            return "unknown"
        return v


class PortfolioMemory:
    """
    Session-scoped in-memory store for trade records.

    Designed to be stored in Streamlit session_state and passed to tools via
    a module-level accessor (set_active / get_active).
    """

    def __init__(self) -> None:
        self._trades: list[TradeRecord] = []

    def add(self, record: TradeRecord) -> None:
        self._trades.append(record)

    def all_trades(self) -> list[TradeRecord]:
        return list(self._trades)

    def clear(self) -> None:
        self._trades.clear()

    def is_empty(self) -> bool:
        return len(self._trades) == 0

    def to_csv(self) -> str:
        """Convert stored trades to CSV string suitable for pnl_calculator."""
        if not self._trades:
            return ""
        rows = []
        for t in self._trades:
            date = t.trade_date if t.trade_date != "unknown" else datetime.now().strftime("%Y%m%d")
            rows.append(
                {
                    "ts_code": t.ts_code,
                    "trade_date": date,
                    "direction": t.direction,
                    "quantity": t.quantity,
                    "price": t.price,
                    "fee": t.fee,
                }
            )
        buf = io.StringIO()
        pd.DataFrame(rows).to_csv(buf, index=False)
        return buf.getvalue()

    def to_dataframe(self) -> pd.DataFrame:
        """Human-readable DataFrame for Streamlit sidebar display."""
        if not self._trades:
            return pd.DataFrame(
                columns=["代码", "名称", "方向", "数量", "价格", "手续费", "日期"]
            )
        rows = []
        for t in self._trades:
            rows.append(
                {
                    "代码": t.ts_code,
                    "名称": t.name or "",
                    "方向": "买入" if t.direction == "buy" else "卖出",
                    "数量": t.quantity,
                    "价格": t.price,
                    "手续费": t.fee,
                    "日期": t.trade_date,
                }
            )
        return pd.DataFrame(rows)

    def to_trades_df(self) -> pd.DataFrame:
        """Raw trades as DataFrame for pnl_calculator."""
        if not self._trades:
            return pd.DataFrame(
                columns=["ts_code", "trade_date", "direction", "quantity", "price", "fee"]
            )
        rows = []
        for t in self._trades:
            date = t.trade_date if t.trade_date != "unknown" else datetime.now().strftime("%Y%m%d")
            rows.append(
                {
                    "ts_code": t.ts_code,
                    "trade_date": date,
                    "direction": t.direction,
                    "quantity": t.quantity,
                    "price": t.price,
                    "fee": t.fee,
                }
            )
        return pd.DataFrame(rows).sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self._trades)


# ---------------------------------------------------------------------------
# Module-level active instance accessor (used by tools without import cycles)
# ---------------------------------------------------------------------------
_active: PortfolioMemory | None = None


def set_active(memory: PortfolioMemory) -> None:
    global _active
    _active = memory


def get_active() -> PortfolioMemory:
    global _active
    if _active is None:
        _active = PortfolioMemory()
    return _active
