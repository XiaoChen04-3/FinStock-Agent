from __future__ import annotations

import io
from datetime import datetime
from typing import Literal

import pandas as pd
from pydantic import BaseModel, field_validator


class TradeRecord(BaseModel):
    ts_code: str
    direction: Literal["buy", "sell"]
    quantity: float
    price: float
    fee: float = 0.0
    trade_date: str = "unknown"
    name: str | None = None

    @field_validator("ts_code")
    @classmethod
    def normalize_code(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("trade_date")
    @classmethod
    def normalize_date(cls, value: str) -> str:
        v = value.strip().replace("-", "")
        if not v or v.lower() == "unknown":
            return "unknown"
        return v


class PortfolioMemory:
    def __init__(self) -> None:
        self._trades: list[TradeRecord] = []

    @classmethod
    def from_trades(cls, trades: list[TradeRecord]) -> "PortfolioMemory":
        memory = cls()
        memory._trades = list(trades)
        return memory

    def add(self, record: TradeRecord) -> None:
        self._trades.append(record)

    def all_trades(self) -> list[TradeRecord]:
        return list(self._trades)

    def clear(self) -> None:
        self._trades.clear()

    def is_empty(self) -> bool:
        return len(self._trades) == 0

    def to_csv(self) -> str:
        if not self._trades:
            return ""
        rows = []
        for trade in self._trades:
            rows.append(
                {
                    "ts_code": trade.ts_code,
                    "trade_date": trade.trade_date if trade.trade_date != "unknown" else datetime.now().strftime("%Y%m%d"),
                    "direction": trade.direction,
                    "quantity": trade.quantity,
                    "price": trade.price,
                    "fee": trade.fee,
                }
            )
        buf = io.StringIO()
        pd.DataFrame(rows).to_csv(buf, index=False)
        return buf.getvalue()

    def to_dataframe(self) -> pd.DataFrame:
        if not self._trades:
            return pd.DataFrame(columns=["Code", "Name", "Direction", "Quantity", "Price", "Fee", "Date"])
        return pd.DataFrame(
            [
                {
                    "Code": trade.ts_code,
                    "Name": trade.name or "",
                    "Direction": trade.direction,
                    "Quantity": trade.quantity,
                    "Price": trade.price,
                    "Fee": trade.fee,
                    "Date": trade.trade_date,
                }
                for trade in self._trades
            ]
        )

    def to_trades_df(self) -> pd.DataFrame:
        if not self._trades:
            return pd.DataFrame(columns=["ts_code", "trade_date", "direction", "quantity", "price", "fee"])
        rows = []
        for trade in self._trades:
            rows.append(
                {
                    "ts_code": trade.ts_code,
                    "trade_date": trade.trade_date if trade.trade_date != "unknown" else datetime.now().strftime("%Y%m%d"),
                    "direction": trade.direction,
                    "quantity": trade.quantity,
                    "price": trade.price,
                    "fee": trade.fee,
                }
            )
        return pd.DataFrame(rows).sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self._trades)


_active: PortfolioMemory | None = None


def set_active(memory: PortfolioMemory) -> None:
    global _active
    _active = memory


def get_active() -> PortfolioMemory:
    global _active
    if _active is None:
        _active = PortfolioMemory()
    return _active
