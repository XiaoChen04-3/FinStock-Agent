from __future__ import annotations

import json
from typing import Annotated, Literal

from langchain_core.tools import tool

from fin_stock_agent.core.identity import local_profile_id
from fin_stock_agent.memory.portfolio_memory import TradeRecord, get_active
from fin_stock_agent.services.portfolio_service import PortfolioService

_ACTIVE_PROFILE_ID = local_profile_id()


def set_tool_user_id(user_id: str) -> None:
    global _ACTIVE_PROFILE_ID
    _ACTIVE_PROFILE_ID = user_id or local_profile_id()


def _payload(ok: bool, **kwargs) -> str:
    return json.dumps({"ok": ok, **kwargs}, ensure_ascii=False, default=str)


@tool
def add_trade(
    ts_code: Annotated[str, "tushare code"],
    direction: Annotated[Literal["buy", "sell"], "buy or sell"],
    quantity: Annotated[float, "quantity"],
    price: Annotated[float, "price"],
    fee: Annotated[float, "fee"] = 0.0,
    trade_date: Annotated[str, "YYYYMMDD"] = "unknown",
    name: Annotated[str, "optional name"] = "",
) -> str:
    """Persist a buy or sell trade and mirror it into the active in-session memory."""
    try:
        record = TradeRecord(
            ts_code=ts_code,
            direction=direction,
            quantity=quantity,
            price=price,
            fee=fee,
            trade_date=trade_date,
            name=name or None,
        )
        get_active().add(record)
        PortfolioService().add_trade(_ACTIVE_PROFILE_ID, record)
        return _payload(True, record=record.model_dump())
    except Exception as exc:
        return _payload(False, error=str(exc))


@tool
def get_portfolio() -> str:
    """Return the user's persisted current holdings with latest value estimates."""
    try:
        holdings = PortfolioService().get_holdings(_ACTIVE_PROFILE_ID)
        return _payload(True, count=len(holdings), holdings=holdings)
    except Exception as exc:
        return _payload(False, error=str(exc))


@tool
def get_pnl_summary() -> str:
    """Return a realized and unrealized PnL summary for the user's portfolio."""
    try:
        return json.dumps(PortfolioService().get_pnl_summary(_ACTIVE_PROFILE_ID), ensure_ascii=False, default=str)
    except Exception as exc:
        return _payload(False, error=str(exc))


def get_portfolio_tools() -> list:
    return [add_trade, get_portfolio, get_pnl_summary]
