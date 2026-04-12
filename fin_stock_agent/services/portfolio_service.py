from __future__ import annotations

from collections import defaultdict

from sqlalchemy import select

from fin_stock_agent.init.trade_calendar import TradingCalendar
from fin_stock_agent.memory.portfolio_memory import PortfolioMemory, TradeRecord
from fin_stock_agent.storage.cache import get_cache
from fin_stock_agent.storage.database import get_session
from fin_stock_agent.storage.models import TradeRecordORM
from fin_stock_agent.utils.pnl_calculator import compute_pnl_from_trades, fetch_last_closes


class PortfolioService:
    def __init__(self) -> None:
        self.cache = get_cache()
        self.trade_calendar = TradingCalendar()

    def add_trade(self, user_id: str, trade: TradeRecord) -> None:
        with get_session() as session:
            session.add(
                TradeRecordORM(
                    user_id=user_id,
                    ts_code=trade.ts_code,
                    name=trade.name,
                    direction=trade.direction,
                    quantity=trade.quantity,
                    price=trade.price,
                    fee=trade.fee,
                    trade_date=self._normalize_trade_date(trade.trade_date),
                )
            )
        self.cache.delete(f"portfolio:{user_id}:context")

    def get_trade_history(self, user_id: str, limit: int = 100) -> list[TradeRecord]:
        with get_session() as session:
            rows = session.execute(
                select(TradeRecordORM)
                .where(TradeRecordORM.user_id == user_id, TradeRecordORM.is_deleted.is_(False))
                .order_by(TradeRecordORM.trade_date.desc(), TradeRecordORM.id.desc())
                .limit(limit)
            ).scalars()
            return [
                TradeRecord(
                    ts_code=row.ts_code,
                    direction=row.direction,
                    quantity=row.quantity,
                    price=row.price,
                    fee=row.fee,
                    trade_date=row.trade_date,
                    name=row.name,
                )
                for row in rows
            ]

    def delete_trade(self, user_id: str, trade_id: int) -> None:
        with get_session() as session:
            row = session.get(TradeRecordORM, trade_id)
            if row and row.user_id == user_id:
                row.is_deleted = True
        self.cache.delete(f"portfolio:{user_id}:context")

    def build_memory(self, user_id: str, limit: int = 500) -> PortfolioMemory:
        return PortfolioMemory.from_trades(self.get_trade_history(user_id, limit=limit))

    def get_holdings(self, user_id: str) -> list[dict]:
        trades = self.get_trade_history(user_id, limit=500)
        if not trades:
            return []
        bucket: dict[str, dict] = defaultdict(lambda: {"quantity": 0.0, "cost": 0.0, "name": ""})
        for trade in reversed(trades):
            item = bucket[trade.ts_code]
            item["name"] = trade.name or item["name"]
            if trade.direction == "buy":
                item["quantity"] += trade.quantity
                item["cost"] += trade.quantity * trade.price + trade.fee
            else:
                if item["quantity"] <= 0:
                    continue
                avg = item["cost"] / item["quantity"] if item["quantity"] else 0.0
                sell_qty = min(trade.quantity, item["quantity"])
                item["quantity"] -= sell_qty
                item["cost"] -= avg * sell_qty
        codes = [code for code, value in bucket.items() if value["quantity"] > 0]
        last_closes, as_of = fetch_last_closes(codes, self.trade_calendar.get_latest_trading_day())
        holdings: list[dict] = []
        for code in codes:
            item = bucket[code]
            qty = item["quantity"]
            avg = item["cost"] / qty if qty else 0.0
            last = last_closes.get(code)
            market_value = (last or 0.0) * qty if last is not None else None
            unrealized = market_value - item["cost"] if market_value is not None else None
            holdings.append(
                {
                    "ts_code": code,
                    "name": item["name"] or code,
                    "quantity": round(qty, 4),
                    "avg_cost": round(avg, 4),
                    "last_price": last,
                    "market_value": round(market_value, 4) if market_value is not None else None,
                    "unrealized_pnl": round(unrealized, 4) if unrealized is not None else None,
                    "as_of_trade_date": as_of,
                }
            )
        holdings.sort(key=lambda row: row["market_value"] or 0.0, reverse=True)
        return holdings

    def get_pnl_summary(self, user_id: str) -> dict:
        trades = self.get_trade_history(user_id, limit=500)
        if not trades:
            return {"ok": True, "positions": [], "realized_pnl_total": 0.0, "floating_pnl_total": 0.0}
        import pandas as pd

        df = pd.DataFrame(
            [
                {
                    "ts_code": trade.ts_code,
                    "trade_date": self._normalize_trade_date(trade.trade_date),
                    "direction": trade.direction,
                    "quantity": trade.quantity,
                    "price": trade.price,
                    "fee": trade.fee,
                }
                for trade in trades
            ]
        )
        codes = sorted(df["ts_code"].unique().tolist())
        last_closes, as_of = fetch_last_closes(codes, self.trade_calendar.get_latest_trading_day())
        summary = compute_pnl_from_trades(df, last_closes)
        summary["as_of_trade_date"] = as_of
        summary["ok"] = True
        return summary

    def build_portfolio_context(self, user_id: str) -> str:
        cache_key = f"portfolio:{user_id}:context"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        holdings = self.get_holdings(user_id)
        if not holdings:
            return "## Current portfolio\nNo persisted holdings yet."
        lines = [
            "## Current portfolio",
            "| Code | Name | Quantity | Avg Cost | Last Price | PnL |",
            "|---|---|---:|---:|---:|---:|",
        ]
        total_market_value = 0.0
        total_unrealized = 0.0
        for row in holdings:
            total_market_value += row["market_value"] or 0.0
            total_unrealized += row["unrealized_pnl"] or 0.0
            lines.append(
                f"| {row['ts_code']} | {row['name']} | {row['quantity']} | "
                f"{row['avg_cost']:.4f} | {row['last_price'] or '-'} | {row['unrealized_pnl'] or 0.0:.2f} |"
            )
        lines.append("")
        lines.append(f"Total market value: {total_market_value:.2f}")
        lines.append(f"Total unrealized pnl: {total_unrealized:.2f}")
        context = "\n".join(lines)
        self.cache.setex(cache_key, 600, context)
        return context

    def _normalize_trade_date(self, value: str) -> str:
        normalized = (value or "").replace("-", "")
        if not normalized or normalized == "unknown":
            return self.trade_calendar.get_latest_trading_day()
        return normalized
