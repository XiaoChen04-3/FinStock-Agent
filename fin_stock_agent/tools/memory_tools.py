from __future__ import annotations

import json
from typing import Annotated, Literal

from langchain_core.tools import tool

from fin_stock_agent.memory.portfolio_memory import TradeRecord, get_active


@tool
def add_trade_record(
    ts_code: Annotated[str, "Tushare 格式股票代码，如 600519.SH；若只知名称请先用 search_stock 查询代码"],
    direction: Annotated[Literal["buy", "sell"], "buy=买入，sell=卖出"],
    quantity: Annotated[float, "成交数量（股）"],
    price: Annotated[float, "成交价格（元/股）"],
    fee: Annotated[float, "手续费（元），不知道则填 0"] = 0.0,
    trade_date: Annotated[str, "成交日期 YYYYMMDD，不知道则填 unknown"] = "unknown",
    name: Annotated[str, "股票名称（可选）"] = "",
) -> str:
    """将一笔买入或卖出交易存入持仓记忆，供后续盈亏计算使用。"""
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
        return json.dumps(
            {
                "ok": True,
                "message": f"已记录：{direction} {ts_code} {quantity}股 @{price}元",
                "record": record.model_dump(),
            },
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


@tool
def get_portfolio_positions() -> str:
    """查看当前持仓记忆中的所有交易记录（原始流水，非盈亏汇总）。"""
    mem = get_active()
    trades = mem.all_trades()
    if not trades:
        return json.dumps(
            {"ok": True, "count": 0, "trades": [], "note": "持仓记忆为空"},
            ensure_ascii=False,
        )
    return json.dumps(
        {
            "ok": True,
            "count": len(trades),
            "trades": [t.model_dump() for t in trades],
        },
        ensure_ascii=False,
        default=str,
    )


@tool
def clear_portfolio_memory() -> str:
    """清空持仓记忆中的所有交易记录（不可恢复）。"""
    mem = get_active()
    count = len(mem)
    mem.clear()
    return json.dumps(
        {"ok": True, "message": f"已清空持仓记忆，共删除 {count} 条记录。"},
        ensure_ascii=False,
    )


def get_memory_tools():
    return [add_trade_record, get_portfolio_positions, clear_portfolio_memory]
