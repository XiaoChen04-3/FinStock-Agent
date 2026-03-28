from __future__ import annotations

import json
from typing import Annotated

from langchain_core.tools import tool

from fin_stock_agent.core.exceptions import TushareRequestError
from fin_stock_agent.memory.portfolio_memory import get_active
from fin_stock_agent.utils.pnl_calculator import compute_pnl_from_trades, fetch_last_closes


@tool
def calculate_portfolio_pnl(
    as_of_date: Annotated[
        str,
        "用于取收盘价的日期 YYYYMMDD，可留空则自动取最近交易日收盘价",
    ] = "",
) -> str:
    """
    从持仓记忆中读取买卖流水，计算加权平均成本、实现盈亏以及浮动盈亏。
    无需上传 CSV；交易记录通过对话自动录入（add_trade_record）。
    """
    mem = get_active()
    if mem.is_empty():
        return json.dumps(
            {"ok": False, "error": "持仓记忆为空。请先告诉我你买了哪些股票，或使用 add_trade_record 工具添加记录。"},
            ensure_ascii=False,
        )
    try:
        df = mem.to_trades_df()
        codes = sorted(df["ts_code"].unique().tolist())
        hint = (as_of_date or "").strip().replace("-", "") or None
        last_closes, as_of = fetch_last_closes(codes, hint)
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
