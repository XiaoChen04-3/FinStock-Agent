"""
Market activity / microstructure tools
=======================================
Money flow, limit up/down list, Dragon & Tiger list, and
northbound (Shanghai/Shenzhen-Hong Kong Connect) fund data.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Annotated

import pandas as pd
from langchain_core.tools import tool

from fin_stock_agent.core.exceptions import TushareRequestError
from fin_stock_agent.utils.tushare_client import get_client


def _df_to_payload(df: pd.DataFrame, *, max_rows: int = 200) -> dict:
    if df is None or df.empty:
        return {"ok": True, "rows": 0, "data": [], "note": "无数据（可能需要更高 Tushare 积分）"}
    out = df.head(max_rows)
    return {
        "ok": True,
        "rows": int(len(df)),
        "truncated": len(df) > max_rows,
        "data": out.to_dict(orient="records"),
    }


def _resolve_trade_date(trade_date: str, api_name: str = "daily") -> str:
    """Return the most recent valid trading date, trying up to 10 days back."""
    td = (trade_date or "").strip().replace("-", "")
    if td:
        return td
    c = get_client()
    end = datetime.now().strftime("%Y%m%d")
    cal = c.call("trade_cal", exchange="SSE", end_date=end, is_open="1")
    if cal is None or cal.empty:
        return datetime.now().strftime("%Y%m%d")
    for _, row in cal.sort_values("cal_date", ascending=False).head(10).iterrows():
        d = str(row["cal_date"]).replace("-", "")
        try:
            test = c.call(api_name, trade_date=d)
            if test is not None and not test.empty:
                return d
        except Exception:
            continue
    return end


# ---------------------------------------------------------------------------
# Money flow
# ---------------------------------------------------------------------------

@tool
def get_moneyflow(
    ts_code: Annotated[str, "股票代码，Tushare 格式如 600519.SH"],
    start_date: Annotated[str, "开始日期 YYYYMMDD"],
    end_date: Annotated[str, "结束日期 YYYYMMDD"],
) -> str:
    """
    获取个股资金流向（moneyflow API）：主力/超大单/大单/小单净流入额及占比。
    用于判断主力资金进出方向。需要足够 Tushare 积分。
    """
    try:
        c = get_client()
        df = c.call(
            "moneyflow",
            ts_code=ts_code.strip().upper(),
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            use_cache=True,
        )
        if df is not None and not df.empty and "trade_date" in df.columns:
            df = df.sort_values("trade_date", ascending=False)
        return json.dumps(_df_to_payload(df, max_rows=60), ensure_ascii=False, default=str)
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


@tool
def get_moneyflow_hsgt(
    start_date: Annotated[str, "开始日期 YYYYMMDD"],
    end_date: Annotated[str, "结束日期 YYYYMMDD"],
) -> str:
    """
    获取沪深港通资金流向汇总（moneyflow_hsgt API）：沪股通/深股通/港股通每日净买入额。
    反映北向资金整体流入/流出趋势。
    """
    try:
        c = get_client()
        df = c.call(
            "moneyflow_hsgt",
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            use_cache=True,
        )
        if df is not None and not df.empty and "trade_date" in df.columns:
            df = df.sort_values("trade_date", ascending=False)
        return json.dumps(_df_to_payload(df, max_rows=60), ensure_ascii=False, default=str)
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Northbound top 10
# ---------------------------------------------------------------------------

@tool
def get_hsgt_top10(
    trade_date: Annotated[str, "交易日 YYYYMMDD；留空自动取最近有效日"],
    market_type: Annotated[str, "市场类型：3=沪股通 4=深股通；留空返回全部"] = "",
) -> str:
    """
    获取沪深港通每日前十大成交股（hsgt_top10 API）：净买入量、成交额、买/卖总额。
    反映北向资金重点关注的 A 股标的。
    """
    try:
        c = get_client()
        td = _resolve_trade_date(trade_date, "hsgt_top10")
        kwargs: dict = {"trade_date": td}
        if market_type:
            kwargs["market_type"] = market_type
        df = c.call("hsgt_top10", **kwargs)
        return json.dumps(
            {"ok": True, "trade_date": td, **_df_to_payload(df, max_rows=20)},
            ensure_ascii=False, default=str,
        )
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Limit up / down list
# ---------------------------------------------------------------------------

@tool
def get_limit_list(
    trade_date: Annotated[str, "交易日 YYYYMMDD；留空自动取最近有效日"],
    limit_type: Annotated[str, "U=涨停 D=跌停；留空返回全部"] = "",
) -> str:
    """
    获取当日涨跌停股票列表（limit_list API）：封板时间、封板资金、炸板次数等。
    用于分析热点题材和情绪。需要一定 Tushare 积分。
    """
    try:
        c = get_client()
        td = _resolve_trade_date(trade_date, "limit_list")
        kwargs: dict = {"trade_date": td}
        if limit_type:
            kwargs["limit_type"] = limit_type.upper()
        df = c.call("limit_list", **kwargs)
        return json.dumps(
            {"ok": True, "trade_date": td, **_df_to_payload(df, max_rows=100)},
            ensure_ascii=False, default=str,
        )
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Dragon & Tiger list (top_list)
# ---------------------------------------------------------------------------

@tool
def get_top_list(
    trade_date: Annotated[str, "交易日 YYYYMMDD；留空自动取最近有效日"],
) -> str:
    """
    获取龙虎榜（top_list API）：涨跌幅异常或成交量异常股票的营业部买卖明细。
    龙虎榜是机构/游资追踪的重要参考。需要一定 Tushare 积分。
    """
    try:
        c = get_client()
        td = _resolve_trade_date(trade_date, "top_list")
        df = c.call("top_list", trade_date=td)
        return json.dumps(
            {"ok": True, "trade_date": td, **_df_to_payload(df, max_rows=100)},
            ensure_ascii=False, default=str,
        )
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Pledge statistics (stock pledges)
# ---------------------------------------------------------------------------

@tool
def get_pledge_stat(
    ts_code: Annotated[str, "股票代码，Tushare 格式如 600519.SH"],
) -> str:
    """
    获取股票质押统计（pledge_stat API）：质押比例、质押股数、质押总市值等。
    高质押比例是重大风险信号。
    """
    try:
        c = get_client()
        df = c.call("pledge_stat", ts_code=ts_code.strip().upper(), use_cache=True)
        return json.dumps(_df_to_payload(df, max_rows=10), ensure_ascii=False, default=str)
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Shareholder data
# ---------------------------------------------------------------------------

@tool
def get_top10_holders(
    ts_code: Annotated[str, "股票代码，Tushare 格式如 600519.SH"],
    period: Annotated[str, "报告期 YYYYMMDD（季报/年报日期），如 20231231；留空取最新"],
) -> str:
    """
    获取前十大股东（top10_holders API）：持股数量、持股比例（含机构、个人）。
    """
    try:
        c = get_client()
        kwargs: dict = {"ts_code": ts_code.strip().upper()}
        if period:
            kwargs["period"] = period.replace("-", "")
        df = c.call("top10_holders", **kwargs, use_cache=True)
        if df is not None and not df.empty and "end_date" in df.columns:
            df = df.sort_values("end_date", ascending=False)
        return json.dumps(_df_to_payload(df, max_rows=40), ensure_ascii=False, default=str)
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


@tool
def get_top10_floatholders(
    ts_code: Annotated[str, "股票代码，Tushare 格式如 600519.SH"],
    period: Annotated[str, "报告期 YYYYMMDD，如 20231231；留空取最新"],
) -> str:
    """
    获取前十大流通股股东（top10_floatholders API）：流通市场主要持仓人，反映机构持仓变化。
    """
    try:
        c = get_client()
        kwargs: dict = {"ts_code": ts_code.strip().upper()}
        if period:
            kwargs["period"] = period.replace("-", "")
        df = c.call("top10_floatholders", **kwargs, use_cache=True)
        if df is not None and not df.empty and "end_date" in df.columns:
            df = df.sort_values("end_date", ascending=False)
        return json.dumps(_df_to_payload(df, max_rows=40), ensure_ascii=False, default=str)
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


def get_activity_tools():
    return [
        get_moneyflow,
        get_moneyflow_hsgt,
        get_hsgt_top10,
        get_limit_list,
        get_top_list,
        get_pledge_stat,
        get_top10_holders,
        get_top10_floatholders,
    ]
