"""
Global markets tools
=====================
Hong Kong stocks (hk_basic / hk_daily) and
US stocks (us_basic / us_daily) via Tushare Pro API.

Note: These APIs require sufficient Tushare points. The tools gracefully
return structured error messages when permissions are insufficient.
"""
from __future__ import annotations

import json
from typing import Annotated

import pandas as pd
from langchain_core.tools import tool

from fin_stock_agent.core.exceptions import TushareRequestError
from fin_stock_agent.utils.tushare_client import get_client


def _df_to_payload(df: pd.DataFrame, *, max_rows: int = 200) -> dict:
    if df is None or df.empty:
        return {"ok": True, "rows": 0, "data": [], "note": "无数据（可能需要更高 Tushare 积分或港/美股权限）"}
    out = df.head(max_rows)
    return {
        "ok": True,
        "rows": int(len(df)),
        "truncated": len(df) > max_rows,
        "data": out.to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# Hong Kong stocks
# ---------------------------------------------------------------------------

@tool
def search_hk_stock(
    keyword: Annotated[str, "港股代码（如 00700.HK）或名称关键字（如 腾讯控股）"],
) -> str:
    """按代码或名称搜索港股基础信息（hk_basic API）：行业、上市日期、市场等。"""
    try:
        c = get_client()
        kw = (keyword or "").strip()
        if not kw:
            return json.dumps({"ok": False, "error": "keyword 不能为空"}, ensure_ascii=False)

        if "." in kw.upper():
            df = c.call(
                "hk_basic",
                ts_code=kw.upper(),
                fields="ts_code,name,fullname,enname,cn_spell,market,list_status,list_date,delist_date,trade_unit,isin,curr_type",
            )
        else:
            df = c.call(
                "hk_basic",
                list_status="L",
                fields="ts_code,name,fullname,enname,cn_spell,market,list_status,list_date,delist_date,trade_unit,isin,curr_type",
            )
            if df is not None and not df.empty:
                mask = (
                    df["name"].str.contains(kw, case=False, na=False)
                    | df.get("fullname", pd.Series(dtype=str)).str.contains(kw, case=False, na=False)
                )
                df = df.loc[mask]

        return json.dumps(_df_to_payload(df, max_rows=30), ensure_ascii=False, default=str)
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


@tool
def get_hk_daily(
    ts_code: Annotated[str, "港股代码，Tushare 格式如 00700.HK"],
    start_date: Annotated[str, "开始日期 YYYYMMDD"],
    end_date: Annotated[str, "结束日期 YYYYMMDD"],
) -> str:
    """获取港股日线行情（hk_daily API）：开/高/低/收/量/涨跌幅等。"""
    try:
        c = get_client()
        df = c.call(
            "hk_daily",
            ts_code=ts_code.strip().upper(),
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            use_cache=True,
        )
        if df is not None and not df.empty and "trade_date" in df.columns:
            df = df.sort_values("trade_date", ascending=False)
        return json.dumps(_df_to_payload(df, max_rows=500), ensure_ascii=False, default=str)
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# US stocks
# ---------------------------------------------------------------------------

@tool
def search_us_stock(
    keyword: Annotated[str, "美股代码（如 AAPL.O）或名称关键字（如 Apple / 苹果）"],
) -> str:
    """按代码或名称搜索美股基础信息（us_basic API）：行业、交易所、上市日期等。"""
    try:
        c = get_client()
        kw = (keyword or "").strip()
        if not kw:
            return json.dumps({"ok": False, "error": "keyword 不能为空"}, ensure_ascii=False)

        if "." in kw.upper():
            df = c.call(
                "us_basic",
                ts_code=kw.upper(),
                fields="ts_code,name,fullname,enname,exchange,list_status,list_date,delist_date,currency",
            )
        else:
            df = c.call(
                "us_basic",
                list_status="L",
                fields="ts_code,name,fullname,enname,exchange,list_status,list_date,delist_date,currency",
            )
            if df is not None and not df.empty:
                mask = (
                    df["name"].str.contains(kw, case=False, na=False)
                    | df.get("fullname", pd.Series(dtype=str)).str.contains(kw, case=False, na=False)
                    | df.get("enname", pd.Series(dtype=str)).str.contains(kw, case=False, na=False)
                )
                df = df.loc[mask]

        return json.dumps(_df_to_payload(df, max_rows=30), ensure_ascii=False, default=str)
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


@tool
def get_us_daily(
    ts_code: Annotated[str, "美股代码，Tushare 格式如 AAPL.O（纳斯达克用 .O，纽交所用 .N）"],
    start_date: Annotated[str, "开始日期 YYYYMMDD"],
    end_date: Annotated[str, "结束日期 YYYYMMDD"],
) -> str:
    """获取美股日线行情（us_daily API）：开/高/低/收/量/涨跌幅等。"""
    try:
        c = get_client()
        df = c.call(
            "us_daily",
            ts_code=ts_code.strip().upper(),
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            use_cache=True,
        )
        if df is not None and not df.empty and "trade_date" in df.columns:
            df = df.sort_values("trade_date", ascending=False)
        return json.dumps(_df_to_payload(df, max_rows=500), ensure_ascii=False, default=str)
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Global index (overseas indices via index_global API)
# ---------------------------------------------------------------------------

@tool
def get_global_index_daily(
    ts_code: Annotated[
        str,
        "全球指数代码，如 HSI.HI（恒生指数）SPX.GI（标普500）IXIC.GI（纳斯达克）DJI.GI（道琼斯）",
    ],
    start_date: Annotated[str, "开始日期 YYYYMMDD"],
    end_date: Annotated[str, "结束日期 YYYYMMDD"],
) -> str:
    """获取海外指数日线行情（index_global API）：恒生指数、标普500、纳斯达克、道琼斯等。"""
    try:
        c = get_client()
        df = c.call(
            "index_global",
            ts_code=ts_code.strip().upper(),
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            use_cache=True,
        )
        if df is not None and not df.empty and "trade_date" in df.columns:
            df = df.sort_values("trade_date", ascending=False)
        return json.dumps(_df_to_payload(df, max_rows=500), ensure_ascii=False, default=str)
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


def get_global_market_tools():
    return [
        search_hk_stock,
        get_hk_daily,
        search_us_stock,
        get_us_daily,
        get_global_index_daily,
    ]
