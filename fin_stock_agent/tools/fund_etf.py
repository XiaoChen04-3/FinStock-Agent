"""Fund / ETF tools using Tushare fund_basic / fund_daily / fund_nav APIs."""
from __future__ import annotations

import json
from typing import Annotated

import pandas as pd
from langchain_core.tools import tool

from fin_stock_agent.core.exceptions import TushareRequestError
from fin_stock_agent.utils.tushare_client import get_client


def _df_to_payload(df: pd.DataFrame, *, max_rows: int = 200) -> dict:
    if df is None or df.empty:
        return {"ok": True, "rows": 0, "data": [], "note": "无数据"}
    out = df.head(max_rows)
    return {
        "ok": True,
        "rows": int(len(df)),
        "truncated": len(df) > max_rows,
        "data": out.to_dict(orient="records"),
    }


@tool
def search_fund(
    keyword: Annotated[str, "基金名称关键字（如 白酒、科技、新能源）或基金代码（如 512690.SH）"],
) -> str:
    """按名称关键字或代码搜索基金基础信息（fund_basic），涵盖 ETF、LOF、混合、指数等类型。"""
    try:
        c = get_client()
        kw = (keyword or "").strip()
        if not kw:
            return json.dumps({"ok": False, "error": "keyword 不能为空"}, ensure_ascii=False)

        if "." in kw.upper():
            df = c.call(
                "fund_basic",
                ts_code=kw.upper(),
                fields="ts_code,name,management,fund_type,found_date,status,market",
            )
        else:
            df = c.call(
                "fund_basic",
                fields="ts_code,name,management,fund_type,found_date,status,market",
            )
            if df is not None and not df.empty:
                mask = (
                    df["name"].str.contains(kw, case=False, na=False)
                    | df["ts_code"].str.contains(kw, case=False, na=False)
                )
                df = df.loc[mask]

        return json.dumps(_df_to_payload(df, max_rows=50), ensure_ascii=False, default=str)
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


@tool
def search_sector_etf(
    keyword: Annotated[str, "主题关键字，如 白酒、新能源、芯片、军工、医疗"],
) -> str:
    """按主题关键字搜索在上交所/深交所上市的主题型 ETF（fund_basic，market=E）。"""
    try:
        c = get_client()
        kw = (keyword or "").strip()
        if not kw:
            return json.dumps({"ok": False, "error": "keyword 不能为空"}, ensure_ascii=False)

        df = c.call(
            "fund_basic",
            market="E",
            fields="ts_code,name,management,fund_type,found_date,status",
        )
        if df is None or df.empty:
            return json.dumps({"ok": True, "rows": 0, "data": [], "note": "暂无 ETF 数据或权限不足"}, ensure_ascii=False)

        mask = df["name"].str.contains(kw, case=False, na=False)
        etfs = df.loc[mask]
        return json.dumps(_df_to_payload(etfs, max_rows=30), ensure_ascii=False, default=str)
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


@tool
def get_fund_daily(
    ts_code: Annotated[str, "基金代码，Tushare 格式如 512690.SH"],
    start_date: Annotated[str, "开始日期 YYYYMMDD"],
    end_date: Annotated[str, "结束日期 YYYYMMDD"],
) -> str:
    """获取场内 ETF/基金日线行情（fund_daily）：open/high/low/close/vol/pct_chg 等。"""
    try:
        c = get_client()
        df = c.call(
            "fund_daily",
            ts_code=ts_code.strip().upper(),
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            use_cache=True,
        )
        return json.dumps(_df_to_payload(df, max_rows=500), ensure_ascii=False, default=str)
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


@tool
def get_fund_nav(
    ts_code: Annotated[str, "基金代码，Tushare 格式如 000001.OF（场外基金用 .OF 后缀）"],
    start_date: Annotated[str, "开始日期 YYYYMMDD"],
    end_date: Annotated[str, "结束日期 YYYYMMDD"],
) -> str:
    """获取基金净值历史（fund_nav）：单位净值、累计净值；主要用于场外基金或 LOF。"""
    try:
        c = get_client()
        df = c.call(
            "fund_nav",
            ts_code=ts_code.strip().upper(),
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            use_cache=True,
        )
        return json.dumps(_df_to_payload(df, max_rows=500), ensure_ascii=False, default=str)
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


def get_fund_etf_tools():
    return [search_fund, search_sector_etf, get_fund_daily, get_fund_nav]
