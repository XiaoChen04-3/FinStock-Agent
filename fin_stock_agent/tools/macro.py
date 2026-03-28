"""Macro-economy tools using Tushare shibor / cn_cpi / cn_m / cn_gdp APIs."""
from __future__ import annotations

import json
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


@tool
def get_shibor(
    start_date: Annotated[str, "开始日期 YYYYMMDD，如 20240101"],
    end_date: Annotated[str, "结束日期 YYYYMMDD，如 20240630"],
) -> str:
    """获取 SHIBOR（上海银行间同业拆借利率）数据（shibor API）：隔夜/1周/1月/3月利率。"""
    try:
        c = get_client()
        df = c.call(
            "shibor",
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            use_cache=True,
        )
        return json.dumps(_df_to_payload(df, max_rows=200), ensure_ascii=False, default=str)
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


@tool
def get_cpi(
    start_month: Annotated[str, "开始月份 YYYYMM，如 202401"],
    end_month: Annotated[str, "结束月份 YYYYMM，如 202412"],
) -> str:
    """获取月度 CPI（居民消费价格指数）数据（cn_cpi API）：同比、环比涨跌幅。"""
    try:
        c = get_client()
        start = (start_month or "").replace("-", "")[:6]
        end = (end_month or "").replace("-", "")[:6]
        df = c.call("cn_cpi", start_m=start, end_m=end, use_cache=True)
        return json.dumps(_df_to_payload(df, max_rows=60), ensure_ascii=False, default=str)
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


@tool
def get_m2(
    start_month: Annotated[str, "开始月份 YYYYMM，如 202401"],
    end_month: Annotated[str, "结束月份 YYYYMM，如 202412"],
) -> str:
    """获取月度 M0/M1/M2 货币供应量数据（cn_m API）；反映流动性状况。"""
    try:
        c = get_client()
        start = (start_month or "").replace("-", "")[:6]
        end = (end_month or "").replace("-", "")[:6]
        df = c.call("cn_m", start_m=start, end_m=end, use_cache=True)
        return json.dumps(_df_to_payload(df, max_rows=60), ensure_ascii=False, default=str)
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


@tool
def get_gdp(
    start_year: Annotated[str, "开始年份 YYYY，如 2020"],
    end_year: Annotated[str, "结束年份 YYYY，如 2024"],
) -> str:
    """获取季度 GDP 数据（cn_gdp API）：GDP 总量、同比增速、三大产业拆分。"""
    try:
        c = get_client()
        df = c.call("cn_gdp", start_q=start_year + "Q1", end_q=end_year + "Q4", use_cache=True)
        return json.dumps(_df_to_payload(df, max_rows=40), ensure_ascii=False, default=str)
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


def get_macro_tools():
    return [get_shibor, get_cpi, get_m2, get_gdp]
