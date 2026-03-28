"""Sector / concept / index member tools using Tushare concept / index_member APIs."""
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
def get_concept_list(
    keyword: Annotated[str, "概念/板块关键字，如 白酒、芯片、新能源、军工；留空返回所有概念"],
) -> str:
    """按关键字搜索 Tushare 概念板块列表（concept API）。返回 concept_id 和板块名称。"""
    try:
        c = get_client()
        kw = (keyword or "").strip()
        df = c.call("concept", src="ts")
        if df is None or df.empty:
            return json.dumps(
                {"ok": True, "rows": 0, "data": [], "note": "无概念数据或权限不足"},
                ensure_ascii=False,
            )
        if kw:
            mask = df["concept_name"].str.contains(kw, case=False, na=False)
            df = df.loc[mask]
        return json.dumps(_df_to_payload(df, max_rows=50), ensure_ascii=False, default=str)
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


@tool
def get_concept_stocks(
    concept_id: Annotated[str, "Tushare 概念板块 ID（由 get_concept_list 获取，如 TS2（白酒））"],
) -> str:
    """获取某概念板块的全部成分股（concept_detail API）。"""
    try:
        c = get_client()
        cid = (concept_id or "").strip()
        if not cid:
            return json.dumps({"ok": False, "error": "concept_id 不能为空"}, ensure_ascii=False)
        df = c.call("concept_detail", id=cid, fields="ts_code,name,in_date")
        return json.dumps(_df_to_payload(df, max_rows=200), ensure_ascii=False, default=str)
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


@tool
def get_index_basic(
    keyword: Annotated[str, "指数名称关键字，如 沪深300、中证500、白酒、科技；或直接填代码如 000300.SH"],
) -> str:
    """按关键字搜索指数基本信息（index_basic API）：指数名称、代码、发布机构、基准日期等。"""
    try:
        c = get_client()
        kw = (keyword or "").strip()
        if not kw:
            return json.dumps({"ok": False, "error": "keyword 不能为空"}, ensure_ascii=False)

        if "." in kw.upper():
            df = c.call(
                "index_basic",
                ts_code=kw.upper(),
                fields="ts_code,name,fullname,market,publisher,category,base_date,base_point,list_date",
            )
        else:
            # Search across major exchanges
            parts = []
            for mkt in ("SSE", "SZSE", "CSI", "MSCI", "SW"):
                part = c.call(
                    "index_basic",
                    market=mkt,
                    fields="ts_code,name,fullname,market,publisher,category,base_date,base_point,list_date",
                )
                if part is not None and not part.empty:
                    parts.append(part)
            if not parts:
                return json.dumps(
                    {"ok": True, "rows": 0, "data": [], "note": "无指数数据"},
                    ensure_ascii=False,
                )
            df = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["ts_code"])
            mask = (
                df["name"].str.contains(kw, case=False, na=False)
                | df.get("fullname", pd.Series(dtype=str)).str.contains(kw, case=False, na=False)
            )
            df = df.loc[mask]

        return json.dumps(_df_to_payload(df, max_rows=30), ensure_ascii=False, default=str)
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


@tool
def get_index_members(
    ts_code: Annotated[str, "指数代码，Tushare 格式如 000300.SH（沪深300）"],
) -> str:
    """获取指数成分股列表（index_member API）：成分股代码、名称、进出日期等。"""
    try:
        c = get_client()
        code = (ts_code or "").strip().upper()
        if not code:
            return json.dumps({"ok": False, "error": "ts_code 不能为空"}, ensure_ascii=False)
        df = c.call(
            "index_member",
            index_code=code,
            fields="index_code,con_code,con_name,in_date,out_date,is_new",
        )
        if df is not None and not df.empty:
            df = df[df.get("is_new", pd.Series("N", index=df.index)) == "Y"]
        return json.dumps(_df_to_payload(df, max_rows=500), ensure_ascii=False, default=str)
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


def get_sector_tools():
    return [get_concept_list, get_concept_stocks, get_index_basic, get_index_members]
