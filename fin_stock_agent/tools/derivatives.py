"""
Derivatives tools
==================
Convertible bonds (可转债): cb_basic / cb_daily
Futures (期货): fut_basic / fut_daily
Options info: opt_basic

Note: These APIs generally require higher Tushare points levels.
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
        return {"ok": True, "rows": 0, "data": [], "note": "无数据（可能需要更高 Tushare 积分）"}
    out = df.head(max_rows)
    return {
        "ok": True,
        "rows": int(len(df)),
        "truncated": len(df) > max_rows,
        "data": out.to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# Convertible Bonds (可转债)
# ---------------------------------------------------------------------------

@tool
def search_convertible_bond(
    keyword: Annotated[str, "可转债代码（如 127043.SZ）或正股名称/代码关键字（如 海天味业 / 603288）"],
) -> str:
    """搜索可转债基础信息（cb_basic API）：正股代码、发行规模、转股价、上市日期等。"""
    try:
        c = get_client()
        kw = (keyword or "").strip()
        if not kw:
            return json.dumps({"ok": False, "error": "keyword 不能为空"}, ensure_ascii=False)

        if "." in kw.upper() and kw[0].isdigit():
            df = c.call(
                "cb_basic",
                ts_code=kw.upper(),
                fields="ts_code,bond_short_name,stk_code,stk_short_name,issue_type,issue_size,maturity,list_date,delist_date,conv_price,conv_start_date,conv_end_date",
            )
        else:
            df = c.call(
                "cb_basic",
                fields="ts_code,bond_short_name,stk_code,stk_short_name,issue_type,issue_size,maturity,list_date,delist_date,conv_price,conv_start_date,conv_end_date",
            )
            if df is not None and not df.empty:
                mask = (
                    df["bond_short_name"].str.contains(kw, case=False, na=False)
                    | df["stk_short_name"].str.contains(kw, case=False, na=False)
                    | df["stk_code"].str.contains(kw, case=False, na=False)
                )
                df = df.loc[mask]

        return json.dumps(_df_to_payload(df, max_rows=30), ensure_ascii=False, default=str)
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


@tool
def get_cb_daily(
    ts_code: Annotated[str, "可转债代码，Tushare 格式如 127043.SZ"],
    start_date: Annotated[str, "开始日期 YYYYMMDD"],
    end_date: Annotated[str, "结束日期 YYYYMMDD"],
) -> str:
    """获取可转债日线行情（cb_daily API）：开/高/低/收/量/转股溢价率/纯债溢价率等。"""
    try:
        c = get_client()
        df = c.call(
            "cb_daily",
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
# Futures (期货)
# ---------------------------------------------------------------------------

_FUTURES_EXCHANGES = {
    "SHFE": "上期所（铜/铝/锌/黄金/白银）",
    "DCE": "大商所（豆粕/玉米/铁矿石）",
    "CZCE": "郑商所（棉花/白糖/PTA）",
    "INE": "上期能源（原油/天然气）",
    "CFFEX": "中金所（股指/国债期货）",
}


@tool
def search_futures(
    keyword: Annotated[str, "期货代码前缀（如 CU、RB、IF）或品种名称（如 铜、螺纹钢、沪深300股指）"],
    exchange: Annotated[str, "交易所：SHFE/DCE/CZCE/INE/CFFEX；留空搜索全部"] = "",
) -> str:
    """搜索期货合约基础信息（fut_basic API）：合约代码、品种名、上市日、到期日。"""
    try:
        c = get_client()
        kw = (keyword or "").strip()
        kwargs: dict = {}
        if exchange:
            kwargs["exchange"] = exchange.upper()

        df = c.call(
            "fut_basic",
            **kwargs,
            fields="ts_code,symbol,name,exchange,fut_code,multiplier,trade_unit,per_unit,quote_unit,quote_unit_desc,d_mode_desc,list_date,delist_date,d_month,last_ddate,trade_time_desc",
        )
        if df is not None and not df.empty and kw:
            mask = (
                df["name"].str.contains(kw, case=False, na=False)
                | df["symbol"].str.contains(kw, case=False, na=False)
                | df.get("fut_code", pd.Series(dtype=str)).str.contains(kw, case=False, na=False)
            )
            df = df.loc[mask]

        return json.dumps(_df_to_payload(df, max_rows=30), ensure_ascii=False, default=str)
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


@tool
def get_futures_daily(
    ts_code: Annotated[str, "期货合约代码，Tushare 格式如 CU2406.SHFE"],
    start_date: Annotated[str, "开始日期 YYYYMMDD"],
    end_date: Annotated[str, "结束日期 YYYYMMDD"],
    exchange: Annotated[str, "交易所代码 SHFE/DCE/CZCE/INE/CFFEX，可留空"] = "",
) -> str:
    """获取期货合约日线行情（fut_daily API）：开/高/低/收/量/持仓量/结算价等。"""
    try:
        c = get_client()
        kwargs: dict = {
            "ts_code": ts_code.strip().upper(),
            "start_date": start_date.replace("-", ""),
            "end_date": end_date.replace("-", ""),
        }
        if exchange:
            kwargs["exchange"] = exchange.upper()
        df = c.call("fut_daily", **kwargs, use_cache=True)
        if df is not None and not df.empty and "trade_date" in df.columns:
            df = df.sort_values("trade_date", ascending=False)
        return json.dumps(_df_to_payload(df, max_rows=500), ensure_ascii=False, default=str)
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Options basic info
# ---------------------------------------------------------------------------

@tool
def search_options(
    underlying: Annotated[str, "标的资产代码，如 510050.SH（50ETF期权）000300.SH（沪深300期权）"],
    exchange: Annotated[str, "交易所代码 SSE/SZSE/CFFEX；留空搜索全部"] = "",
) -> str:
    """搜索期权合约基础信息（opt_basic API）：合约代码、行权价、到期日、Call/Put 类型等。"""
    try:
        c = get_client()
        kwargs: dict = {"underlying": underlying.strip().upper()}
        if exchange:
            kwargs["exchange"] = exchange.upper()
        df = c.call(
            "opt_basic",
            **kwargs,
            fields="ts_code,name,exercise_type,underlying,underlying_type,call_put,exercise_price,s_month,maturity_date,list_date,delist_date,exchange",
        )
        return json.dumps(_df_to_payload(df, max_rows=50), ensure_ascii=False, default=str)
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


def get_derivatives_tools():
    return [
        search_convertible_bond,
        get_cb_daily,
        search_futures,
        get_futures_daily,
        search_options,
    ]
