from __future__ import annotations

import json
from datetime import datetime
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
def search_stock(
    keyword_or_code: Annotated[
        str,
        "股票代码如 600519.SH / 000001.SZ，或名称关键字如 茅台",
    ],
) -> str:
    """按 ts_code 精确匹配或按名称模糊搜索 A 股基础信息（stock_basic）。"""
    try:
        c = get_client()
        kw = (keyword_or_code or "").strip()
        if not kw:
            return json.dumps({"ok": False, "error": "keyword_or_code 不能为空"}, ensure_ascii=False)
        if "." in kw.upper():
            df = c.call(
                "stock_basic",
                ts_code=kw.upper(),
                list_status="L",
                fields="ts_code,symbol,name,area,industry,list_date,market,exchange",
            )
        else:
            df = c.call(
                "stock_basic",
                list_status="L",
                fields="ts_code,symbol,name,area,industry,list_date,market,exchange",
            )
            if df is not None and not df.empty:
                mask = df["name"].str.contains(kw, case=False, na=False) | df[
                    "ts_code"
                ].str.contains(kw, case=False, na=False)
                df = df.loc[mask]
        return json.dumps(_df_to_payload(df, max_rows=50), ensure_ascii=False, default=str)
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


@tool
def get_daily_bars(
    ts_code: Annotated[str, "Tushare 格式代码，如 600519.SH"],
    start_date: Annotated[str, "开始日期 YYYYMMDD"],
    end_date: Annotated[str, "结束日期 YYYYMMDD"],
) -> str:
    """获取股票日线行情（未复权）：open/high/low/close/vol/pct_chg 等。"""
    try:
        c = get_client()
        df = c.call(
            "daily",
            ts_code=ts_code.strip().upper(),
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            use_cache=True,
        )
        return json.dumps(_df_to_payload(df, max_rows=500), ensure_ascii=False, default=str)
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


@tool
def get_daily_basic_snapshot(
    trade_date: Annotated[str, "交易日 YYYYMMDD，留空则尝试最近约10个交易日"],
) -> str:
    """全市场当日估值快照（daily_basic）：PE、PB、换手率、市值等；需足够 Tushare 积分。"""
    try:
        c = get_client()
        td = (trade_date or "").strip().replace("-", "")
        if not td:
            cal = c.call(
                "trade_cal", exchange="SSE",
                end_date=datetime.now().strftime("%Y%m%d"), is_open="1",
            )
            if cal is None or cal.empty:
                return json.dumps({"ok": False, "error": "无法获取交易日历"}, ensure_ascii=False)
            last_open = None
            for _, row in cal.sort_values("cal_date", ascending=False).head(12).iterrows():
                d = str(row["cal_date"]).replace("-", "")
                test = c.call("daily_basic", trade_date=d)
                if test is not None and not test.empty:
                    last_open = d
                    break
            if not last_open:
                return json.dumps(
                    {"ok": False, "error": "近期无 daily_basic 数据或权限不足"}, ensure_ascii=False
                )
            td = last_open
        df = c.call("daily_basic", trade_date=td)
        meta = {"trade_date": td, **_df_to_payload(df, max_rows=150)}
        return json.dumps(meta, ensure_ascii=False, default=str)
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


@tool
def get_index_daily(
    ts_code: Annotated[str, "指数代码，如 000001.SH 上证指数"],
    start_date: Annotated[str, "YYYYMMDD"],
    end_date: Annotated[str, "YYYYMMDD"],
) -> str:
    """指数日线行情（index_daily）。"""
    try:
        c = get_client()
        df = c.call(
            "index_daily",
            ts_code=ts_code.strip().upper(),
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            use_cache=True,
        )
        return json.dumps(_df_to_payload(df, max_rows=600), ensure_ascii=False, default=str)
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


_MAJOR_INDEX = {
    "上证指数": "000001.SH",
    "深证成指": "399001.SZ",
    "创业板指": "399006.SZ",
    "沪深300": "000300.SH",
    "中证500": "000905.SH",
    "科创50": "000688.SH",
}


@tool
def get_major_indices_performance(
    start_date: Annotated[str, "YYYYMMDD"],
    end_date: Annotated[str, "YYYYMMDD"],
) -> str:
    """主要宽基指数区间涨跌幅汇总（基于 index_daily 收盘价）。"""
    try:
        c = get_client()
        s = start_date.replace("-", "")
        e = end_date.replace("-", "")
        rows: list[dict] = []
        for name, code in _MAJOR_INDEX.items():
            df = c.call("index_daily", ts_code=code, start_date=s, end_date=e, use_cache=True)
            if df is None or df.empty:
                rows.append({"name": name, "ts_code": code, "error": "无数据"})
                continue
            d = df.sort_values("trade_date")
            first = float(d.iloc[0]["close"])
            last = float(d.iloc[-1]["close"])
            chg = (last / first - 1.0) * 100 if first else 0.0
            rows.append(
                {
                    "name": name,
                    "ts_code": code,
                    "start_close": first,
                    "end_close": last,
                    "pct_chg_period": round(chg, 4),
                    "start_date": str(d.iloc[0]["trade_date"]),
                    "end_date": str(d.iloc[-1]["trade_date"]),
                }
            )
        return json.dumps({"ok": True, "indices": rows}, ensure_ascii=False, default=str)
    except TushareRequestError as ex:
        return json.dumps({"ok": False, "error": str(ex)}, ensure_ascii=False)


@tool
def get_sw_industry_top_movers(
    trade_date: Annotated[str, "交易日 YYYYMMDD；可留空使用最近有效交易日"],
    top_n: Annotated[int, "返回涨跌前 N 个申万一级行业"] = 5,
) -> str:
    """申万一级行业指数当日涨跌幅排行（sw_daily + index_classify）。积分不足时返回错误说明。"""
    try:
        c = get_client()
        td = (trade_date or "").strip().replace("-", "")
        if not td:
            end_d = datetime.now().strftime("%Y%m%d")
            cal = c.call("trade_cal", exchange="SSE", end_date=end_d, is_open="1")
            if cal is None or cal.empty:
                return json.dumps({"ok": False, "error": "无法获取交易日历"}, ensure_ascii=False)
            for _, row in cal.sort_values("cal_date", ascending=False).head(20).iterrows():
                d = str(row["cal_date"]).replace("-", "")
                test = c.call("sw_daily", trade_date=d)
                if test is not None and not test.empty:
                    td = d
                    break
            if not td:
                return json.dumps(
                    {"ok": False, "error": "无 sw_daily 数据或权限/积分不足"}, ensure_ascii=False
                )

        cls = c.call("index_classify", level="L1", src="SW2021")
        if cls is None or cls.empty:
            cls = c.call("index_classify", level="L1", src="SW2014")
        if cls is None or cls.empty:
            return json.dumps(
                {"ok": False, "error": "index_classify 无数据或权限不足"}, ensure_ascii=False
            )

        codes = cls["index_code"].dropna().unique().tolist()[:25]
        daily_parts: list[pd.DataFrame] = []
        for code in codes:
            part = c.call("sw_daily", ts_code=code, trade_date=td, use_cache=True)
            if part is not None and not part.empty:
                daily_parts.append(part)

        if not daily_parts:
            return json.dumps(
                {"ok": False, "error": f"{td} 无行业日线数据或 sw_daily 权限不足"}, ensure_ascii=False
            )

        df = pd.concat(daily_parts, ignore_index=True)
        name_map = cls.set_index("index_code")["industry_name"].to_dict()
        df["industry_name"] = df["ts_code"].map(name_map).fillna(df["ts_code"])
        pct_col = "pct_change" if "pct_change" in df.columns else "pct_chg"
        if pct_col not in df.columns:
            return json.dumps(
                {"ok": False, "error": "sw_daily 返回中无涨跌幅字段"}, ensure_ascii=False
            )

        n = max(1, min(int(top_n), 30))
        df_sorted = df.sort_values(pct_col, ascending=False, na_position="last")
        top_up = df_sorted.head(n)
        top_down = df_sorted.tail(n).sort_values(pct_col, ascending=True)
        cols = [c for c in ["ts_code", "industry_name", "close", pct_col] if c in df.columns]
        return json.dumps(
            {
                "ok": True,
                "trade_date": td,
                "pct_field": pct_col,
                "top_gainers": top_up[cols].to_dict(orient="records"),
                "top_losers": top_down[cols].to_dict(orient="records"),
            },
            ensure_ascii=False,
            default=str,
        )
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


def get_market_tools():
    return [
        search_stock,
        get_daily_bars,
        get_daily_basic_snapshot,
        get_index_daily,
        get_major_indices_performance,
        get_sw_industry_top_movers,
    ]
