"""
Fundamental analysis tools
===========================
Financial statements (income / balance / cashflow), financial ratios,
earnings forecasts, and quantitative stock screening.
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
# Financial statements
# ---------------------------------------------------------------------------

@tool
def get_income_statement(
    ts_code: Annotated[str, "股票代码，Tushare 格式如 600519.SH"],
    start_date: Annotated[str, "报告起始日期 YYYYMMDD，默认近两年"],
    end_date: Annotated[str, "报告结束日期 YYYYMMDD，默认今日"],
) -> str:
    """获取股票利润表数据（income API）：营业收入、净利润、归母净利润等，用于分析盈利能力。"""
    try:
        c = get_client()
        df = c.call(
            "income",
            ts_code=ts_code.strip().upper(),
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            fields="ts_code,ann_date,f_ann_date,end_date,report_type,total_revenue,revenue,total_profit,n_income,n_income_attr_p",
            use_cache=True,
        )
        if df is not None and not df.empty and "end_date" in df.columns:
            df = df.sort_values("end_date", ascending=False)
        return json.dumps(_df_to_payload(df, max_rows=20), ensure_ascii=False, default=str)
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


@tool
def get_balance_sheet(
    ts_code: Annotated[str, "股票代码，Tushare 格式如 600519.SH"],
    start_date: Annotated[str, "报告起始日期 YYYYMMDD，默认近两年"],
    end_date: Annotated[str, "报告结束日期 YYYYMMDD，默认今日"],
) -> str:
    """获取股票资产负债表（balancesheet API）：总资产、总负债、净资产、货币资金等。"""
    try:
        c = get_client()
        df = c.call(
            "balancesheet",
            ts_code=ts_code.strip().upper(),
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            fields="ts_code,ann_date,f_ann_date,end_date,report_type,total_assets,total_liab,total_hldr_eqy_inc_min_int,money_cap",
            use_cache=True,
        )
        if df is not None and not df.empty and "end_date" in df.columns:
            df = df.sort_values("end_date", ascending=False)
        return json.dumps(_df_to_payload(df, max_rows=20), ensure_ascii=False, default=str)
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


@tool
def get_cashflow(
    ts_code: Annotated[str, "股票代码，Tushare 格式如 600519.SH"],
    start_date: Annotated[str, "报告起始日期 YYYYMMDD，默认近两年"],
    end_date: Annotated[str, "报告结束日期 YYYYMMDD，默认今日"],
) -> str:
    """获取股票现金流量表（cashflow API）：经营/投资/筹资活动现金流净额，自由现金流分析必备。"""
    try:
        c = get_client()
        df = c.call(
            "cashflow",
            ts_code=ts_code.strip().upper(),
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            fields="ts_code,ann_date,f_ann_date,end_date,report_type,n_cashflow_act,n_cashflow_inv_act,n_cash_flows_fnc_act,n_incr_cash_cash_equ",
            use_cache=True,
        )
        if df is not None and not df.empty and "end_date" in df.columns:
            df = df.sort_values("end_date", ascending=False)
        return json.dumps(_df_to_payload(df, max_rows=20), ensure_ascii=False, default=str)
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


@tool
def get_financial_indicators(
    ts_code: Annotated[str, "股票代码，Tushare 格式如 600519.SH"],
    start_date: Annotated[str, "起始日期 YYYYMMDD，默认近两年"],
    end_date: Annotated[str, "结束日期 YYYYMMDD，默认今日"],
) -> str:
    """获取股票财务指标汇总（fina_indicator API）：ROE、ROA、毛利率、净利率、资产负债率、EPS 等核心财务比率。"""
    try:
        c = get_client()
        df = c.call(
            "fina_indicator",
            ts_code=ts_code.strip().upper(),
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            fields="ts_code,ann_date,end_date,eps,bps,roe,roa,grossprofit_margin,netprofit_margin,debt_to_assets,current_ratio,quick_ratio,inv_turn,ar_turn",
            use_cache=True,
        )
        if df is not None and not df.empty and "end_date" in df.columns:
            df = df.sort_values("end_date", ascending=False)
        return json.dumps(_df_to_payload(df, max_rows=20), ensure_ascii=False, default=str)
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


@tool
def get_financial_forecast(
    ts_code: Annotated[str, "股票代码，Tushare 格式如 600519.SH"],
    start_date: Annotated[str, "公告起始日期 YYYYMMDD，默认近半年"],
    end_date: Annotated[str, "公告结束日期 YYYYMMDD，默认今日"],
) -> str:
    """获取公司业绩预告（forecast API）：预期净利润增减幅、类型（预增/预减/扭亏）等。"""
    try:
        c = get_client()
        df = c.call(
            "forecast",
            ts_code=ts_code.strip().upper(),
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            use_cache=True,
        )
        if df is not None and not df.empty and "ann_date" in df.columns:
            df = df.sort_values("ann_date", ascending=False)
        return json.dumps(_df_to_payload(df, max_rows=20), ensure_ascii=False, default=str)
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Quantitative stock screener
# ---------------------------------------------------------------------------

@tool
def screen_stocks(
    pe_max: Annotated[float | None, "最大市盈率 PE(TTM)，如 30"] = None,
    pe_min: Annotated[float | None, "最小市盈率 PE(TTM)"] = None,
    pb_max: Annotated[float | None, "最大市净率 PB，如 5"] = None,
    mv_min_yi: Annotated[float | None, "最小市值（亿元），如 100 表示 100 亿"] = None,
    mv_max_yi: Annotated[float | None, "最大市值（亿元）"] = None,
    dv_min: Annotated[float | None, "最小股息率 %，如 3 表示 >3%"] = None,
    turnover_max: Annotated[float | None, "最大换手率 %"] = None,
    industry: Annotated[str | None, "申万行业关键字，如 银行、半导体"] = None,
    trade_date: Annotated[str, "查询日期 YYYYMMDD，留空自动取最近有效交易日"] = "",
    limit: Annotated[int, "返回最多多少条，默认 20"] = 20,
) -> str:
    """
    量化条件选股（daily_basic API）：按 PE、PB、市值、股息率、换手率、行业等多维度筛选 A 股。
    返回符合条件的股票列表（市值从大到小排序）。
    """
    try:
        c = get_client()
        td = (trade_date or "").strip().replace("-", "")

        if not td:
            from datetime import datetime
            cal = c.call(
                "trade_cal", exchange="SSE",
                end_date=datetime.now().strftime("%Y%m%d"), is_open="1",
            )
            if cal is not None and not cal.empty:
                for _, row in cal.sort_values("cal_date", ascending=False).head(10).iterrows():
                    d = str(row["cal_date"]).replace("-", "")
                    test = c.call("daily_basic", trade_date=d,
                                  fields="ts_code,trade_date,pe_ttm,pb,total_mv,turnover_rate,dv_ratio")
                    if test is not None and not test.empty:
                        td = d
                        break
            if not td:
                return json.dumps({"ok": False, "error": "无法获取最近有效交易日 daily_basic"}, ensure_ascii=False)

        df = c.call(
            "daily_basic",
            trade_date=td,
            fields="ts_code,trade_date,close,pe_ttm,pb,total_mv,turnover_rate,dv_ratio",
        )
        if df is None or df.empty:
            return json.dumps({"ok": False, "error": f"{td} 无 daily_basic 数据"}, ensure_ascii=False)

        # Filter by industry if requested
        if industry:
            basics = c.call("stock_basic", list_status="L",
                            fields="ts_code,name,industry")
            if basics is not None and not basics.empty:
                matched = basics[basics["industry"].str.contains(industry, case=False, na=False)]
                df = df[df["ts_code"].isin(matched["ts_code"])]

        # Numeric filters (total_mv in Tushare is 万元, so 1亿 = 10000)
        if pe_min is not None:
            df = df[(df["pe_ttm"] >= pe_min) | df["pe_ttm"].isna()]
        if pe_max is not None:
            df = df[df["pe_ttm"] <= pe_max]
        if pb_max is not None:
            df = df[df["pb"] <= pb_max]
        if mv_min_yi is not None:
            df = df[df["total_mv"] >= mv_min_yi * 10000]
        if mv_max_yi is not None:
            df = df[df["total_mv"] <= mv_max_yi * 10000]
        if dv_min is not None:
            df = df[df["dv_ratio"] >= dv_min]
        if turnover_max is not None:
            df = df[df["turnover_rate"] <= turnover_max]

        df = df.sort_values("total_mv", ascending=False).head(max(1, int(limit)))

        # Attach stock names
        try:
            basics = c.call("stock_basic", list_status="L", fields="ts_code,name,industry")
            if basics is not None and not basics.empty:
                df = df.merge(basics, on="ts_code", how="left")
        except Exception:
            pass

        return json.dumps(
            {"ok": True, "trade_date": td, **_df_to_payload(df, max_rows=int(limit))},
            ensure_ascii=False, default=str,
        )
    except TushareRequestError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)


def get_fundamentals_tools():
    return [
        get_income_statement,
        get_balance_sheet,
        get_cashflow,
        get_financial_indicators,
        get_financial_forecast,
        screen_stocks,
    ]
