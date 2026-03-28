"""
Tushare API permission levels mapped to the minimum score required for each tool.

Reference: https://tushare.pro/document/1?doc_id=108
Permission tiers: 120(free) / 2000 / 3000 / 5000 / 6000 / 8000 / 10000 / 15000
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Score threshold per tool name (matches @tool function names)
# ---------------------------------------------------------------------------
TOOL_MIN_SCORES: dict[str, int] = {
    # ── Always available (our own tools, no Tushare needed) ─────────────────
    "get_current_datetime": 0,
    "add_trade_record": 0,
    "get_portfolio_positions": 0,
    "clear_portfolio_memory": 0,
    "calculate_portfolio_pnl": 0,

    # ── 120积分 (免费) ────────────────────────────────────────────────────────
    "search_stock": 120,           # stock_basic – 基础股票列表

    # ── 2000积分 – 低频行情 / 基础数据 / 财务三大报表 / 宏观经济 ────────────────
    "get_daily_bars": 2000,        # daily        – A股日线
    "get_daily_basic_snapshot": 2000,  # daily_basic  – 每日指标
    "get_index_daily": 2000,       # index_daily  – 指数日线
    "get_index_basic": 2000,       # index_basic  – 指数基础信息
    "get_major_indices_performance": 2000,
    "get_sw_industry_top_movers": 2000,
    "get_income_statement": 2000,  # income       – 利润表
    "get_balance_sheet": 2000,     # balancesheet – 资产负债表
    "get_cashflow": 2000,          # cashflow     – 现金流量表
    "get_financial_indicators": 2000,  # fina_indicator
    "get_financial_forecast": 2000,    # forecast
    "screen_stocks": 2000,
    "get_shibor": 2000,            # shibor       – 银行间利率
    "get_cpi": 2000,               # cn_cpi       – CPI
    "get_m2": 2000,                # cn_m        – M2
    "get_gdp": 2000,               # cn_gdp       – GDP

    # ── 3000积分 – 参考数据 / 沪港通列表 / 概念板块 ──────────────────────────
    "search_sector_index": 2000,   # index_basic  – 行业/主题指数搜索
    "get_concept_list": 3000,      # concept      – 概念/题材列表
    "get_concept_stocks": 3000,    # concept_detail
    "get_index_members": 3000,     # index_member – 指数成分股
    "get_top10_holders": 3000,     # top10_holders
    "get_top10_floatholders": 3000,
    "get_pledge_stat": 3000,       # pledge_stat  – 股权质押

    # ── 5000积分 – 资金流向 / 龙虎榜 / 融资融券 / 可转债 / 期货 ─────────────
    "get_moneyflow": 5000,         # moneyflow    – 个股资金流向
    "get_moneyflow_hsgt": 5000,    # moneyflow_hsgt – 北向/南向资金
    "get_hsgt_top10": 5000,        # hsgt_top10   – 沪深港通十大成交股
    "get_limit_list": 5000,        # limit_list   – 涨跌停列表
    "get_top_list": 5000,          # top_list     – 龙虎榜
    "search_convertible_bond": 5000,   # cb_basic
    "get_cb_daily": 5000,          # cb_daily     – 可转债日线
    "search_futures": 5000,        # fut_basic    – 期货基础信息
    "get_futures_daily": 5000,     # fut_daily    – 期货日线

    # ── 6000积分 – ETF/基金 / 海外指数 / 期权 ────────────────────────────────
    "search_fund": 6000,           # fund_basic   – 基金基础信息
    "search_sector_etf": 6000,     # fund_basic   – 主题ETF搜索
    "get_fund_daily": 6000,        # fund_daily   – 场内ETF日线
    "get_fund_nav": 6000,          # fund_nav     – 场外基金净值
    "get_global_index_daily": 6000,    # index_global – 海外指数
    "search_options": 6000,        # opt_basic    – 期权基础信息

    # ── 8000积分 – 港股行情 ───────────────────────────────────────────────────
    "search_hk_stock": 8000,       # hk_basic     – 港股基础信息
    "get_hk_daily": 8000,          # hk_daily     – 港股日线

    # ── 10000积分 – 美股行情 ──────────────────────────────────────────────────
    "search_us_stock": 10000,      # us_basic     – 美股基础信息
    "get_us_daily": 10000,         # us_daily     – 美股日线
}

# Human-readable descriptions for each tier
TIER_DESCRIPTIONS: dict[int, str] = {
    0:     "内置功能（无需Tushare积分）",
    120:   "120积分（免费）：股票基础搜索",
    2000:  "2000积分：A股日线、财务三大报表、宏观经济（GDP/CPI/M2/SHIBOR）",
    3000:  "3000积分：概念板块、指数成分股、股东/质押数据",
    5000:  "5000积分：资金流向、龙虎榜、涨跌停、可转债、期货",
    6000:  "6000积分：ETF/基金行情、海外指数、期权",
    8000:  "8000积分：港股行情",
    10000: "10000积分：美股行情",
}


def get_available_tool_names(score: int) -> set[str]:
    """
    Return the set of tool names accessible at the given Tushare score.

    When score == 0 (not specified), ALL tools are returned so existing
    behaviour is preserved.
    """
    if score <= 0:
        return set(TOOL_MIN_SCORES.keys())  # unrestricted
    return {name for name, min_score in TOOL_MIN_SCORES.items() if score >= min_score}


def describe_available_tiers(score: int) -> str:
    """Return a short human-readable string listing which tiers are unlocked."""
    if score <= 0:
        return "未配置（默认开放全部工具）"
    unlocked = [desc for tier, desc in sorted(TIER_DESCRIPTIONS.items()) if score >= tier]
    return "；".join(unlocked) if unlocked else "积分不足，仅可用内置功能"
