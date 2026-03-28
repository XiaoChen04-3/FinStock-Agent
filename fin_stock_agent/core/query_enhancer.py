"""
Query Enhancement Framework
============================
Intent recognition → query rewriting → multi-query expansion.

Flow:
  user input
    ↓  enhance_query()
  EnhancedQuery (intent + rewritten + sub_queries + suggested_tools)
    ↓  injected into router._prep_session
  Agent (ReAct / P&E)
"""
from __future__ import annotations

import json
import re
from enum import Enum

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from fin_stock_agent.core.llm import get_llm

# ---------------------------------------------------------------------------
# Intent types
# ---------------------------------------------------------------------------

class IntentType(str, Enum):
    STOCK_QUERY     = "stock_query"     # Single stock query
    SECTOR_ANALYSIS = "sector"          # Sector / industry analysis
    INDEX_COMPARE   = "index_compare"   # Index comparison
    FUND_ETF        = "fund_etf"        # Fund / ETF
    MACRO           = "macro"           # Macro economy
    PORTFOLIO       = "portfolio"       # Holdings / PnL
    GENERAL         = "general"         # Catch-all


# Human-readable labels for display
INTENT_LABELS: dict[IntentType, str] = {
    IntentType.STOCK_QUERY:     "个股查询",
    IntentType.SECTOR_ANALYSIS: "板块/行业分析",
    IntentType.INDEX_COMPARE:   "指数对比",
    IntentType.FUND_ETF:        "基金/ETF",
    IntentType.MACRO:           "宏观经济",
    IntentType.PORTFOLIO:       "持仓/盈亏",
    IntentType.GENERAL:         "通用查询",
}

# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

class EnhancedQuery(BaseModel):
    original: str
    rewritten: str
    intent: IntentType = IntentType.GENERAL
    sub_queries: list[str] = Field(default_factory=list)
    suggested_tools: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)

    def intent_label(self) -> str:
        return INTENT_LABELS.get(self.intent, self.intent.value)

    def to_context_block(self) -> str:
        """Build the structured context block injected into the agent message."""
        if not self.sub_queries:
            return self.rewritten
        lines = [
            f"[查询增强] 意图识别：{self.intent_label()}",
            f"核心问题：{self.rewritten}",
            "",
            "请从以下多个角度分别查询并综合分析：",
        ]
        for i, q in enumerate(self.sub_queries, 1):
            lines.append(f"  {i}. {q}")
        if self.suggested_tools:
            lines.append("")
            lines.append("优先使用工具：" + "、".join(self.suggested_tools))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_ENHANCE_PROMPT = """你是一个专业的金融查询分析助手。
对用户的原始问题进行意图识别、查询重写、多角度扩展。

## 意图类型
- stock_query：询问单只股票的行情、基本面、涨跌等
- sector：询问某板块/行业（如白酒、新能源、半导体）的整体表现
- index_compare：对比多个宽基指数或区间涨跌
- fund_etf：询问某基金或 ETF（如白酒 ETF、科技 ETF）
- macro：询问宏观经济指标（CPI、GDP、利率、M2 等）
- portfolio：询问用户自己的持仓、盈亏计算
- general：其他通用问题

## 输出要求
输出严格的单个 JSON 对象，不加任何说明文字：
{{
  "intent": "<意图类型>",
  "rewritten": "<规范化重写，用于传给 Agent 的精确问题>",
  "sub_queries": ["<角度1>", "<角度2>", "<角度3>"],
  "suggested_tools": ["<工具名1>", "<工具名2>"],
  "keywords": ["<关键实体1>", "<关键实体2>"]
}}

## 规则
- sub_queries 最多 4 条，每条对应一个独立的数据查询角度；若问题简单则可为空数组
- 板块分析时必须包含：①相关主题 ETF ②行业指数 ③龙头股
- suggested_tools 仅从以下名称中选择：
  get_current_datetime,
  search_stock, get_daily_bars, get_daily_basic_snapshot, get_index_daily,
  get_major_indices_performance, get_sw_industry_top_movers,
  search_fund, get_fund_daily, get_fund_nav, search_sector_etf,
  get_concept_list, get_concept_stocks, get_index_members, get_index_basic,
  get_shibor, get_cpi, get_m2, get_gdp,
  get_income_statement, get_balance_sheet, get_cashflow, get_financial_indicators,
  get_financial_forecast, screen_stocks,
  get_moneyflow, get_moneyflow_hsgt, get_hsgt_top10, get_limit_list, get_top_list,
  get_pledge_stat, get_top10_holders, get_top10_floatholders,
  search_hk_stock, get_hk_daily, search_us_stock, get_us_daily, get_global_index_daily,
  search_convertible_bond, get_cb_daily,
  search_futures, get_futures_daily, search_options,
  calculate_portfolio_pnl, get_portfolio_positions, add_trade_record
- keywords 提取股票/ETF/板块/指数名称，最多 6 个
- sub_queries 为空时输出 []

用户原始问题：{question}"""


# ---------------------------------------------------------------------------
# Heuristic fast-path (no LLM call for trivial cases)
# ---------------------------------------------------------------------------

_PORTFOLIO_KEYWORDS = {"持仓", "盈亏", "收益", "亏损", "仓位", "买入记录", "卖出记录"}
_MACRO_KEYWORDS = {"cpi", "ppi", "gdp", "m2", "m1", "shibor", "利率", "通胀", "货币", "经济增长"}
_FUND_KEYWORDS = {"etf", "基金", "净值", "lof", "场内基金"}
_DERIVATIVES_KEYWORDS = {"可转债", "转债", "期货", "期权", "大宗商品", "铜期货", "原油期货", "股指期货"}
_ACTIVITY_KEYWORDS = {"涨停", "跌停", "龙虎榜", "北向资金", "沪股通", "深股通", "资金流入", "资金流出"}
_GLOBAL_KEYWORDS = {"港股", "美股", "恒生", "纳斯达克", "标普", "道琼斯", "美国市场", "香港"}


def _fast_classify(question: str) -> IntentType | None:
    q_lower = question.lower()
    if any(k in q_lower for k in _PORTFOLIO_KEYWORDS):
        return IntentType.PORTFOLIO
    if any(k in q_lower for k in _MACRO_KEYWORDS):
        return IntentType.MACRO
    if any(k in q_lower for k in _FUND_KEYWORDS) and "股票" not in q_lower:
        return IntentType.FUND_ETF
    if any(k in q_lower for k in _DERIVATIVES_KEYWORDS):
        return IntentType.GENERAL   # derivatives mapped to general for now
    if any(k in q_lower for k in _GLOBAL_KEYWORDS):
        return IntentType.INDEX_COMPARE
    if any(k in q_lower for k in _ACTIVITY_KEYWORDS):
        return IntentType.STOCK_QUERY
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def enhance_query(question: str) -> EnhancedQuery:
    """
    Analyse the user question and return an EnhancedQuery.
    Falls back to a pass-through EnhancedQuery on any error.
    """
    q = (question or "").strip()
    if not q:
        return EnhancedQuery(original=q, rewritten=q)

    # Fast path: very short / simple question
    fast_intent = _fast_classify(q)
    if len(q) < 20 and fast_intent is None:
        return EnhancedQuery(original=q, rewritten=q, intent=IntentType.GENERAL)

    try:
        llm = get_llm()
        prompt = _ENHANCE_PROMPT.format(question=q)
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content if isinstance(response.content, str) else ""

        # Extract JSON from response (handle markdown fences)
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON found in response")

        data = json.loads(raw[start:end])
        intent_str = data.get("intent", "general")
        try:
            intent = IntentType(intent_str)
        except ValueError:
            intent = IntentType.GENERAL

        return EnhancedQuery(
            original=q,
            rewritten=data.get("rewritten") or q,
            intent=intent,
            sub_queries=data.get("sub_queries") or [],
            suggested_tools=data.get("suggested_tools") or [],
            keywords=data.get("keywords") or [],
        )

    except Exception:
        # Graceful fallback: keep original question, apply fast intent if known
        return EnhancedQuery(
            original=q,
            rewritten=q,
            intent=fast_intent or IntentType.GENERAL,
        )
