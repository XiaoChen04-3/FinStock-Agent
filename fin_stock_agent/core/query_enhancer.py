from __future__ import annotations

import json
import re
from enum import Enum
from typing import Literal

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from fin_stock_agent.core.llm import get_llm
from fin_stock_agent.init.name_resolver import NameResolver


class IntentType(str, Enum):
    STOCK_INFO = "stock_info"
    STOCK_PRICE = "stock_price"
    STOCK_FUNDAMENTALS = "stock_fundamentals"
    TECHNICAL = "technical"
    FUND_INFO = "fund_info"
    FUND_NAV = "fund_nav"
    INDEX_PRICE = "index_price"
    GLOBAL_MARKET = "global_market"
    MACRO = "macro"
    PORTFOLIO_QUERY = "portfolio_query"
    PORTFOLIO_TRADE = "portfolio_trade"
    NEWS = "news"
    COMPREHENSIVE = "comprehensive"
    GENERAL_CHAT = "general_chat"


INTENT_LABELS = {
    IntentType.STOCK_INFO: "股票信息",
    IntentType.STOCK_PRICE: "股票行情",
    IntentType.STOCK_FUNDAMENTALS: "股票基本面",
    IntentType.TECHNICAL: "技术分析",
    IntentType.FUND_INFO: "基金信息",
    IntentType.FUND_NAV: "基金净值",
    IntentType.INDEX_PRICE: "指数行情",
    IntentType.GLOBAL_MARKET: "全球市场",
    IntentType.MACRO: "宏观数据",
    IntentType.PORTFOLIO_QUERY: "持仓查询",
    IntentType.PORTFOLIO_TRADE: "持仓交易",
    IntentType.NEWS: "新闻",
    IntentType.COMPREHENSIVE: "综合分析",
    IntentType.GENERAL_CHAT: "通用问答",
}


class EnhancedQuery(BaseModel):
    original: str
    rewritten: str
    intent: IntentType = IntentType.GENERAL_CHAT
    resolved_codes: dict[str, str] = Field(default_factory=dict)
    sub_queries: list[str] = Field(default_factory=list)
    suggested_tools: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    complexity: Literal["simple", "complex"] = "simple"

    def intent_label(self) -> str:
        return INTENT_LABELS.get(self.intent, self.intent.value)

    def to_context_block(self) -> str:
        lines = [
            "[query_enhancement]",
            f"intent: {self.intent.value}",
            f"rewritten: {self.rewritten}",
            f"complexity: {self.complexity}",
        ]
        if self.resolved_codes:
            lines.append(f"resolved_codes: {json.dumps(self.resolved_codes, ensure_ascii=False)}")
        if self.suggested_tools:
            lines.append("suggested_tools: " + ", ".join(self.suggested_tools))
        if self.sub_queries:
            lines.append("sub_queries:")
            for item in self.sub_queries:
                lines.append(f"- {item}")
        return "\n".join(lines)


PROMPT = """你是 FinStock-Agent 的查询意图分析助手，负责对用户的中文金融问题进行改写与结构化分类。

## 任务
分析用户问题，输出一个标准 JSON 对象。严禁输出 JSON 以外的任何内容（包括解释、注释、Markdown 代码块）。

## 输出 JSON 字段说明
- intent（字符串）：问题的主意图，必须从下方意图表中选择其中一个值。
- rewritten（字符串）：改写后的标准查询，补全缺省词语、消除歧义（如"贵州茅台"→"贵州茅台(600519.SH)"）；若无需改写则与原问题保持一致。
- resolved_codes（对象）："实体名称 → 证券代码"的映射，仅填写可从候选项中确认的映射；无法确认时返回空对象 {{}}。
- sub_queries（字符串数组）：对复杂问题的子问题拆解列表；简单问题返回空数组 []。
- suggested_tools（字符串数组）：建议优先调用的工具名称列表；不确定时返回空数组 []。
- keywords（字符串数组）：问题中的核心实体关键词（如股票名称、指标名、行业名）。
- complexity（字符串）：问题复杂度，必须是 "simple"（单一直接问题）或 "complex"（需多步骤或多工具分析）。

## 意图类型表
| intent 值             | 适用场景                                         |
|-----------------------|--------------------------------------------------|
| stock_info            | 股票基本信息（上市日期、行业分类、公司简介等）   |
| stock_price           | 股票实时行情或历史K线价格                        |
| stock_fundamentals    | 股票财务基本面（PE、PB、ROE、营收、净利润等）    |
| technical             | 技术指标分析（均线、MACD、RSI、布林带等）        |
| fund_info             | 基金基本信息（类型、规模、基金经理、费率等）     |
| fund_nav              | 基金净值查询（单位净值、累计净值、历史净值）     |
| index_price           | 指数行情（沪深300、上证指数、创业板指等）        |
| global_market         | 全球市场（美股、港股、原油、黄金、汇率等）       |
| macro                 | 宏观经济数据（CPI、PPI、GDP、M2、PMI等）         |
| portfolio_query       | 用户持仓查询、盈亏分析、持仓结构统计             |
| portfolio_trade       | 持仓操作（记录买入/卖出、更新持仓）              |
| news                  | 财经新闻、市场资讯、公告、研报摘要               |
| comprehensive         | 综合分析（同时涉及行情、基本面、新闻等多维度）   |
| general_chat          | 通用问答、投资知识科普或无法归类的问题           |

## 候选证券代码（本地解析结果）
{resolved_candidates}

## 用户问题
{question}
"""


def enhance_query(
    question: str,
    resolver: NameResolver | None = None,
    callbacks: list | None = None,
) -> EnhancedQuery:
    q = (question or "").strip()
    if not q:
        return EnhancedQuery(original="", rewritten="")
    resolver = resolver or NameResolver()
    candidates = resolver.search(q, top_k=5)
    resolved_candidates = {item["name"]: item["ts_code"] for item in candidates}
    try:
        llm = get_llm("query_enhancer")
        prompt = PROMPT.format(
            resolved_candidates=json.dumps(resolved_candidates, ensure_ascii=False),
            question=q,
        )
        cfg = {"callbacks": callbacks} if callbacks else {}
        response = llm.invoke([HumanMessage(content=prompt)], config=cfg)
        raw = response.content if isinstance(response.content, str) else ""
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end <= 0:
            raise ValueError("No JSON content")
        data = json.loads(raw[start:end])
        intent = IntentType(data.get("intent", IntentType.GENERAL_CHAT.value))
        return EnhancedQuery(
            original=q,
            rewritten=data.get("rewritten") or q,
            intent=intent,
            resolved_codes=data.get("resolved_codes") or resolved_candidates,
            sub_queries=data.get("sub_queries") or [],
            suggested_tools=data.get("suggested_tools") or [],
            keywords=data.get("keywords") or [item["name"] for item in candidates],
            complexity=data.get("complexity") or ("complex" if len(data.get("sub_queries") or []) > 1 else "simple"),
        )
    except Exception:
        fallback_intent = IntentType.GENERAL_CHAT
        if any(word in q for word in ("买", "卖", "持仓", "盈亏")):
            fallback_intent = IntentType.PORTFOLIO_QUERY
        elif any(word in q.lower() for word in ("cpi", "gdp", "m2")):
            fallback_intent = IntentType.MACRO
        elif "新闻" in q:
            fallback_intent = IntentType.NEWS
        return EnhancedQuery(
            original=q,
            rewritten=q,
            intent=fallback_intent,
            resolved_codes=resolved_candidates,
            keywords=[item["name"] for item in candidates],
            complexity="complex" if any(word in q for word in ("并", "对比", "分析")) else "simple",
        )
