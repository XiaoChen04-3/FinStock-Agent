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


PROMPT = """You are rewriting and classifying a financial user query.

Return one JSON object with these fields:
- intent
- rewritten
- resolved_codes
- sub_queries
- suggested_tools
- keywords
- complexity

Allowed intents:
stock_info, stock_price, stock_fundamentals, technical, fund_info, fund_nav,
index_price, global_market, macro, portfolio_query, portfolio_trade, news,
comprehensive, general_chat

Complexity must be simple or complex.

Resolved code candidates from local resolver:
{resolved_candidates}

Question:
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
