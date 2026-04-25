from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage

from fin_stock_agent.core.llm import get_llm, merge_token_usage
from fin_stock_agent.prompts.reporting_prompts import SENTIMENT_ANALYSIS_PROMPT
from fin_stock_agent.reporting.report_models import AgentResult, AgentTimer, ReportContext

logger = logging.getLogger(__name__)

_POSITIVE_KEYWORDS = [
    "上涨", "涨停", "创新高", "利好", "政策支持", "反弹", "增长", "突破",
    "回升", "走强", "跑赢", "超预期", "加仓", "买入", "净流入", "扩张",
]
_NEGATIVE_KEYWORDS = [
    "下跌", "跌停", "利空", "减持", "抛售", "风险", "波动", "压力",
    "走弱", "跑输", "不及预期", "卖出", "净流出", "收缩", "违约", "亏损",
]
_RISK_KEYWORDS = ["下跌", "风险", "波动", "减持", "抛售", "违约", "亏损", "跌停"]


class SentimentAnalysisAgent:
    def __init__(self) -> None:
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def run(self, ctx: ReportContext, news_filter_result: AgentResult) -> AgentResult:
        timer = AgentTimer()
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        news_items = news_filter_result.output.get("top_news") or []
        if not news_items:
            return AgentResult(
                agent_name="sentiment_analysis",
                status="fallback",
                output={
                    "sentiment_score": 0.5,
                    "sentiment_label": "neutral",
                    "market_summary": "暂无可用新闻，市场情绪维持中性。",
                    "risk_signals": [],
                },
                elapsed_ms=timer.elapsed_ms(),
            )

        fallback = self._rule_based(news_items)
        try:
            llm = get_llm("news_analysis")
            prompt = SENTIMENT_ANALYSIS_PROMPT.format(
                news_items=json.dumps(news_items, ensure_ascii=False)
            )
            response = llm.invoke([HumanMessage(content=prompt)])
            self.last_usage = merge_token_usage(response)
            raw = response.content if isinstance(response.content, str) else ""
            raw = re.sub(r"```(?:json)?", "", raw).strip()
            start, end = raw.find("{"), raw.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(raw[start:end])
                fallback.update(
                    {
                        "sentiment_score": float(data.get("sentiment_score", fallback["sentiment_score"])),
                        "sentiment_label": str(data.get("sentiment_label") or fallback["sentiment_label"]),
                        "market_summary": str(data.get("market_summary") or fallback["market_summary"]),
                        "risk_signals": list(data.get("risk_signals") or fallback["risk_signals"]),
                    }
                )
        except Exception as exc:
            logger.warning("SentimentAnalysisAgent LLM 调用失败，降级使用规则判断: %s", exc)

        return AgentResult(
            agent_name="sentiment_analysis",
            status="success",
            output=fallback,
            token_usage=dict(self.last_usage),
            elapsed_ms=timer.elapsed_ms(),
        )

    @staticmethod
    def _rule_based(news_items: list[dict]) -> dict:
        blob = " ".join(str(item.get("title", "")) for item in news_items)
        positive_hits = sum(1 for word in _POSITIVE_KEYWORDS if word in blob)
        negative_hits = sum(1 for word in _NEGATIVE_KEYWORDS if word in blob)
        score = 0.5 + 0.08 * positive_hits - 0.08 * negative_hits
        score = max(0.0, min(1.0, score))
        if score >= 0.75:
            label = "bullish"
        elif score >= 0.6:
            label = "watch"
        elif score <= 0.25:
            label = "bearish"
        elif score <= 0.4:
            label = "cautious"
        else:
            label = "neutral"
        risk_signals = [
            item["title"]
            for item in news_items
            if any(k in item.get("title", "") for k in _RISK_KEYWORDS)
        ]
        return {
            "sentiment_score": round(score, 4),
            "sentiment_label": label,
            "market_summary": "市场情绪基于当日精选新闻综合判断。",
            "risk_signals": risk_signals[:3],
        }


NewsAnalysisAgent = SentimentAnalysisAgent
