from __future__ import annotations

from fin_stock_agent.reporting.report_models import AgentResult, AgentTimer, ReportContext


class HoldingCorrelationAgent:
    def run(
        self,
        ctx: ReportContext,
        sentiment_result: AgentResult,
        fund_trend_result: AgentResult | None,
        news_filter_result: AgentResult | None = None,
    ) -> AgentResult:
        timer = AgentTimer()
        if not ctx.holdings:
            return AgentResult(
                agent_name="holding_correlation",
                status="fallback",
                output={"recommendations": {}},
                elapsed_ms=timer.elapsed_ms(),
            )

        sentiment = sentiment_result.output if sentiment_result else {}
        fund_analyses = (fund_trend_result.output if fund_trend_result else {}).get("analyses", {})
        related_news = []
        if news_filter_result is not None:
            related_news = news_filter_result.output.get("personalized_news") or news_filter_result.output.get("top_news") or []

        recommendations: dict[str, dict] = {}
        sentiment_label = str(sentiment.get("sentiment_label") or "neutral")
        for holding in ctx.holdings:
            code = holding["ts_code"]
            analysis = fund_analyses.get(code, {})
            signal = str(analysis.get("signal") or "neutral")
            action, confidence = self._decide(signal, sentiment_label)
            recommendations[code] = {
                "action": action,
                "confidence": confidence,
                "reasoning": self._reasoning(holding["name"], signal, sentiment_label, action),
                "relevant_titles": [item.get("title", "") for item in related_news[:3] if item.get("title")],
            }

        return AgentResult(
            agent_name="holding_correlation",
            status="success",
            output={"recommendations": recommendations},
            elapsed_ms=timer.elapsed_ms(),
        )

    @staticmethod
    def _decide(signal: str, sentiment_label: str) -> tuple[str, float]:
        if signal == "bullish" and sentiment_label in {"bullish", "watch"}:
            return "buy", 0.72
        if signal == "cautious" or sentiment_label in {"cautious", "bearish"}:
            return "sell", 0.64
        return "hold", 0.55

    @staticmethod
    def _reasoning(name: str, signal: str, sentiment_label: str, action: str) -> str:
        _signal_cn = {"bullish": "偏多", "neutral": "中性", "cautious": "偏空", "insufficient_data": "数据不足"}
        _sentiment_cn = {"bullish": "偏多", "watch": "关注", "neutral": "中性", "cautious": "谨慎", "bearish": "偏空"}
        _action_cn = {"buy": "加仓", "hold": "持有", "sell": "卖出"}
        return (
            f"{name} 基金信号为{_signal_cn.get(signal, signal)}，"
            f"市场情绪为{_sentiment_cn.get(sentiment_label, sentiment_label)}，"
            f"综合建议：{_action_cn.get(action, action)}。"
        )


AgenticNewsAnalyzer = HoldingCorrelationAgent
