from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage

from fin_stock_agent.core.llm import get_llm, merge_token_usage
from fin_stock_agent.prompts.reporting_prompts import FUND_TREND_PROMPT
from fin_stock_agent.reporting.report_models import AgentResult, AgentTimer, ReportContext

logger = logging.getLogger(__name__)


def _build_base_analysis(holding: dict, nav_rows: list[dict]) -> dict:
    latest = None
    first = None
    if nav_rows:
        sorted_rows = sorted(nav_rows, key=lambda item: str(item.get("nav_date") or item.get("trade_date") or ""))
        first = sorted_rows[0]
        latest = sorted_rows[-1]
    latest_nav = None
    first_nav = None
    if latest is not None:
        latest_nav = float(latest.get("adj_nav") or latest.get("unit_nav") or latest.get("accum_nav") or 0.0)
    if first is not None:
        first_nav = float(first.get("adj_nav") or first.get("unit_nav") or first.get("accum_nav") or 0.0)
    return_3y = None
    if latest_nav is not None and first_nav and first_nav > 0:
        return_3y = latest_nav / first_nav - 1
    _TREND_LABEL = {"up": "上行", "down": "下行", "stable": "震荡", "insufficient_data": "数据不足"}
    if latest_nav is None or first_nav is None:
        trend = "insufficient_data"
        signal = "neutral"
        analysis = "净值历史数据不足，暂无法判断趋势。"
    else:
        if (return_3y or 0.0) >= 0.25:
            trend = "up"
            signal = "bullish"
        elif (return_3y or 0.0) <= -0.1:
            trend = "down"
            signal = "cautious"
        else:
            trend = "stable"
            signal = "neutral"
        analysis = f"近三年收益约 {(return_3y or 0.0):.1%}，中期趋势为{_TREND_LABEL.get(trend, trend)}。"
    return {
        "trend": trend,
        "signal": signal,
        "analysis": analysis,
        "metrics": {"return_3y": return_3y},
    }


class FundTrendAgent:
    def __init__(self) -> None:
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def run(self, ctx: ReportContext) -> AgentResult:
        timer = AgentTimer()
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        analyses = {
            holding["ts_code"]: _build_base_analysis(holding, ctx.raw_nav_history.get(holding["ts_code"], []))
            for holding in ctx.holdings
        }
        if not ctx.holdings:
            return AgentResult(
                agent_name="fund_trend",
                status="fallback",
                output={"analyses": analyses},
                elapsed_ms=timer.elapsed_ms(),
            )

        try:
            llm = get_llm("fund_analysis")
            payload = [
                {
                    "ts_code": holding["ts_code"],
                    "name": holding["name"],
                    "metrics": analyses[holding["ts_code"]]["metrics"],
                    "nav_points": (ctx.raw_nav_history.get(holding["ts_code"], []) or [])[-5:],
                }
                for holding in ctx.holdings
            ]
            prompt = FUND_TREND_PROMPT.format(
                payload=json.dumps(payload, ensure_ascii=False, default=str)
            )
            response = llm.invoke([HumanMessage(content=prompt)])
            self.last_usage = merge_token_usage(response)
            raw = response.content if isinstance(response.content, str) else ""
            raw = re.sub(r"```(?:json)?", "", raw).strip()
            start, end = raw.find("{"), raw.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(raw[start:end]).get("analyses") or {}
                for code, item in data.items():
                    if code in analyses:
                        analyses[code]["trend"] = str(item.get("trend") or analyses[code]["trend"])
                        analyses[code]["signal"] = str(item.get("signal") or analyses[code]["signal"])
                        analyses[code]["analysis"] = str(item.get("analysis") or analyses[code]["analysis"])
        except Exception as exc:
            logger.warning("FundTrendAgent LLM 调用失败，降级使用规则分析: %s", exc)

        return AgentResult(
            agent_name="fund_trend",
            status="success",
            output={"analyses": analyses},
            token_usage=dict(self.last_usage),
            elapsed_ms=timer.elapsed_ms(),
        )


FundAnalysisAgent = FundTrendAgent
