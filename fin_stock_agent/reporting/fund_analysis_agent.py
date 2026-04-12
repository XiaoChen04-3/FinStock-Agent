from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage

from fin_stock_agent.core.llm import get_llm, merge_token_usage

logger = logging.getLogger(__name__)


def _build_base_analysis(holding: dict, nav_rows: list[dict]) -> dict:
    row = holding
    latest = None
    first = None
    if nav_rows:
        sorted_rows = sorted(
            nav_rows,
            key=lambda item: str(item.get("nav_date") or item.get("trade_date") or ""),
        )
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
    if latest_nav is None or first_nav is None:
        trend = "insufficient_data"
        signal = "neutral"
        analysis = "近三年基金净值数据不足，建议结合基金持仓结构与当日新闻继续观察。"
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
        analysis = (
            f"近三年累计收益约 {(return_3y or 0.0):.1%}，"
            f"当前走势判断为{trend}，可与当日消息面和仓位盈亏结合判断是否继续持有。"
        )
    return {
        "name": row["name"],
        "metrics": {
            "return_3y": return_3y,
            "recent_3d_change": row.get("unrealized_pnl", 0.0),
        },
        "trend": trend,
        "analysis": analysis,
        "signal": signal,
    }


class FundAnalysisAgent:
    def __init__(self) -> None:
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def run(self, holdings: list[dict], nav_history: dict[str, list[dict]], recent_3_days: list[str]) -> dict:
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        analyses = {
            holding["ts_code"]: _build_base_analysis(holding, nav_history.get(holding["ts_code"], []))
            for holding in holdings
        }
        if not holdings:
            return {"analyses": analyses}

        try:
            llm = get_llm("fund_analysis")
            payload = [
                {
                    "ts_code": holding["ts_code"],
                    "name": holding["name"],
                    "quantity": holding.get("quantity"),
                    "avg_cost": holding.get("avg_cost"),
                    "market_value": holding.get("market_value"),
                    "unrealized_pnl": holding.get("unrealized_pnl"),
                    "base_metrics": analyses[holding["ts_code"]]["metrics"],
                    "nav_points": (nav_history.get(holding["ts_code"], []) or [])[-5:],
                }
                for holding in holdings
            ]
            prompt = (
                "You are a Chinese fund analysis agent.\n"
                "Review the portfolio holdings and return JSON only.\n"
                "JSON schema:\n"
                '{"analyses":{"ts_code":{"analysis":"short Chinese analysis","signal":"bullish|neutral|cautious","trend":"up|stable|down|insufficient_data"}}}\n\n'
                f"Recent trading days: {json.dumps(recent_3_days, ensure_ascii=False)}\n"
                f"Input holdings: {json.dumps(payload, ensure_ascii=False, default=str)}"
            )
            response = llm.invoke([HumanMessage(content=prompt)])
            self.last_usage = merge_token_usage(response)
            raw = response.content if isinstance(response.content, str) else ""
            raw = re.sub(r"```(?:json)?", "", raw).strip()
            start, end = raw.find("{"), raw.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(raw[start:end]).get("analyses") or {}
                for code, row in data.items():
                    if code in analyses:
                        analyses[code]["analysis"] = str(row.get("analysis") or analyses[code]["analysis"])
                        analyses[code]["signal"] = str(row.get("signal") or analyses[code]["signal"])
                        analyses[code]["trend"] = str(row.get("trend") or analyses[code]["trend"])
        except Exception as exc:
            logger.warning("FundAnalysisAgent LLM failed: %s", exc)

        return {"analyses": analyses}
