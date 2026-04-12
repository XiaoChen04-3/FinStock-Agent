from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from typing import Any

from langchain_core.messages import HumanMessage

from fin_stock_agent.core.llm import get_llm, merge_token_usage
from fin_stock_agent.init.name_resolver import NameResolver

logger = logging.getLogger(__name__)


class MarketOpportunityAgent:
    def __init__(self) -> None:
        self.name_resolver = NameResolver()
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def run(self, daily_briefing: dict[str, Any] | None = None, *, limit: int = 5) -> dict[str, list[dict]]:
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        briefing = daily_briefing or {}
        top_10 = list(briefing.get("top_10") or [])
        candidates = self._build_candidates(top_10)
        if not candidates:
            return {"ideas": []}

        ideas = self._rank_candidates(top_10, candidates, limit=limit)
        if not ideas:
            ideas = self._fallback(candidates, limit=limit)
        return {"ideas": ideas[:limit]}

    def _build_candidates(self, top_10: list[dict]) -> list[dict]:
        grouped: dict[str, dict[str, Any]] = defaultdict(dict)
        for row in top_10[:10]:
            title = str(row.get("title") or "").strip()
            reason = str(row.get("reason") or "").strip()
            impact = int(row.get("impact") or 3)
            if not title:
                continue

            matches = self._search_matches(title)
            if not matches and reason:
                matches = self._search_matches(reason)

            for match in matches[:3]:
                code = str(match.get("ts_code") or "").strip().upper()
                if not code:
                    continue

                item = grouped.setdefault(
                    code,
                    {
                        "ts_code": code,
                        "fund_name": str(match.get("name") or code),
                        "theme": str(match.get("name") or code),
                        "score": 0.0,
                        "related_news": [],
                        "reason_fragments": [],
                    },
                )
                item["score"] += float(match.get("score") or 0.0) + impact * 12
                if title and title not in item["related_news"]:
                    item["related_news"].append(title)
                if reason and reason not in item["reason_fragments"]:
                    item["reason_fragments"].append(reason)
        return sorted(grouped.values(), key=lambda row: row["score"], reverse=True)

    def _search_matches(self, text: str) -> list[dict]:
        results = self.name_resolver.search_funds(text, top_k=5)
        return [row for row in results if float(row.get("score") or 0.0) >= 8][:3]

    def _rank_candidates(self, top_10: list[dict], candidates: list[dict], *, limit: int) -> list[dict]:
        try:
            llm = get_llm("report_synthesis")
            prompt = (
                "你是一位中国公募基金研究助理。"
                "请根据当日市场影响最大的新闻，以及已筛出的候选基金，选出最值得关注的基金。"
                "只能从候选基金里挑选，不要虚构代码。"
                "只输出 JSON。\n"
                'JSON schema: {"ideas":[{"ts_code":"基金代码","action":"buy|watch","confidence":0.0,'
                '"theme":"主题","reason":"不超过60字的中文理由","related_news":["标题1","标题2"]}]}\n\n'
                f"Top 10 news: {json.dumps(top_10[:10], ensure_ascii=False)}\n"
                f"Candidates: {json.dumps(candidates[:12], ensure_ascii=False)}"
            )
            response = llm.invoke([HumanMessage(content=prompt)])
            self.last_usage = merge_token_usage(response)
            raw = response.content if isinstance(response.content, str) else ""
            raw = re.sub(r"```(?:json)?", "", raw).strip()
            start, end = raw.find("{"), raw.rfind("}") + 1
            if start < 0 or end <= start:
                return []

            payload = json.loads(raw[start:end])
            rows = payload.get("ideas") or []
            candidate_map = {row["ts_code"]: row for row in candidates}
            ranked: list[dict] = []
            for row in rows:
                code = str(row.get("ts_code") or "").strip().upper()
                base = candidate_map.get(code)
                if base is None:
                    continue
                ranked.append(
                    {
                        "theme": str(row.get("theme") or base["theme"]),
                        "fund_name": base["fund_name"],
                        "ts_code": code,
                        "action": str(row.get("action") or "watch"),
                        "confidence": float(row.get("confidence") or 0.6),
                        "reason": str(row.get("reason") or "")[:120] or self._fallback_reason(base),
                        "related_news": list(row.get("related_news") or base["related_news"][:3]),
                    }
                )
            return ranked[:limit]
        except Exception as exc:
            logger.warning("MarketOpportunityAgent LLM failed: %s", exc)
            return []

    def _fallback(self, candidates: list[dict], *, limit: int) -> list[dict]:
        ideas: list[dict] = []
        for row in candidates[:limit]:
            ideas.append(
                {
                    "theme": row["theme"],
                    "fund_name": row["fund_name"],
                    "ts_code": row["ts_code"],
                    "action": "watch",
                    "confidence": min(0.88, 0.45 + row["score"] / 160),
                    "reason": self._fallback_reason(row),
                    "related_news": row["related_news"][:3],
                }
            )
        return ideas

    def _fallback_reason(self, candidate: dict[str, Any]) -> str:
        news_title = next(iter(candidate.get("related_news") or []), "")
        if news_title:
            return f"受“{news_title}”等消息带动，该主题基金值得优先跟踪。"
        return "该基金与今日高影响主题较匹配，建议列入优先观察名单。"
