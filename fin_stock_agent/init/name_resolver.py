from __future__ import annotations

import logging
import re
import warnings
from collections import Counter

from sqlalchemy import select

from fin_stock_agent.storage.database import get_session
from fin_stock_agent.storage.models import FundLookupRecord, IndexLookupRecord

try:
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API.*",
        category=UserWarning,
        module=r"jieba\._compat",
    )
    import jieba
    jieba.setLogLevel(logging.ERROR)
except Exception:  # pragma: no cover
    jieba = None


def _tokenize(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    if jieba is not None:
        return [token.strip() for token in jieba.cut(raw) if token.strip()]
    return [token for token in re.split(r"[\s_\-/]+", raw) if token]


class NameResolver:
    def resolve_fund(self, text: str) -> str | None:
        return self._resolve(text, kind="fund")

    def resolve_index(self, text: str) -> str | None:
        return self._resolve(text, kind="index")

    def search(self, text: str, top_k: int = 5) -> list[dict]:
        return self._search_records(text, top_k=top_k)

    def search_funds(self, text: str, top_k: int = 8) -> list[dict]:
        """Like ``search`` but returns only ``fund_lookup`` matches (no indices)."""
        scored = [r for r in self._search_records(text, top_k=top_k * 3) if r.get("kind") == "fund"]
        return scored[:top_k]

    def get_keywords_for_holdings(self, ts_codes: list[str]) -> list[str]:
        keywords: list[str] = []
        normalized = sorted({code.strip().upper() for code in ts_codes if code})
        if not normalized:
            return []
        with get_session() as session:
            for model in (FundLookupRecord, IndexLookupRecord):
                rows = session.execute(select(model).where(model.ts_code.in_(normalized))).scalars()
                for row in rows:
                    keywords.extend(_tokenize(row.name))
                    keywords.append(row.name)
        counts = Counter(item for item in keywords if item)
        return [item for item, _ in counts.most_common(12)]

    def build_prompt_mapping(self, names: list[str]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for name in names:
            code = self.resolve_fund(name) or self.resolve_index(name)
            if code:
                mapping[name] = code
        return mapping

    def _resolve(self, text: str, kind: str) -> str | None:
        matches = self._search_records(text, top_k=5)
        for item in matches:
            if item["kind"] == kind:
                return item["ts_code"]
        return None

    def _search_records(self, text: str, top_k: int = 5) -> list[dict]:
        raw = (text or "").strip()
        if not raw:
            return []
        tokens = _tokenize(raw)
        scored: list[dict] = []
        with get_session() as session:
            for model, kind in ((FundLookupRecord, "fund"), (IndexLookupRecord, "index")):
                rows = session.execute(select(model)).scalars()
                for row in rows:
                    name = row.name or ""
                    code = row.ts_code or ""
                    score = 0
                    if raw == name or raw.upper() == code.upper():
                        score += 100
                    if name.startswith(raw):
                        score += 50
                    if raw in name:
                        score += 40
                    if raw and all(char in name for char in raw):
                        score += 20
                    score += sum(8 for token in tokens if token and token in name)
                    score += sum(2 for token in tokens if token and token in code)
                    if score <= 0:
                        continue
                    scored.append({"ts_code": code, "name": name, "kind": kind, "score": score})
        scored.sort(key=lambda item: (-item["score"], item["name"]))
        return scored[:top_k]
