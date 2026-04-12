from __future__ import annotations

import json
from typing import Annotated

from langchain_core.tools import tool

from fin_stock_agent.news.news_reader import NewsReader


@tool
def get_latest_news(keyword: Annotated[str, "optional keyword"] = "") -> str:
    """Return cached latest financial news, optionally filtered by a keyword."""
    reader = NewsReader()
    items = reader.get_cached_news(keywords=[keyword] if keyword else [], limit=20)
    if not items:
        return json.dumps(
            {"ok": False, "error": "No cached news available yet."},
            ensure_ascii=False,
        )
    return json.dumps({"ok": True, "rows": len(items), "data": items}, ensure_ascii=False, default=str)


def get_news_tools() -> list:
    return [get_latest_news]
