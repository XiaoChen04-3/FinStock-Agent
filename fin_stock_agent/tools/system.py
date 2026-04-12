from __future__ import annotations

import json
from typing import Annotated

from langchain_core.tools import tool

from fin_stock_agent.init.name_resolver import NameResolver


@tool
def search_fund_or_index(keyword: Annotated[str, "fund or index keyword"]) -> str:
    """Search local fund and index lookup tables by fuzzy keyword."""
    items = NameResolver().search(keyword, top_k=10)
    if not items:
        return json.dumps({"ok": False, "error": f"No fund or index matched {keyword}"}, ensure_ascii=False)
    return json.dumps({"ok": True, "rows": len(items), "data": items}, ensure_ascii=False)


def get_system_tools() -> list:
    return [search_fund_or_index]
