from __future__ import annotations

import json
from typing import Annotated

from langchain_core.tools import tool

from fin_stock_agent.core.exceptions import TushareRequestError
from fin_stock_agent.utils.tushare_client import get_client


def _payload(ok: bool, **kwargs) -> str:
    return json.dumps({"ok": ok, **kwargs}, ensure_ascii=False, default=str)


def _fetch_macro(method: str, limit: int = 24) -> str:
    try:
        df = get_client().call(method)
        if df is None or df.empty:
            return _payload(False, error=f"No data returned from {method}")
        return _payload(True, rows=len(df), data=df.head(limit).to_dict(orient="records"))
    except TushareRequestError as exc:
        return _payload(False, error=str(exc))


@tool
def get_cpi(period: Annotated[str, "optional period, unused for now"] = "") -> str:
    """Fetch recent China CPI data from Tushare."""
    return _fetch_macro("cn_cpi")


@tool
def get_m2(period: Annotated[str, "optional period, unused for now"] = "") -> str:
    """Fetch recent China broad money supply data from Tushare."""
    return _fetch_macro("cn_m")


@tool
def get_gdp(period: Annotated[str, "optional period, unused for now"] = "") -> str:
    """Fetch recent China GDP data from Tushare."""
    return _fetch_macro("cn_gdp")


def get_macro_tools() -> list:
    return [get_cpi, get_m2, get_gdp]
