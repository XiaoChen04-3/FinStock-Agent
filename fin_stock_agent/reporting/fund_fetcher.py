from __future__ import annotations

import logging
from datetime import datetime, timedelta

from fin_stock_agent.utils.pnl_calculator import _latest_fund_nav
from fin_stock_agent.utils.tushare_client import get_client

logger = logging.getLogger(__name__)


class TushareFundFetcher:
    def fetch_history(self, ts_codes: list[str], years: int = 3) -> dict[str, list[dict]]:
        """Pull ``fund_nav`` history for each fund code (3y window).

        Normalises codes to upper-case; skips blanks.  If the first window
        returns empty, retries with an extra year of look-back once (helps
        thinly-traded or API edge cases).
        """
        client = get_client()
        end = datetime.now().strftime("%Y%m%d")
        base_days = 365 * years + 45

        history: dict[str, list[dict]] = {}
        for raw in ts_codes:
            code = (raw or "").strip().upper()
            if not code:
                continue
            if not code.endswith(".OF"):
                logger.debug("fetch_history: %s is not *.OF, skipping fund_nav bulk fetch", code)
                history[code] = []
                continue

            rows: list[dict] = []
            for extra in (0, 365):
                start = (datetime.now() - timedelta(days=base_days + extra)).strftime("%Y%m%d")
                try:
                    df = client.call("fund_nav", ts_code=code, start_date=start, end_date=end)
                except Exception as exc:
                    logger.warning("fund_nav failed for %s: %s", code, exc)
                    df = None
                if df is not None and not df.empty:
                    rows = df.to_dict(orient="records")
                    break
            history[code] = rows
            if not rows:
                logger.warning(
                    "No fund_nav rows for %s in %dy window — check Tushare权限/积分 or code.",
                    code,
                    years,
                )
        return history

    def fetch_unit_nav_on_or_before(self, ts_code: str, trade_date: str) -> tuple[float | None, str | None]:
        """Unit (or adj) NAV on *trade_date* or the latest prior published date.

        *trade_date* is ``YYYYMMDD`` (with or without dashes).  Looks back up to
        40 calendar days for funds that skip publishing on some days.
        """
        code = (ts_code or "").strip().upper()
        ymd = trade_date.replace("-", "")[:8]
        if len(ymd) != 8 or not ymd.isdigit():
            return None, None
        client = get_client()
        end = ymd
        start = (datetime.strptime(ymd, "%Y%m%d") - timedelta(days=40)).strftime("%Y%m%d")
        try:
            df = client.call("fund_nav", ts_code=code, start_date=start, end_date=end)
        except Exception as exc:
            logger.warning("fund_nav lookup failed for %s @ %s: %s", code, ymd, exc)
            return None, None
        if df is None or df.empty:
            return None, None
        sort_col = "nav_date" if "nav_date" in df.columns else "ann_date"
        if sort_col not in df.columns:
            sort_col = df.columns[0]
        df = df.sort_values(sort_col)
        row = df.iloc[-1]
        nav = row.get("adj_nav")
        if nav is None or (isinstance(nav, float) and nav != nav):
            nav = row.get("unit_nav")
        if nav is None or (isinstance(nav, float) and nav != nav):
            nav = row.get("accum_nav")
        if nav is None:
            return None, None
        used = str(row.get("nav_date") or row.get("ann_date") or "")
        return float(nav), used.replace("-", "")[:8] if used else None
