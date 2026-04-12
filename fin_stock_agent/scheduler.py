"""Background scheduler – trading-calendar refresh and news polling.

Jobs:
  refresh_trade_calendar   daily at 00:05
  refresh_news             every 30 minutes (fetches CLS / Eastmoney / THS)
"""
from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler

from fin_stock_agent.init.trade_calendar import TradingCalendar

logger = logging.getLogger(__name__)
_SCHEDULER = BackgroundScheduler()


def _refresh_news_job() -> None:
    """Scheduled job: pull latest news from all sources into the DB cache."""
    try:
        from fin_stock_agent.news.news_reader import NewsReader

        result = NewsReader().fetch_today_sync()
        if result.degraded:
            logger.warning("News refresh degraded: %s", result.message)
        else:
            logger.info(
                "News refresh OK: %d items from %s",
                len(result.items),
                result.fetched_sources,
            )
    except Exception as exc:
        logger.warning("News refresh job failed: %s", exc)


def ensure_scheduler_started() -> BackgroundScheduler:
    if _SCHEDULER.running:
        return _SCHEDULER

    calendar = TradingCalendar()

    _SCHEDULER.add_job(
        calendar.refresh,
        "cron",
        hour=0,
        minute=5,
        id="refresh_trade_calendar",
        replace_existing=True,
    )

    _SCHEDULER.add_job(
        _refresh_news_job,
        "interval",
        minutes=30,
        id="refresh_news",
        replace_existing=True,
    )

    _SCHEDULER.start()
    logger.info("Scheduler started (trade-calendar + news-refresh jobs registered).")
    return _SCHEDULER
