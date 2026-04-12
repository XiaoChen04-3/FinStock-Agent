from __future__ import annotations

__all__ = ["DailyReporter"]


def __getattr__(name: str):
    if name == "DailyReporter":
        from fin_stock_agent.reporting.daily_reporter import DailyReporter

        return DailyReporter
    raise AttributeError(name)
