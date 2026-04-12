from __future__ import annotations

from fin_stock_agent.core.identity import local_profile_id
from fin_stock_agent.reporting.daily_reporter import DailyReporter
from fin_stock_agent.tools import portfolio as portfolio_tools


def test_daily_reporter_imports_without_package_side_effect_failure() -> None:
    reporter = DailyReporter()
    assert reporter is not None


def test_portfolio_tools_default_to_local_profile_id() -> None:
    portfolio_tools.set_tool_user_id("")
    assert portfolio_tools._ACTIVE_PROFILE_ID == local_profile_id()
