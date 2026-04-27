from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from fin_stock_agent.app_runtime import (
    StartupPreloadSnapshot,
    consolidate_single_user_data,
    ensure_startup_preload,
    get_startup_preload_snapshot,
)
from fin_stock_agent.init.system_init import SystemInit
from fin_stock_agent.reporting.daily_reporter import DailyReporter
from fin_stock_agent.reporting.report_tasks import ensure_report_generation
from fin_stock_agent.scheduler import ensure_scheduler_started
from fin_stock_agent.services.user_memory_service import UserMemoryService
from fin_stock_agent.stats.tracker import write_stats_event


@dataclass(frozen=True)
class AppBootstrapSnapshot:
    ready: bool
    report_date: str | None = None
    preload_snapshot: StartupPreloadSnapshot | None = None


def ensure_app_bootstrap(user_id: str, session_id: str) -> AppBootstrapSnapshot:
    if not SystemInit().check_and_setup():
        return AppBootstrapSnapshot(ready=False)

    if "single_user_bootstrapped" not in st.session_state:
        st.session_state["single_user_bootstrapped"] = True
        consolidate_single_user_data()
        UserMemoryService().initialize_runtime(user_id)

    if "scheduler_started" not in st.session_state:
        ensure_scheduler_started()
        st.session_state["scheduler_started"] = True

    if "app_session_logged" not in st.session_state:
        st.session_state["app_session_logged"] = True
        write_stats_event("app_session_started", session_id=session_id, user_id=user_id)

    report_date = DailyReporter().resolve_report_date()
    ensure_startup_preload(user_id, report_date)
    preload_snapshot = get_startup_preload_snapshot(user_id, report_date)
    ensure_report_generation(user_id=user_id, report_date=report_date, force=False)
    return AppBootstrapSnapshot(
        ready=True,
        report_date=report_date,
        preload_snapshot=preload_snapshot,
    )
