from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import Any

from fin_stock_agent.reporting.daily_reporter import DailyReporter
from fin_stock_agent.stats.tracker import write_stats_event
from fin_stock_agent.services.local_user_service import LocalUserService
from fin_stock_agent.services.portfolio_service import PortfolioService

_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="app-runtime")
_LOCK = Lock()
_PRELOAD_TASKS: dict[str, "_TrackedPreload"] = {}


@dataclass
class _TrackedPreload:
    key: str
    future: Future | None
    requested_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    payload: dict[str, Any] | None = None
    error: str | None = None


@dataclass(frozen=True)
class StartupPreloadSnapshot:
    state: str
    requested_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    payload: dict[str, Any] | None = None
    error: str | None = None


def get_local_user_id() -> str:
    return LocalUserService().canonical_user_id()


def consolidate_single_user_data() -> dict[str, int]:
    return LocalUserService().consolidate_legacy_data(get_local_user_id())


def ensure_startup_preload(user_id: str, report_date: str) -> StartupPreloadSnapshot:
    key = _task_key(user_id, report_date)
    with _LOCK:
        _refresh_locked(key)
        task = _PRELOAD_TASKS.get(key)
        if task is not None and task.future is not None and not task.future.done():
            return _snapshot_from_task(task)
        if task is not None and task.finished_at is not None:
            return _snapshot_from_task(task)

        tracked = _TrackedPreload(
            key=key,
            future=None,
            requested_at=datetime.now(),
        )
        _PRELOAD_TASKS[key] = tracked
        tracked.future = _EXECUTOR.submit(_run_preload, user_id, report_date)
        return _snapshot_from_task(tracked)


def get_startup_preload_snapshot(user_id: str, report_date: str) -> StartupPreloadSnapshot:
    key = _task_key(user_id, report_date)
    with _LOCK:
        _refresh_locked(key)
        task = _PRELOAD_TASKS.get(key)
        if task is None:
            return StartupPreloadSnapshot(state="idle")
        return _snapshot_from_task(task)


def _run_preload(user_id: str, report_date: str) -> dict[str, Any]:
    key = _task_key(user_id, report_date)
    with _LOCK:
        task = _PRELOAD_TASKS.get(key)
        if task is not None:
            task.started_at = datetime.now()
            task.error = None

    service = PortfolioService()
    reporter = DailyReporter()
    trades = service.get_trade_history(user_id, limit=500)
    holdings = service.get_holdings(user_id)
    report = reporter.get_existing_report(user_id=user_id, date=report_date)
    payload = {
        "trade_count": len(trades),
        "holding_count": len(holdings),
        "has_existing_report": report is not None,
        "report_generated_at": report.generated_at.isoformat() if report is not None else None,
    }

    write_stats_event(
        "startup_preload_completed",
        user_id=user_id,
        report_date=report_date,
        **payload,
    )

    with _LOCK:
        task = _PRELOAD_TASKS.get(key)
        if task is not None:
            task.finished_at = datetime.now()
            task.payload = payload
            task.error = None
    return payload


def _refresh_locked(key: str) -> None:
    task = _PRELOAD_TASKS.get(key)
    if task is None or task.finished_at is not None:
        return
    if task.future is not None and task.future.done():
        task.finished_at = datetime.now()
        try:
            task.payload = task.future.result()
            task.error = None
        except Exception as exc:
            task.error = str(exc)
            write_stats_event(
                "startup_preload_failed",
                user_id=key.split(":", 1)[0],
                report_date=key.split(":", 1)[1],
                has_error=True,
                error=str(exc),
            )


def _snapshot_from_task(task: _TrackedPreload) -> StartupPreloadSnapshot:
    if task.error:
        state = "failed"
    elif task.finished_at is not None:
        state = "completed"
    else:
        state = "running"
    return StartupPreloadSnapshot(
        state=state,
        requested_at=task.requested_at,
        started_at=task.started_at,
        finished_at=task.finished_at,
        payload=task.payload,
        error=task.error,
    )


def _task_key(user_id: str, report_date: str) -> str:
    return f"{user_id}:{report_date}"
