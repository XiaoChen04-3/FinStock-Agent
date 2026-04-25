from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import Lock

_TASK_TTL_HOURS = 6

from fin_stock_agent.core.config import get_config
from fin_stock_agent.stats.tracker import write_stats_event

_EXECUTOR = ThreadPoolExecutor(
    max_workers=get_config().concurrency.daily_report_workers,
    thread_name_prefix="daily-report",
)
_LOCK = Lock()
_TASKS: dict[str, "_TrackedTask"] = {}


@dataclass
class _TrackedTask:
    key: str
    future: Future | None
    requested_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None


@dataclass(frozen=True)
class ReportTaskSnapshot:
    state: str
    report_exists: bool
    requested_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None


def ensure_report_generation(user_id: str, report_date: str | None = None, *, force: bool = False) -> ReportTaskSnapshot:
    reporter = _new_reporter()
    resolved_date = reporter.resolve_report_date(report_date)
    existing = reporter.get_existing_report(user_id=user_id, date=resolved_date)
    key = _task_key(user_id, resolved_date)

    with _LOCK:
        _refresh_locked(key)
        task = _TASKS.get(key)
        if task is not None and task.future is not None and task.future.running():
            return _snapshot_from_task(task, report_exists=existing is not None)
        if task is not None and task.future is not None and not task.future.done():
            return _snapshot_from_task(task, report_exists=existing is not None)
        if task is not None and task.error and not force:
            return _snapshot_from_task(task, report_exists=existing is not None)
        if existing is not None and not force:
            return ReportTaskSnapshot(
                state="completed",
                report_exists=True,
                finished_at=existing.generated_at,
            )

        tracked = _TrackedTask(
            key=key,
            future=None,
            requested_at=datetime.now(),
        )
        _TASKS[key] = tracked
        tracked.future = _EXECUTOR.submit(_run_generate, user_id, resolved_date, force)
        write_stats_event(
            "daily_report_task_submitted",
            user_id=user_id,
            report_date=resolved_date,
            force=force,
        )
        return _snapshot_from_task(tracked, report_exists=existing is not None)


def get_report_task_snapshot(user_id: str, report_date: str | None = None) -> ReportTaskSnapshot:
    reporter = _new_reporter()
    resolved_date = reporter.resolve_report_date(report_date)
    existing = reporter.get_existing_report(user_id=user_id, date=resolved_date)
    key = _task_key(user_id, resolved_date)

    with _LOCK:
        _refresh_locked(key)
        task = _TASKS.get(key)
        if task is None:
            return ReportTaskSnapshot(
                state="completed" if existing is not None else "idle",
                report_exists=existing is not None,
                finished_at=existing.generated_at if existing is not None else None,
            )
        return _snapshot_from_task(task, report_exists=existing is not None)


def _run_generate(user_id: str, report_date: str, force: bool) -> None:
    key = _task_key(user_id, report_date)
    with _LOCK:
        task = _TASKS.get(key)
        if task is not None:
            task.started_at = datetime.now()
            task.error = None
    try:
        _new_reporter().generate(user_id=user_id, date=report_date, force=force)
        with _LOCK:
            task = _TASKS.get(key)
            if task is not None:
                task.finished_at = datetime.now()
                task.error = None
    except Exception as exc:
        with _LOCK:
            task = _TASKS.get(key)
            if task is not None:
                task.finished_at = datetime.now()
                task.error = str(exc)
        raise


def _evict_expired_locked() -> None:
    cutoff = datetime.now() - timedelta(hours=_TASK_TTL_HOURS)
    expired = [k for k, t in _TASKS.items() if t.finished_at is not None and t.finished_at < cutoff]
    for k in expired:
        del _TASKS[k]


def _refresh_locked(key: str) -> None:
    _evict_expired_locked()
    task = _TASKS.get(key)
    if task is None or task.finished_at is not None:
        return
    if task.future is not None and task.future.done():
        task.finished_at = datetime.now()
        exc = task.future.exception()
        task.error = str(exc) if exc else None


def _snapshot_from_task(task: _TrackedTask, *, report_exists: bool) -> ReportTaskSnapshot:
    if task.error:
        state = "failed"
    elif task.finished_at is not None:
        state = "completed"
    else:
        state = "running"
    return ReportTaskSnapshot(
        state=state,
        report_exists=report_exists,
        requested_at=task.requested_at,
        started_at=task.started_at,
        finished_at=task.finished_at,
        error=task.error,
    )


def _task_key(user_id: str, report_date: str) -> str:
    return f"{user_id}:{report_date}"


def _new_reporter():
    from fin_stock_agent.reporting.daily_reporter import DailyReporter

    return DailyReporter()
