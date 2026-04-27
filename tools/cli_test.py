from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fin_stock_agent.agents import router
from fin_stock_agent.core.config import AppConfig
from fin_stock_agent.core.identity import local_profile_id
from fin_stock_agent.core.settings import settings
from fin_stock_agent.init.trade_calendar import TradingCalendar
from fin_stock_agent.memory.conversation import ConversationMemory
from fin_stock_agent.reporting.daily_reporter import DailyReporter
from fin_stock_agent.reporting.report_tasks import ensure_report_generation, get_report_task_snapshot
from fin_stock_agent.services.daily_report_digest_service import DailyReportDigestService
from fin_stock_agent.services.plan_library_service import PlanLibraryService
from fin_stock_agent.services.user_memory_service import UserMemoryService
from fin_stock_agent.storage.cache import get_cache
from fin_stock_agent.storage.database import get_session, init_db
from fin_stock_agent.storage.models import (
    ConversationSummaryORM,
    DailyReportDigestORM,
    PlanLibraryORM,
)

try:
    from rich.console import Console
    from rich.table import Table
except Exception:  # pragma: no cover
    Console = None
    Table = None


console = Console() if Console is not None else None


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    user_id = args.user_id or local_profile_id()
    try:
        AppConfig.load()
        init_db()
        _ensure_trade_calendar()
        return _dispatch(args, user_id)
    except RuntimeError as exc:
        _print_error(str(exc))
        return 1
    except KeyboardInterrupt:
        _print("已中断。")
        return 0
    except Exception as exc:
        _print_error(f"[ERROR] {type(exc).__name__}: {exc}")
        return 2


def _dispatch(args, user_id: str) -> int:
    if args.command in {"chat", "stream", "plan", "report"}:
        _require_openai_key()

    if args.command == "chat":
        answer = router.run_agent(args.question, user_id=user_id)
        _print(answer)
        return 0
    if args.command == "stream":
        for event_type, content in router.stream_agent(args.question, user_id=user_id):
            _render_stream_event(event_type, content)
        return 0
    if args.command == "plan":
        with _force_plan_mode():
            for event_type, content in router.stream_agent(args.question, user_id=user_id):
                _render_stream_event(event_type, content)
        return 0
    if args.command == "report":
        report_date = DailyReporter().resolve_report_date()
        ensure_report_generation(user_id, report_date, force=args.force)
        while True:
            snapshot = get_report_task_snapshot(user_id, report_date)
            if snapshot.state in {"completed", "failed"}:
                break
            _print(".", end="")
            time.sleep(1)
        _print("")
        if snapshot.state != "completed":
            raise RuntimeError(snapshot.error or "日报生成失败")
        report = DailyReporter().get_existing_report(user_id, report_date)
        if report is None:
            raise RuntimeError("日报任务已完成，但无法加载报告内容")
        _print(json.dumps({
            "report_date": report.report_date,
            "overall_summary": report.overall_summary,
            "news_sentiment_label": report.news_sentiment_label,
            "top_news": report.top_news[:3],
        }, ensure_ascii=False, indent=2))
        return 0
    if args.command == "memory":
        _show_memory(user_id)
        return 0
    if args.command == "plan-lib":
        _show_plan_library(user_id)
        return 0
    if args.command == "reset-memory":
        _reset_memory(user_id)
        _print(f"已重置用户 {user_id} 的记忆。")
        return 0
    if args.command == "flush-profile":
        changed = UserMemoryService().commit_pending_profile()
        _print(f"已提交 pending 用户画像：{changed}")
        return 0
    raise RuntimeError(f"未知命令：{args.command}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FinStock-Agent 本地 CLI 测试工具")
    parser.add_argument("--user-id", default="", help="目标用户 ID，默认使用本机规范用户。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("memory")
    subparsers.add_parser("plan-lib")
    subparsers.add_parser("reset-memory")
    subparsers.add_parser("flush-profile")

    parser_chat = subparsers.add_parser("chat")
    parser_chat.add_argument("question")
    parser_stream = subparsers.add_parser("stream")
    parser_stream.add_argument("question")
    parser_plan = subparsers.add_parser("plan")
    parser_plan.add_argument("question")

    parser_report = subparsers.add_parser("report")
    parser_report.add_argument("--force", action="store_true")
    return parser


def _ensure_trade_calendar() -> None:
    try:
        TradingCalendar().get_recent_trading_days(1)
    except Exception:
        pass


def _require_openai_key() -> None:
    if not settings.openai_api_key:
        raise RuntimeError("环境变量中未配置 OPENAI_API_KEY。")


def _render_stream_event(event_type: str, content: str) -> None:
    if console is None:
        prefix = {
            "thinking": "[thinking]",
            "tool_start": "[tool]",
            "tool_interaction": "[tool]",
            "answer": "[answer]",
            "error": "[error]",
            "mode": "[mode]",
        }.get(event_type, "")
        if event_type == "token":
            _print(content, end="")
        else:
            _print(f"{prefix} {content}".strip())
        return

    if event_type == "thinking":
        console.print(f"[dim]{content}[/dim]")
    elif event_type == "tool_start":
        console.print(f"[yellow]tool[/yellow] {content}")
    elif event_type == "tool_interaction":
        console.print(f"[blue]{content}[/blue]")
    elif event_type == "answer":
        console.print(f"[green]{content}[/green]")
    elif event_type == "error":
        console.print(f"[red]{content}[/red]")
    elif event_type == "mode":
        console.print(f"[cyan]{content}[/cyan]")
    elif event_type == "token":
        console.print(content, end="")
    else:
        console.print(content)


def _show_memory(user_id: str) -> None:
    service = UserMemoryService()
    service.initialize_runtime(user_id)
    profile = service.get_profile(user_id)
    profile_snapshot = service.snapshot()
    summaries = ConversationMemory(user_id=user_id, session_id="cli").get_recent_summaries(user_id, limit=5)
    digests = DailyReportDigestService().get_recent_digests(user_id, limit=5)
    if console is None or Table is None:
        payload = {
            "profile": profile.model_dump(mode="json"),
            "profile_file": {
                "path": profile_snapshot["path"],
                "pending_path": profile_snapshot["pending_path"],
                "token_estimate": profile_snapshot["token_estimate"],
                "pending_exists": profile_snapshot["pending_exists"],
                "active_text": profile_snapshot["active_text"],
            },
            "summaries": summaries,
            "digests": [row.report_date for row in digests],
        }
        _print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return

    table = Table(title=f"用户 {user_id} 的记忆快照")
    table.add_column("项目")
    table.add_column("内容")
    table.add_row("风险承受", profile.risk_level or "未知")
    table.add_row("投资期限", profile.investment_horizon or "未知")
    table.add_row("关注主题", "，".join(profile.focus_themes) or "-")
    table.add_row("自选标的", "，".join(profile.watchlist) or "-")
    table.add_row("画像文件", f"{profile_snapshot['path']}\npending：{profile_snapshot['pending_exists']}")
    table.add_row("画像内容", profile_snapshot["active_text"])
    table.add_row("对话摘要", "\n".join(summaries) or "-")
    table.add_row("日报摘要", "\n".join(f"{row.report_date}: {row.overall_summary or ''}" for row in digests) or "-")
    console.print(table)


def _show_plan_library(user_id: str) -> None:
    plans = PlanLibraryService().list_plans(user_id, limit=20)
    _print(json.dumps(plans, ensure_ascii=False, indent=2, default=str))


def _reset_memory(user_id: str) -> None:
    ConversationMemory(user_id=user_id, session_id="cli").clear_user(user_id)
    DailyReportDigestService().clear_user(user_id)
    PlanLibraryService().clear_user(user_id)
    UserMemoryService().reset_profile_file()
    with get_session() as session:
        for model in (ConversationSummaryORM, DailyReportDigestORM, PlanLibraryORM):
            rows = session.query(model).filter(model.user_id == user_id).all()
            for row in rows:
                session.delete(row)
    cache = get_cache()
    cache.delete(f"portfolio:{user_id}:context")


@contextmanager
def _force_plan_mode():
    original = router.classify_mode
    router.classify_mode = lambda enhanced: "plan_execute"
    try:
        yield
    finally:
        router.classify_mode = original


def _print(message: str, end: str = "\n") -> None:
    if console is not None:
        console.print(message, end=end)
    else:
        print(message, end=end)


def _print_error(message: str) -> None:
    if console is not None:
        console.print(f"[red]{message}[/red]")
    else:
        print(message, file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
