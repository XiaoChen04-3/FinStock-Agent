from __future__ import annotations

import uuid
from datetime import datetime

import fin_stock_agent.app_runtime as runtime_module
from fin_stock_agent.core.settings import settings
from fin_stock_agent.news.models import NewsFetchResult
from fin_stock_agent.news.news_reader import NewsReader
from fin_stock_agent.reporting.daily_reporter import DailyReporter
from fin_stock_agent.reporting.models import DailyReport
from fin_stock_agent.services.local_user_service import LocalUserService
from fin_stock_agent.storage.database import get_session, init_db
from fin_stock_agent.storage.models import ConversationSummaryORM, DailyReportORM, NewsCacheORM, TradeRecordORM


def test_local_user_service_consolidates_legacy_rows() -> None:
    init_db()
    canonical_user_id = f"local-user-{uuid.uuid4()}"
    legacy_user_id = f"legacy-user-{uuid.uuid4()}"
    report_date = "2026-04-12"

    with get_session() as session:
        session.add(
            TradeRecordORM(
                user_id=legacy_user_id,
                ts_code="110022.OF",
                name="Persisted Fund",
                direction="buy",
                quantity=10.0,
                price=1.0,
                fee=0.0,
                trade_date="20260411",
            )
        )
        session.add(
            ConversationSummaryORM(
                user_id=legacy_user_id,
                session_id="legacy-session",
                turn_idx=1,
                question="问题",
                summary="摘要",
            )
        )
        session.add(
            DailyReportORM(
                user_id=legacy_user_id,
                report_date=report_date,
                report_json='{"user_id":"legacy","report_date":"2026-04-12"}',
                model_name="test",
                elapsed_ms=1.0,
            )
        )

    summary = LocalUserService().consolidate_legacy_data(canonical_user_id)

    assert summary["legacy_user_count"] >= 1
    assert summary["trade_rows_migrated"] >= 1
    assert summary["conversation_rows_migrated"] >= 1

    with get_session() as session:
        trade_codes = [row.ts_code for row in session.query(TradeRecordORM).filter(TradeRecordORM.user_id == canonical_user_id).all()]
        convo_sessions = [
            row.session_id for row in session.query(ConversationSummaryORM).filter(ConversationSummaryORM.user_id == canonical_user_id).all()
        ]
        report_dates = [
            row.report_date for row in session.query(DailyReportORM).filter(DailyReportORM.user_id == canonical_user_id).all()
        ]

    assert "110022.OF" in trade_codes
    assert "legacy-session" in convo_sessions
    assert sum(1 for value in report_dates if value == report_date) == 1


def test_news_reader_prunes_news_cache_to_recent_trading_window() -> None:
    init_db()
    reader = NewsReader()
    reader.trade_calendar.get_recent_trading_days = lambda n: ["20260411", "20260410", "20260409"]

    old_url = f"https://example.com/old/{uuid.uuid4()}"
    kept_url = f"https://example.com/keep/{uuid.uuid4()}"
    with get_session() as session:
        session.add(
            NewsCacheORM(
                url=old_url,
                title="old",
                summary="old",
                source="test",
                published_at=datetime(2026, 4, 8, 9, 0, 0),
            )
        )
        session.add(
            NewsCacheORM(
                url=kept_url,
                title="keep",
                summary="keep",
                source="test",
                published_at=datetime(2026, 4, 9, 9, 0, 0),
            )
        )

    removed = reader.prune_cache(retain_trading_days=3)

    assert removed >= 1
    with get_session() as session:
        urls = {row.url for row in session.query(NewsCacheORM).filter(NewsCacheORM.url.in_([old_url, kept_url])).all()}
    assert old_url not in urls
    assert kept_url in urls


def test_daily_reporter_logs_generation_event(monkeypatch, tmp_path) -> None:
    init_db()
    monkeypatch.setattr(settings, "log_dir", tmp_path)

    captured: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "fin_stock_agent.reporting.daily_reporter.write_stats_event",
        lambda event_type, **payload: captured.append((event_type, payload)),
    )

    reporter = DailyReporter()
    reporter.cache = type("Cache", (), {"get": lambda self, key: None, "setex": lambda self, key, ttl, payload: None})()
    reporter.portfolio_service = type(
        "PortfolioSvc",
        (),
        {"get_holdings": lambda self, user_id: [{"ts_code": "110022.OF", "name": "测试基金", "quantity": 10.0, "avg_cost": 1.2, "last_price": 1.3, "market_value": 13.0, "unrealized_pnl": 1.0}]},
    )()
    reporter.name_resolver = type("Resolver", (), {"get_keywords_for_holdings": lambda self, codes: ["测试基金"]})()
    reporter.trade_calendar = type("Calendar", (), {"get_recent_trading_days": lambda self, n: ["20260411", "20260410", "20260409"]})()
    reporter.news_reader = type(
        "Reader",
        (),
        {"fetch_today_sync": lambda self: NewsFetchResult(items=[], fetched_sources=["cls"], degraded=False, message="")},
    )()
    reporter.briefing_agent = type(
        "Briefing",
        (),
        {"last_usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}, "run": lambda self, items: {"top_10": []}},
    )()
    reporter.news_agent = type(
        "NewsAgent",
        (),
        {
            "last_usage": {"prompt_tokens": 13, "completion_tokens": 5, "total_tokens": 18},
            "run": lambda self, items, keywords, daily_briefing=None: {"daily_briefing_top10": []},
        },
    )()
    reporter.market_opportunity_agent = type(
        "MarketIdeas",
        (),
        {"last_usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}, "run": lambda self, briefing, limit=5: {"ideas": []}},
    )()
    reporter.fund_fetcher = type("Fetcher", (), {"fetch_history": lambda self, codes, years=3: {}})()
    reporter.fund_agent = type(
        "FundAgent",
        (),
        {"last_usage": {"prompt_tokens": 17, "completion_tokens": 9, "total_tokens": 26}, "run": lambda self, holdings, nav_history, recent_days: {"analyses": {}}},
    )()
    reporter.agentic_analyzer = type(
        "Analyzer",
        (),
        {
            "last_usage": {"prompt_tokens": 19, "completion_tokens": 11, "total_tokens": 30},
            "analyze": lambda self, holdings, items, daily_briefing=None, name_keywords=None: {},
        },
    )()
    reporter.synthesis_agent = type(
        "Synthesis",
        (),
        {
            "last_usage": {"prompt_tokens": 23, "completion_tokens": 13, "total_tokens": 36},
            "run": lambda self, **kwargs: DailyReport(
                user_id=kwargs["user_id"],
                report_date=kwargs["report_date"],
                generated_at=datetime(2026, 4, 12, 9, 30, 0),
                recent_trading_days=kwargs["recent_trading_days"],
                total_market_value=13.0,
                total_unrealized_pnl=1.0,
                total_unrealized_pnl_pct=0.08,
                today_portfolio_change_pct=0.0,
                fund_statuses=[],
                overall_summary="summary",
                market_context="context",
                news_sentiment_label="neutral",
                top_news=[],
                market_fund_ideas=[],
                total_elapsed_ms=12.0,
            )
        },
    )()
    reporter._save_report = lambda **kwargs: None

    report = reporter.generate(user_id="local-user", date="2026-04-12", force=True)

    assert report.report_date == "2026-04-12"
    assert captured
    event_type, payload = captured[-1]
    assert event_type == "daily_report_generated"
    assert payload["user_id"] == "local-user"
    assert payload["holdings_count"] == 1
    assert payload["news_count"] == 0
    assert payload["stage1_tokens"] == 41
    assert payload["stage2_tokens"] == 56
    assert payload["stage3_tokens"] == 36
    assert payload["total_tokens"] == 133


def test_startup_preload_logs_report_date_once(monkeypatch) -> None:
    runtime_module._PRELOAD_TASKS.clear()
    key = "local-user:2026-04-12"
    runtime_module._PRELOAD_TASKS[key] = runtime_module._TrackedPreload(
        key=key,
        future=None,
        requested_at=datetime(2026, 4, 12, 9, 0, 0),
    )

    captured: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        runtime_module,
        "write_stats_event",
        lambda event_type, **payload: captured.append((event_type, payload)),
    )

    monkeypatch.setattr(
        runtime_module,
        "PortfolioService",
        lambda: type(
            "PortfolioSvc",
            (),
            {
                "get_trade_history": lambda self, user_id, limit=500: [object(), object()],
                "get_holdings": lambda self, user_id: [{"ts_code": "110022.OF"}],
            },
        )(),
    )
    monkeypatch.setattr(
        runtime_module,
        "DailyReporter",
        lambda: type(
            "Reporter",
            (),
            {
                "get_existing_report": lambda self, user_id, date=None: type(
                    "Report",
                    (),
                    {"generated_at": datetime(2026, 4, 12, 9, 30, 0)},
                )(),
            },
        )(),
    )

    payload = runtime_module._run_preload("local-user", "2026-04-12")

    assert payload["holding_count"] == 1
    assert payload["trade_count"] == 2
    assert captured
    event_type, event_payload = captured[-1]
    assert event_type == "startup_preload_completed"
    assert event_payload["report_date"] == "2026-04-12"
