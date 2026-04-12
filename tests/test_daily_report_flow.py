from __future__ import annotations

import time
from datetime import datetime
from types import SimpleNamespace

import fin_stock_agent.reporting.report_synthesis_agent as synthesis_module
import fin_stock_agent.reporting.report_tasks as task_module
from fin_stock_agent.reporting.report_synthesis_agent import ReportSynthesisAgent


def test_report_tasks_deduplicate_background_generation(monkeypatch) -> None:
    generated: dict[tuple[str, str], object] = {}
    calls: list[tuple[str, str | None, bool]] = []
    task_module._TASKS.clear()

    class FakeReporter:
        def resolve_report_date(self, date: str | None = None) -> str:
            return date or "2026-04-12"

        def get_existing_report(self, user_id: str, date: str | None = None):
            return generated.get((user_id, self.resolve_report_date(date)))

        def generate(self, user_id: str, date: str | None = None, force: bool = False):
            calls.append((user_id, date, force))
            time.sleep(0.05)
            report = SimpleNamespace(generated_at=datetime(2026, 4, 12, 10, 5, 0))
            generated[(user_id, self.resolve_report_date(date))] = report
            return report

    monkeypatch.setattr(task_module, "_new_reporter", lambda: FakeReporter())

    first = task_module.ensure_report_generation("user-1", "2026-04-12")
    second = task_module.ensure_report_generation("user-1", "2026-04-12")

    assert first.state == "running"
    assert second.state == "running"

    deadline = time.time() + 2
    snapshot = second
    while time.time() < deadline:
        snapshot = task_module.get_report_task_snapshot("user-1", "2026-04-12")
        if snapshot.state == "completed":
            break
        time.sleep(0.02)

    assert snapshot.state == "completed"
    assert len(calls) == 1

    third = task_module.ensure_report_generation("user-1", "2026-04-12")
    assert third.state == "completed"
    assert len(calls) == 1


def test_report_synthesis_keeps_holding_analysis_without_market_ideas(monkeypatch) -> None:
    def _raise(_role: str):
        raise RuntimeError("llm unavailable")

    monkeypatch.setattr(synthesis_module, "get_llm", _raise)

    report = ReportSynthesisAgent().run(
        user_id="user-1",
        report_date="2026-04-12",
        recent_trading_days=["20260410", "20260411"],
        holdings=[
            {
                "ts_code": "000001.OF",
                "name": "黄金主题基金",
                "quantity": 1000.0,
                "avg_cost": 1.2,
                "last_price": 1.36,
                "market_value": 1360.0,
                "unrealized_pnl": 160.0,
            }
        ],
        news_ctx={
            "sentiment_label": "watch",
            "risk_signals": ["海外波动加大"],
            "topic_summary": {
                "market": "市场风格继续偏主题轮动。",
                "catalog": "**重点十条**\n1. 黄金价格再度走强",
            },
            "daily_briefing_top10": [{"title": "黄金价格再度走强", "source": "cls"}],
        },
        fund_ctx={
            "analyses": {
                "000001.OF": {
                    "analysis": "近三年整体走势偏强，但短期仍需关注波动。",
                    "trend": "up",
                    "metrics": {"return_3y": 0.32},
                }
            }
        },
        holding_recommendations={
            "000001.OF": {
                "action": "hold",
                "confidence": 0.68,
                "reasoning": "消息面偏暖，当前更适合继续持有等待趋势确认。",
                "relevant_titles": ["黄金价格再度走强"],
            }
        },
        elapsed_ms=12.5,
    )

    assert report.top_news[0]["title"] == "黄金价格再度走强"
    assert "建议持有" in report.overall_summary

    status = report.fund_statuses[0]
    assert status.trend == "up"
    assert status.three_year_return_pct == 0.32
    assert "近三年整体走势偏强" in status.reason
    assert status.related_news == ["黄金价格再度走强"]
