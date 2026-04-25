from __future__ import annotations

from pathlib import Path

import pytest

from fin_stock_agent.core.config import AppConfig, ConfigValidationError


def test_config_loads_example_values() -> None:
    AppConfig.reset_for_tests()
    cfg = AppConfig.load("config.yaml", exit_on_error=False, force_reload=True)
    assert cfg.models.planner
    assert cfg.memory.vector_store.backend == "chromadb"
    assert cfg.plan_execute.min_plan_steps <= cfg.plan_execute.max_plan_steps


def test_config_rejects_invalid_threshold(tmp_path: Path) -> None:
    broken = tmp_path / "config.yaml"
    broken.write_text(
        "models:\n"
        "  query_enhancer: a\n"
        "  conversation_summarizer: b\n"
        "  react_agent: c\n"
        "  planner: d\n"
        "  replanner: e\n"
        "  executor: f\n"
        "  finalizer: g\n"
        "  news_filter: h\n"
        "  sentiment_analysis: i\n"
        "  fund_trend: j\n"
        "  holding_correlation: k\n"
        "  report_generation: l\n"
        "  embedding: m\n"
        "memory:\n"
        "  prompt_memory_max_chars: 2000\n"
        "  semantic_search:\n"
        "    conversation_top_k: 5\n"
        "    digest_top_k: 3\n"
        "    time_fallback_limit: 5\n"
        "    similarity_threshold: 1.5\n"
        "  plan_library:\n"
        "    reuse_threshold: 0.9\n"
        "    reference_threshold: 0.82\n"
        "    min_quality_score: 0.5\n"
        "    max_records_per_user: 500\n"
        "  vector_store:\n"
        "    backend: chromadb\n"
        "    max_total_records: 5000\n"
        "daily_report:\n"
        "  news_fetch_limit: 50\n"
        "  briefing_top_n: 10\n"
        "  personalized_news_top_n: 5\n"
        "  nav_history_years: 3\n"
        "  recent_trading_days: 3\n"
        "  cache_ttl_hours: 12\n"
        "  report_summary_max_chars: 180\n"
        "plan_execute:\n"
        "  max_plan_steps: 6\n"
        "  min_plan_steps: 2\n"
        "  max_errors_before_fallback: 3\n"
        "  context_max_chars: 2000\n"
        "test_api:\n"
        "  host: 127.0.0.1\n"
        "  port: 8765\n"
        "concurrency:\n"
        "  post_turn_workers: 4\n"
        "  daily_report_workers: 2\n",
        encoding="utf-8",
    )

    AppConfig.reset_for_tests()
    with pytest.raises(ConfigValidationError):
        AppConfig.load(broken, exit_on_error=False, force_reload=True)
