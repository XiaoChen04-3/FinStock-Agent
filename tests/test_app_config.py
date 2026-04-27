from __future__ import annotations

from pathlib import Path

import pytest

from fin_stock_agent.core.config import AppConfig, ConfigValidationError


def test_config_loads_example_values() -> None:
    AppConfig.reset_for_tests()
    cfg = AppConfig.load("config.yaml", exit_on_error=False, force_reload=True)
    assert cfg.models.planner
    assert cfg.models.memory_extractor
    assert cfg.memory.vector_store.backend == "chromadb"
    assert cfg.memory.user_profile.max_tokens == 500
    assert cfg.plan_execute.min_plan_steps <= cfg.plan_execute.max_plan_steps


def test_config_rejects_invalid_threshold(tmp_path: Path) -> None:
    broken = tmp_path / "config.yaml"
    broken.write_text(
        "models:\n"
        "  query_enhancer: a\n"
        "  conversation_summarizer: b\n"
        "  memory_extractor: c\n"
        "  react_agent: d\n"
        "  planner: e\n"
        "  replanner: f\n"
        "  executor: g\n"
        "  finalizer: h\n"
        "  news_filter: i\n"
        "  sentiment_analysis: j\n"
        "  fund_trend: k\n"
        "  holding_correlation: l\n"
        "  report_generation: m\n"
        "  embedding: n\n"
        "memory:\n"
        "  prompt_memory_max_chars: 2000\n"
        "  user_profile:\n"
        "    path: .data/user.md\n"
        "    pending_path: .data/user.pending.md\n"
        "    max_tokens: 500\n"
        "    commit_on_shutdown: true\n"
        "    freeze_during_runtime: true\n"
        "    extraction_recent_turns: 1\n"
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


def test_config_rejects_invalid_user_profile_paths(tmp_path: Path) -> None:
    broken = tmp_path / "config.yaml"
    base = Path("config.yaml").read_text(encoding="utf-8")
    broken.write_text(base.replace('pending_path: ".data/user.pending.md"', 'pending_path: ".data/user.md"'), encoding="utf-8")

    AppConfig.reset_for_tests()
    with pytest.raises(ConfigValidationError):
        AppConfig.load(broken, exit_on_error=False, force_reload=True)
