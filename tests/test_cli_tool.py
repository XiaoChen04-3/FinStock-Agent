from __future__ import annotations

from tools import cli_test


def test_force_plan_mode_context_manager_restores_router() -> None:
    original = cli_test.router.classify_mode
    with cli_test._force_plan_mode():
        assert cli_test.router.classify_mode(None) == "plan_execute"
    assert cli_test.router.classify_mode is original
