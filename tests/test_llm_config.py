from __future__ import annotations

from fin_stock_agent.core.config import get_config
from fin_stock_agent.core.llm import build_llm_kwargs, describe_agent_chain, get_llm_profile, role_uses_thinking


def test_online_qa_llm_roles_are_configured() -> None:
    cfg = get_config()
    assert get_llm_profile("query_enhancer").model == cfg.models.query_enhancer
    assert get_llm_profile("react").model == cfg.models.react_agent
    assert get_llm_profile("planner").model == cfg.models.planner
    assert get_llm_profile("replan").model == cfg.models.replanner
    assert get_llm_profile("executor").model == cfg.models.executor
    assert get_llm_profile("finalize").model == cfg.models.finalizer

    planner_kwargs = build_llm_kwargs("planner")
    assert planner_kwargs["extra_body"] == {"enable_thinking": True}
    assert role_uses_thinking("planner") is True
    assert role_uses_thinking("replan") is True


def test_daily_report_llm_roles_are_configured() -> None:
    cfg = get_config()
    assert get_llm_profile("daily_briefing").model == cfg.models.news_filter
    assert get_llm_profile("news_analysis").model == cfg.models.sentiment_analysis
    assert get_llm_profile("fund_analysis").model == cfg.models.fund_trend
    assert get_llm_profile("agentic_news").model == cfg.models.holding_correlation
    assert get_llm_profile("report_synthesis").model == cfg.models.report_generation


def test_agent_chain_description_matches_runtime_routes() -> None:
    cfg = get_config()
    assert describe_agent_chain("react") == f"query_enhancer={cfg.models.query_enhancer}; react={cfg.models.react_agent}"
    assert f"planner/replan={cfg.models.planner}(enable_thinking=true,streaming,json-repair)" in describe_agent_chain(
        "plan_execute"
    )
