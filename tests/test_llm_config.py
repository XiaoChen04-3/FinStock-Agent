from fin_stock_agent.core.llm import build_llm_kwargs, describe_agent_chain, get_llm_profile, role_uses_thinking


def test_online_qa_llm_roles_are_configured() -> None:
    assert get_llm_profile("query_enhancer").model == "qwen3.5-flash"
    assert get_llm_profile("react").model == "qwen3.6-plus"
    assert get_llm_profile("planner").model == "qwen3-max"
    assert get_llm_profile("replan").model == "qwen3-max"
    assert get_llm_profile("executor").model == "qwen3.6-plus"
    assert get_llm_profile("finalize").model == "qwen3.6-plus"

    planner_kwargs = build_llm_kwargs("planner")
    assert planner_kwargs["extra_body"] == {"enable_thinking": True}
    assert role_uses_thinking("planner") is True
    assert role_uses_thinking("replan") is True


def test_daily_report_llm_roles_are_configured() -> None:
    assert get_llm_profile("daily_briefing").model == "qwen3.5-flash"
    assert get_llm_profile("news_analysis").model == "qwen3.6-plus"
    assert get_llm_profile("fund_analysis").model == "qwen3.5-flash"
    assert get_llm_profile("agentic_news").model == "qwen3.6-plus"
    assert get_llm_profile("report_synthesis").model == "qwen3.6-plus"


def test_agent_chain_description_matches_runtime_routes() -> None:
    assert describe_agent_chain("react") == "query_enhancer=qwen3.5-flash; react=qwen3.6-plus"
    assert "planner/replan=qwen3-max(enable_thinking=true,streaming,json-repair)" in describe_agent_chain("plan_execute")
