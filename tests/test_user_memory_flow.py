from __future__ import annotations

import uuid

from fin_stock_agent.agents import router
from fin_stock_agent.core.query_enhancer import EnhancedQuery, IntentType
from fin_stock_agent.services.local_user_service import LocalUserService
from fin_stock_agent.services.memory_manager import MemoryManager
from fin_stock_agent.services.user_memory_service import UserMemoryService
from fin_stock_agent.storage.database import get_session, init_db
from fin_stock_agent.storage.models import UserMemoryEventORM, UserMemoryProfileORM


def test_user_memory_service_remember_turn_extracts_profile_and_events() -> None:
    init_db()
    user_id = f"memory-user-{uuid.uuid4()}"
    service = UserMemoryService()

    result = service.remember_turn(
        user_id=user_id,
        session_id="session-1",
        turn_idx=1,
        question="以后默认用简洁风格回答，我是稳健投资者，重点关注黄金和红利，不要推荐具体股票，继续跟踪510300.SH。",
        answer="好的，我后续会按这个偏好来分析。",
    )

    profile = service.get_profile(user_id)
    events = service.get_recent_events(user_id, limit=10)

    assert result["profile_updated"] is True
    assert profile.risk_level == "稳健"
    assert "黄金" in profile.focus_themes
    assert "红利" in profile.focus_themes
    assert "简洁" in "".join(profile.answer_style)
    assert "不要推荐具体股票" in profile.decision_constraints
    assert "510300.SH" in profile.watchlist
    assert len(events) >= 4


def test_memory_manager_context_includes_profile_and_prompt_memory() -> None:
    init_db()
    user_id = f"memory-context-{uuid.uuid4()}"
    manager = MemoryManager(user_id=user_id, session_id="session-2")

    manager.after_turn(
        1,
        "以后先给结论再列风险，我比较保守，关注黄金。",
        "收到，后续我会按这个风格回答。",
    )

    context = manager.build_context_block("看看我的组合")
    prompt_context = manager.build_prompt_memory_block("看看我的组合")

    assert "## 用户画像记忆" in context
    assert "风险偏好：保守" in context
    assert "关注主题：黄金" in context
    assert "## 投资者画像" in prompt_context
    assert "风险偏好：保守" in prompt_context
    assert "## 近期摘要" in prompt_context


def test_prep_session_includes_persisted_memory_context(monkeypatch) -> None:
    init_db()
    user_id = f"memory-session-{uuid.uuid4()}"
    UserMemoryService().remember_turn(
        user_id=user_id,
        session_id="session-3",
        turn_idx=1,
        question="以后默认表格化一点，我更关注红利和银行。",
        answer="好的。",
    )

    monkeypatch.setattr(
        router,
        "enhance_query",
        lambda question, resolver=None, callbacks=None: EnhancedQuery(
            original=question,
            rewritten=question,
            intent=IntentType.PORTFOLIO_QUERY,
            complexity="simple",
        ),
    )

    _, messages, _, _ = router._prep_session(
        "看看我的组合",
        user_id=user_id,
        session_id="session-3",
        memory=None,
        history_messages=[],
    )

    system_message = messages[0].content
    assert "## 用户画像记忆" in system_message
    assert "关注主题：红利, 银行" in system_message


def test_local_user_service_migrates_user_memory_rows() -> None:
    init_db()
    canonical_user_id = f"canonical-{uuid.uuid4()}"
    legacy_user_id = f"legacy-{uuid.uuid4()}"

    with get_session() as session:
        session.add(
            UserMemoryProfileORM(
                user_id=legacy_user_id,
                risk_level="稳健",
                focus_themes_json='["黄金"]',
            )
        )
        session.add(
            UserMemoryEventORM(
                user_id=legacy_user_id,
                session_id="legacy-session",
                turn_idx=1,
                event_type="focus_themes",
                summary="用户关注主题：黄金",
            )
        )

    summary = LocalUserService().consolidate_legacy_data(canonical_user_id)

    assert summary["memory_event_rows_migrated"] >= 1
    assert summary["memory_profile_rows_migrated"] >= 1

    profile = UserMemoryService().get_profile(canonical_user_id)
    events = UserMemoryService().get_recent_events(canonical_user_id, limit=5)
    assert profile.risk_level == "稳健"
    assert "黄金" in profile.focus_themes
    assert any(event.summary == "用户关注主题：黄金" for event in events)
