from __future__ import annotations

import uuid

from fin_stock_agent.agents import router
from fin_stock_agent.agents.memory_extraction_agent import UserProfileExtractionOutput
from fin_stock_agent.core.query_enhancer import EnhancedQuery, IntentType
from fin_stock_agent.core.settings import settings
from fin_stock_agent.memory.user_profile_file import get_user_profile_file_service, reset_user_profile_file_service_for_tests
from fin_stock_agent.services.local_user_service import LocalUserService
from fin_stock_agent.services.memory_manager import MemoryManager
from fin_stock_agent.services.user_memory_service import UserMemoryService
from fin_stock_agent.storage.database import get_session, init_db
from fin_stock_agent.storage.models import UserMemoryEventORM, UserMemoryProfileORM


def _isolate_profile_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(settings, "project_root", tmp_path)
    reset_user_profile_file_service_for_tests()


def test_user_memory_service_stages_profile_file_without_db_writes(monkeypatch, tmp_path) -> None:
    init_db()
    _isolate_profile_file(monkeypatch, tmp_path)
    user_id = f"memory-user-{uuid.uuid4()}"

    def fake_update(**kwargs):
        return UserProfileExtractionOutput(
            should_update=True,
            reason="stable preference",
            profile_md="# 用户画像\n\n## 投资偏好\n- 风险承受：稳健\n\n## 关注范围\n- 关注主题：黄金、红利\n- 自选标的：510300.SH\n\n## 回答偏好\n- 简洁\n\n## 决策约束\n- 不主动推荐具体股票\n",
        )

    monkeypatch.setattr("fin_stock_agent.services.user_memory_service.update_user_profile_from_turn", fake_update)
    service = UserMemoryService()

    result = service.remember_turn(
        user_id=user_id,
        session_id="session-1",
        turn_idx=1,
        question="以后默认用简洁风格回答，我是稳健投资者，重点关注黄金和红利。",
        answer="好的。",
    )

    profile = service.get_profile(user_id)
    events = service.get_recent_events(user_id, limit=10)
    snapshot = service.snapshot()

    assert result["profile_updated"] is True
    assert events == []
    assert snapshot["pending_exists"] is True
    assert "黄金" not in snapshot["active_text"]
    assert "黄金" in snapshot["staged_text"]
    assert profile.is_empty() is False  # default file has a decision constraint

    with get_session() as session:
        assert session.query(UserMemoryProfileORM).filter(UserMemoryProfileORM.user_id == user_id).count() == 0
        assert session.query(UserMemoryEventORM).filter(UserMemoryEventORM.user_id == user_id).count() == 0


def test_profile_file_freezes_until_commit(monkeypatch, tmp_path) -> None:
    init_db()
    _isolate_profile_file(monkeypatch, tmp_path)
    user_id = f"memory-context-{uuid.uuid4()}"
    service = UserMemoryService()
    service.reset_profile_file("# 用户画像\n\n## 投资偏好\n- 风险承受：保守\n\n## 关注范围\n- 关注主题：黄金\n\n## 回答偏好\n- 先结论\n\n## 决策约束\n- 仅作研究参考\n")

    def fake_update(**kwargs):
        return UserProfileExtractionOutput(
            should_update=True,
            reason="new preference",
            profile_md="# 用户画像\n\n## 投资偏好\n- 风险承受：积极\n\n## 关注范围\n- 关注主题：科技\n\n## 回答偏好\n- 表格化\n\n## 决策约束\n- 仅作研究参考\n",
        )

    monkeypatch.setattr("fin_stock_agent.services.user_memory_service.update_user_profile_from_turn", fake_update)
    manager = MemoryManager(user_id=user_id, session_id="session-2")

    manager.after_turn(1, "以后我更积极，关注科技。", "收到。")

    context = manager.build_context_block("看看我的组合")
    prompt_context = manager.build_prompt_memory_block("看看我的组合")
    assert "保守" in context
    assert "黄金" in context
    assert "积极" not in service.snapshot()["active_text"]
    assert "保守" in prompt_context

    changed = service.commit_pending_profile()
    assert changed is True
    reset_user_profile_file_service_for_tests()
    monkeypatch.setattr(settings, "project_root", tmp_path)
    refreshed_context = MemoryManager(user_id=user_id, session_id="session-3").build_context_block("看看我的组合")
    assert "积极" in refreshed_context
    assert "科技" in refreshed_context


def test_prep_session_includes_startup_profile_snapshot(monkeypatch, tmp_path) -> None:
    init_db()
    _isolate_profile_file(monkeypatch, tmp_path)
    user_id = f"memory-session-{uuid.uuid4()}"
    UserMemoryService().reset_profile_file("# 用户画像\n\n## 投资偏好\n- 风险承受：稳健\n\n## 关注范围\n- 关注主题：红利、银行\n\n## 回答偏好\n- 表格化\n\n## 决策约束\n- 仅作研究参考\n")

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
    assert "红利、银行" in system_message


def test_local_user_service_exports_legacy_profile_to_file(monkeypatch, tmp_path) -> None:
    init_db()
    _isolate_profile_file(monkeypatch, tmp_path)
    monkeypatch.setattr("fin_stock_agent.services.local_user_service.write_stats_event", lambda *args, **kwargs: None)
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
    snapshot = get_user_profile_file_service().snapshot()

    assert summary["memory_profile_rows_migrated"] >= 1
    assert "稳健" in snapshot.active_text
    assert "黄金" in snapshot.active_text
    assert UserMemoryService().get_recent_events(canonical_user_id, limit=5) == []
