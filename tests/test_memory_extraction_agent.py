from __future__ import annotations

from fin_stock_agent.agents import memory_extraction_agent as agent
from fin_stock_agent.core.settings import settings
from fin_stock_agent.memory.user_profile_file import reset_user_profile_file_service_for_tests


def _isolate(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(settings, "project_root", tmp_path)
    reset_user_profile_file_service_for_tests()


def test_memory_extraction_agent_accepts_structured_profile(monkeypatch, tmp_path) -> None:
    _isolate(monkeypatch, tmp_path)

    def fake_invoke_json(role, messages, config=None):
        assert role == "memory_extractor"
        return {
            "should_update": True,
            "reason": "stable preference",
            "profile_md": "# 用户画像\n\n## 投资偏好\n- 风险承受：稳健\n\n## 关注范围\n- 关注主题：红利\n\n## 回答偏好\n- 简洁\n\n## 决策约束\n- 仅作研究参考\n",
        }

    monkeypatch.setattr(agent, "invoke_json", fake_invoke_json)

    output = agent.update_user_profile_from_turn(
        current_profile_md="# 用户画像\n\n## 投资偏好\n- 风险承受：未知\n",
        question="以后简洁回答，我偏稳健，关注红利。",
        answer="好的。",
        max_tokens=500,
    )

    assert output.should_update is True
    assert "红利" in output.profile_md


def test_memory_extraction_agent_compresses_overlong_output(monkeypatch, tmp_path) -> None:
    _isolate(monkeypatch, tmp_path)
    calls = {"count": 0}

    def fake_invoke_json(role, messages, config=None):
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "should_update": True,
                "reason": "too long",
                "profile_md": "# 用户画像\n\n" + "长" * 600,
            }
        return {
            "profile_md": "# 用户画像\n\n## 投资偏好\n- 风险承受：稳健\n\n## 决策约束\n- 仅作研究参考\n",
        }

    monkeypatch.setattr(agent, "invoke_json", fake_invoke_json)

    output = agent.update_user_profile_from_turn(
        current_profile_md="# 用户画像\n\n## 投资偏好\n- 风险承受：未知\n",
        question="我偏稳健。",
        answer="好的。",
        max_tokens=500,
    )

    assert output.should_update is True
    assert calls["count"] == 2
    assert "稳健" in output.profile_md


def test_memory_extraction_agent_contains_llm_failures(monkeypatch, tmp_path) -> None:
    _isolate(monkeypatch, tmp_path)

    def fake_invoke_json(role, messages, config=None):
        raise RuntimeError("network down")

    monkeypatch.setattr(agent, "invoke_json", fake_invoke_json)

    output = agent.update_user_profile_from_turn(
        current_profile_md="# 用户画像\n\n## 投资偏好\n- 风险承受：未知\n",
        question="我偏稳健。",
        answer="好的。",
        max_tokens=500,
    )

    assert output.should_update is False
    assert "画像提取失败" in output.reason
