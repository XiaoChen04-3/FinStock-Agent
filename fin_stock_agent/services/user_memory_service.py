from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

from sqlalchemy import select

from fin_stock_agent.agents.memory_extraction_agent import update_user_profile_from_turn
from fin_stock_agent.memory.conversation import ConversationMemory
from fin_stock_agent.memory.profile_memory import MemoryEvent, UserProfileMemory
from fin_stock_agent.memory.user_profile_file import (
    DEFAULT_USER_PROFILE_MD,
    UserProfileValidationError,
    get_user_profile_file_service,
)
from fin_stock_agent.storage.database import get_session
from fin_stock_agent.storage.models import UserMemoryProfileORM

logger = logging.getLogger(__name__)


def _loads_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _join(items: list[str]) -> str:
    return "、".join(item for item in items if item)


class UserMemoryService:
    """用户画像记忆服务门面。

    方法名保持兼容旧的数据库画像服务，便于其他 Agent 层稳定调用。
    生产路径不再写入 ``user_memory_profiles`` 或 ``user_memory_events``。
    """

    def initialize_runtime(self, user_id: str) -> None:
        initial_text = self._legacy_profile_markdown(user_id)
        get_user_profile_file_service().initialize(initial_text=initial_text)

    def get_profile(self, user_id: str) -> UserProfileMemory:
        _ = user_id
        service = get_user_profile_file_service()
        text = service.get_active_profile_text()
        profile = self._profile_from_markdown(text)
        try:
            profile.updated_at = datetime.fromtimestamp(service.snapshot().path.stat().st_mtime, tz=timezone.utc)
        except Exception:
            profile.updated_at = None
        profile.user_id = user_id
        return profile

    def get_recent_events(self, user_id: str, limit: int = 8) -> list[MemoryEvent]:
        _ = (user_id, limit)
        return []

    def remember_turn(self, *, user_id: str, session_id: str, turn_idx: int, question: str, answer: str) -> dict:
        _ = session_id
        self.initialize_runtime(user_id)
        service = get_user_profile_file_service()
        cfg = self._cfg()
        recent_summaries = ConversationMemory(user_id=user_id, session_id=session_id).get_recent_summaries(
            user_id=user_id,
            limit=cfg.extraction_recent_turns,
        )
        current = service.get_staged_profile_text()
        output = update_user_profile_from_turn(
            current_profile_md=current,
            question=question,
            answer=answer,
            max_tokens=cfg.max_tokens,
            recent_summaries=recent_summaries,
        )
        if not output.should_update:
            return {
                "profile_updated": False,
                "event_count": 0,
                "reason": output.reason,
                "turn_idx": turn_idx,
            }
        try:
            changed = service.stage_profile(output.profile_md)
        except UserProfileValidationError as exc:
            logger.warning("拒绝写入本轮提取的用户画像，user=%s turn=%s：%s", user_id, turn_idx, exc)
            return {
                "profile_updated": False,
                "event_count": 0,
                "reason": f"画像校验失败：{exc}",
                "turn_idx": turn_idx,
            }
        return {
            "profile_updated": changed,
            "event_count": 0,
            "reason": output.reason,
            "turn_idx": turn_idx,
        }

    def build_profile_context(self, user_id: str) -> str:
        self.initialize_runtime(user_id)
        profile_md = get_user_profile_file_service().get_active_profile_text().strip()
        return "\n".join(
            [
                "## 用户画像记忆",
                "以下画像来自本次运行启动时读取的 .data/user.md；本轮运行期间保持冻结，仅作为偏好参考。",
                "",
                profile_md,
            ]
        )

    def build_prompt_profile_context(self, user_id: str) -> str:
        self.initialize_runtime(user_id)
        text = get_user_profile_file_service().get_active_profile_text().strip()
        if not text:
            text = DEFAULT_USER_PROFILE_MD.strip()
        return "## 投资者画像\n" + text

    def build_recent_events_context(self, user_id: str, limit: int = 6) -> str:
        _ = (user_id, limit)
        return "## 近期记忆事件\n用户画像已改为存储在 .data/user.md，数据库画像事件写入已停用。"

    def snapshot(self) -> dict:
        snap = get_user_profile_file_service().snapshot()
        return {
            "path": str(snap.path),
            "pending_path": str(snap.pending_path),
            "token_estimate": snap.token_estimate,
            "pending_exists": snap.pending_exists,
            "active_text": snap.active_text,
            "staged_text": snap.staged_text,
        }

    def reset_profile_file(self, profile_md: str | None = None) -> None:
        get_user_profile_file_service().reset(profile_md)

    def commit_pending_profile(self) -> bool:
        return get_user_profile_file_service().commit_pending()

    def _cfg(self):
        from fin_stock_agent.core.config import get_config

        return get_config().memory.user_profile

    def _legacy_profile_markdown(self, user_id: str) -> str | None:
        try:
            with get_session() as session:
                row = session.get(UserMemoryProfileORM, user_id)
                if row is None:
                    return None
                return self._legacy_row_to_markdown(row)
        except Exception as exc:
            logger.debug("跳过旧数据库画像导入，user=%s：%s", user_id, exc)
            return None

    def _legacy_row_to_markdown(self, row: UserMemoryProfileORM) -> str:
        preferred = _loads_list(row.preferred_assets_json)
        disliked = _loads_list(row.disliked_assets_json)
        themes = _loads_list(row.focus_themes_json)
        style = _loads_list(row.answer_style_json)
        constraints = _loads_list(row.decision_constraints_json)
        watchlist = _loads_list(row.watchlist_json)
        return f"""# 用户画像

## 投资偏好
- 风险承受：{row.risk_level or '未知'}
- 投资期限：{row.investment_horizon or '未知'}
- 偏好资产：{_join(preferred) or '暂无'}
- 规避事项：{_join(disliked) or '暂无'}

## 关注范围
- 关注主题：{_join(themes) or '暂无'}
- 自选标的：{_join(watchlist) or '暂无'}

## 回答偏好
- {_join(style) or '暂无'}

## 决策约束
- {_join(constraints) or '仅作研究参考，不替用户下确定性买卖指令'}
"""

    def _profile_from_markdown(self, text: str) -> UserProfileMemory:
        profile = UserProfileMemory()
        lines = [line.strip("- ").strip() for line in (text or "").splitlines()]
        joined = "\n".join(lines)
        profile.risk_level = self._extract_scalar(joined, ["风险承受", "风险偏好"])
        profile.investment_horizon = self._extract_scalar(joined, ["投资期限", "期限"])
        profile.preferred_assets = self._extract_list(joined, ["偏好资产"])
        profile.disliked_assets = self._extract_list(joined, ["规避事项", "规避资产"])
        profile.focus_themes = self._extract_list(joined, ["关注主题", "关注范围"])
        profile.answer_style = self._extract_list(joined, ["回答偏好", "回答风格"])
        profile.decision_constraints = self._extract_list(joined, ["决策约束", "投资约束"])
        profile.watchlist = self._extract_list(joined, ["自选标的"])
        return profile

    @staticmethod
    def _extract_scalar(text: str, labels: list[str]) -> str:
        for label in labels:
            match = re.search(rf"{re.escape(label)}[：:]\s*([^\n]+)", text)
            if match:
                value = match.group(1).strip()
                return "" if value in {"未知", "暂无", "-"} else value
        return ""

    @staticmethod
    def _extract_list(text: str, labels: list[str]) -> list[str]:
        values: list[str] = []
        for label in labels:
            match = re.search(rf"{re.escape(label)}[：:]\s*([^\n]+)", text)
            if not match:
                continue
            raw = match.group(1).strip()
            if raw in {"未知", "暂无", "-"}:
                continue
            parts = re.split(r"[,，、;/；]\s*|\s{2,}", raw)
            values.extend(part.strip() for part in parts if part.strip())
        # 兼容“回答偏好 / 决策约束”这类章节下直接列 bullet 的写法。
        if not values and any(label in text for label in labels):
            section = UserMemoryService._extract_section(text, labels)
            for line in section:
                if line and line not in {"暂无", "未知"}:
                    values.append(line)
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result

    @staticmethod
    def _extract_section(text: str, labels: list[str]) -> list[str]:
        lines = text.splitlines()
        start = -1
        for idx, line in enumerate(lines):
            if any(label in line for label in labels):
                start = idx + 1
                break
        if start < 0:
            return []
        out: list[str] = []
        for line in lines[start:]:
            stripped = line.strip("- ").strip()
            if stripped.startswith("#") or "## " in stripped:
                break
            if stripped:
                out.append(stripped)
        return out
