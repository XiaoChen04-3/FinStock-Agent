from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import select

from fin_stock_agent.memory.memory_extractor import extract_memory_updates
from fin_stock_agent.memory.profile_memory import MemoryEvent, UserProfileMemory
from fin_stock_agent.storage.database import get_session
from fin_stock_agent.storage.models import UserMemoryEventORM, UserMemoryProfileORM


def _loads_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return [str(item) for item in value if str(item).strip()]


def _dumps_list(items: list[str]) -> str:
    return json.dumps(list(items), ensure_ascii=False)


class UserMemoryService:
    def get_profile(self, user_id: str) -> UserProfileMemory:
        with get_session() as session:
            row = session.get(UserMemoryProfileORM, user_id)
            if row is None:
                return UserProfileMemory(user_id=user_id)
            return UserProfileMemory(
                user_id=user_id,
                risk_level=row.risk_level or "",
                investment_horizon=row.investment_horizon or "",
                preferred_assets=_loads_list(row.preferred_assets_json),
                disliked_assets=_loads_list(row.disliked_assets_json),
                focus_themes=_loads_list(row.focus_themes_json),
                answer_style=_loads_list(row.answer_style_json),
                decision_constraints=_loads_list(row.decision_constraints_json),
                watchlist=_loads_list(row.watchlist_json),
                updated_at=row.updated_at,
            )

    def get_recent_events(self, user_id: str, limit: int = 8) -> list[MemoryEvent]:
        with get_session() as session:
            rows = session.execute(
                select(UserMemoryEventORM)
                .where(UserMemoryEventORM.user_id == user_id)
                .order_by(UserMemoryEventORM.created_at.desc(), UserMemoryEventORM.id.desc())
                .limit(limit)
            ).scalars()
            events: list[MemoryEvent] = []
            for row in rows:
                payload = {}
                if row.payload_json:
                    try:
                        payload = json.loads(row.payload_json)
                    except json.JSONDecodeError:
                        payload = {}
                events.append(
                    MemoryEvent(
                        event_type=row.event_type,
                        summary=row.summary,
                        payload=payload,
                        confidence=float(row.confidence or 0.0),
                        source_text=row.source_text or "",
                        created_at=row.created_at,
                    )
                )
            return events

    def remember_turn(self, *, user_id: str, session_id: str, turn_idx: int, question: str, answer: str) -> dict:
        profile = self.get_profile(user_id)
        extracted = extract_memory_updates(question, answer, existing_profile=profile)
        if extracted.profile_updates:
            self._upsert_profile(user_id, extracted.profile_updates)
        if extracted.events:
            self._append_events(
                user_id=user_id,
                session_id=session_id,
                turn_idx=turn_idx,
                events=extracted.events,
            )
        return {
            "profile_updated": bool(extracted.profile_updates),
            "event_count": len(extracted.events),
        }

    def build_profile_context(self, user_id: str) -> str:
        profile = self.get_profile(user_id)
        if profile.is_empty():
            return "## User profile memory\nNo persisted user profile yet."

        lines = ["## User profile memory"]
        if profile.risk_level:
            lines.append(f"- Risk level: {profile.risk_level}")
        if profile.investment_horizon:
            lines.append(f"- Investment horizon: {profile.investment_horizon}")
        if profile.focus_themes:
            lines.append(f"- Focus themes: {', '.join(profile.focus_themes[:8])}")
        if profile.preferred_assets:
            lines.append(f"- Preferred assets: {', '.join(profile.preferred_assets[:8])}")
        if profile.disliked_assets:
            lines.append(f"- Avoided assets: {', '.join(profile.disliked_assets[:8])}")
        if profile.answer_style:
            lines.append(f"- Answer style: {', '.join(profile.answer_style[:8])}")
        if profile.decision_constraints:
            lines.append(f"- Constraints: {', '.join(profile.decision_constraints[:8])}")
        if profile.watchlist:
            lines.append(f"- Watchlist: {', '.join(profile.watchlist[:8])}")
        return "\n".join(lines)

    def build_recent_events_context(self, user_id: str, limit: int = 6) -> str:
        events = self.get_recent_events(user_id=user_id, limit=limit)
        if not events:
            return "## Recent memory events\nNo persisted memory events yet."
        lines = ["## Recent memory events"]
        for event in events:
            lines.append(f"- {event.summary}")
        return "\n".join(lines)

    def _upsert_profile(self, user_id: str, updates: dict) -> None:
        with get_session() as session:
            row = session.get(UserMemoryProfileORM, user_id)
            if row is None:
                row = UserMemoryProfileORM(user_id=user_id)
                session.add(row)
            if "risk_level" in updates:
                row.risk_level = updates["risk_level"] or None
            if "investment_horizon" in updates:
                row.investment_horizon = updates["investment_horizon"] or None
            if "preferred_assets" in updates:
                row.preferred_assets_json = _dumps_list(updates["preferred_assets"])
            if "disliked_assets" in updates:
                row.disliked_assets_json = _dumps_list(updates["disliked_assets"])
            if "focus_themes" in updates:
                row.focus_themes_json = _dumps_list(updates["focus_themes"])
            if "answer_style" in updates:
                row.answer_style_json = _dumps_list(updates["answer_style"])
            if "decision_constraints" in updates:
                row.decision_constraints_json = _dumps_list(updates["decision_constraints"])
            if "watchlist" in updates:
                row.watchlist_json = _dumps_list(updates["watchlist"])
            row.updated_at = datetime.utcnow()

    def _append_events(self, *, user_id: str, session_id: str, turn_idx: int, events: list[MemoryEvent]) -> None:
        with get_session() as session:
            for event in events:
                session.add(
                    UserMemoryEventORM(
                        user_id=user_id,
                        session_id=session_id,
                        turn_idx=turn_idx,
                        event_type=event.event_type,
                        summary=event.summary[:300],
                        payload_json=json.dumps(event.payload, ensure_ascii=False) if event.payload else None,
                        confidence=event.confidence,
                        source_text=event.source_text[:2000],
                    )
                )
