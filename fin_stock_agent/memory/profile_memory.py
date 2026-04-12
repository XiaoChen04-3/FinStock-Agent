from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class UserProfileMemory(BaseModel):
    user_id: str = ""
    risk_level: str = ""
    investment_horizon: str = ""
    preferred_assets: list[str] = Field(default_factory=list)
    disliked_assets: list[str] = Field(default_factory=list)
    focus_themes: list[str] = Field(default_factory=list)
    answer_style: list[str] = Field(default_factory=list)
    decision_constraints: list[str] = Field(default_factory=list)
    watchlist: list[str] = Field(default_factory=list)
    updated_at: datetime | None = None

    def is_empty(self) -> bool:
        return not any(
            [
                self.risk_level,
                self.investment_horizon,
                self.preferred_assets,
                self.disliked_assets,
                self.focus_themes,
                self.answer_style,
                self.decision_constraints,
                self.watchlist,
            ]
        )


class MemoryEvent(BaseModel):
    event_type: str
    summary: str
    payload: dict = Field(default_factory=dict)
    confidence: float = 0.0
    source_text: str = ""
    created_at: datetime | None = None


class MemoryExtractionResult(BaseModel):
    profile_updates: dict = Field(default_factory=dict)
    events: list[MemoryEvent] = Field(default_factory=list)
