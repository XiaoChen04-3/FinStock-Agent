from __future__ import annotations

from time import perf_counter
from typing import Literal

from pydantic import BaseModel, Field, ConfigDict

from fin_stock_agent.core.config import DailyReportConfig
from fin_stock_agent.memory.profile_memory import UserProfileMemory
from fin_stock_agent.news.models import NewsItem

AgentStatus = Literal["success", "fallback", "failed"]


class ReportContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: str
    report_date: str
    holdings: list[dict]
    user_profile: UserProfileMemory
    news_items: list[NewsItem]
    raw_nav_history: dict[str, list[dict]]
    recent_trading_days: list[str]
    config: DailyReportConfig


class AgentResult(BaseModel):
    agent_name: str
    status: AgentStatus
    output: dict = Field(default_factory=dict)
    token_usage: dict = Field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )
    elapsed_ms: float = 0.0
    error: str | None = None


class AgentTimer:
    def __init__(self) -> None:
        self._started = perf_counter()

    def elapsed_ms(self) -> float:
        return (perf_counter() - self._started) * 1000
