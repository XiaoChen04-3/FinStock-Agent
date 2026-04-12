from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fin_stock_agent.core.llm import describe_agent_chain
from fin_stock_agent.core.settings import settings
from fin_stock_agent.core.time_utils import local_now_iso
from fin_stock_agent.storage.database import get_session
from fin_stock_agent.storage.models import StatRecordORM

_PERSIST_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="stats-persist")


@dataclass
class RunStats:
    event_type: str
    session_id: str
    query_text: str
    user_id: str = ""
    query_rewritten: str = ""
    intent: str = ""
    agent_mode: str = ""
    query_complexity: str = ""
    suggested_tools: list[str] = field(default_factory=list)
    resolved_codes: dict[str, str] = field(default_factory=dict)
    tool_names_called: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    elapsed_ms: float = 0.0
    llm_elapsed_ms: float = 0.0
    has_error: bool = False
    started_at: str = ""
    finished_at: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class StatsTracker:
    def __init__(
        self,
        session_id: str,
        query_text: str,
        *,
        user_id: str = "",
        event_type: str = "chat_query",
    ) -> None:
        self.data = RunStats(
            event_type=event_type,
            session_id=session_id,
            user_id=user_id,
            query_text=query_text,
            started_at=local_now_iso(),
        )
        self._started_at = time.perf_counter()

    def set_intent(self, intent: str) -> None:
        self.data.intent = intent

    def set_mode(self, mode: str) -> None:
        self.data.agent_mode = mode

    def add_tool(self, name: str) -> None:
        self.data.tool_names_called.append(name)

    def set_query_context(
        self,
        *,
        rewritten: str = "",
        complexity: str = "",
        suggested_tools: list[str] | None = None,
        resolved_codes: dict[str, str] | None = None,
        history_message_count: int | None = None,
    ) -> None:
        self.data.query_rewritten = rewritten
        self.data.query_complexity = complexity
        self.data.suggested_tools = list(suggested_tools or [])
        self.data.resolved_codes = dict(resolved_codes or {})
        if history_message_count is not None:
            self.data.extra["history_message_count"] = history_message_count

    def add_metadata(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            self.data.extra[key] = value

    def finish(self, *, has_error: bool = False, background: bool = False) -> RunStats:
        self.data.elapsed_ms = (time.perf_counter() - self._started_at) * 1000
        self.data.has_error = has_error
        self.data.total_tokens = self.data.prompt_tokens + self.data.completion_tokens
        self.data.finished_at = local_now_iso()
        if background:
            _PERSIST_EXECUTOR.submit(self._persist)
        else:
            self._persist()
        return self.data

    def _persist(self) -> None:
        record = {
            "ts": local_now_iso(),
            "event_type": self.data.event_type,
            "session_id": self.data.session_id,
            "user_id": self.data.user_id,
            "query_text": self.data.query_text,
            "query_rewritten": self.data.query_rewritten,
            "intent": self.data.intent,
            "agent_mode": self.data.agent_mode,
            "query_complexity": self.data.query_complexity,
            "suggested_tools": self.data.suggested_tools,
            "resolved_codes": self.data.resolved_codes,
            "prompt_tokens": self.data.prompt_tokens,
            "completion_tokens": self.data.completion_tokens,
            "total_tokens": self.data.total_tokens,
            "elapsed_ms": self.data.elapsed_ms,
            "llm_elapsed_ms": self.data.llm_elapsed_ms,
            "tool_call_count": len(self.data.tool_names_called),
            "tool_names_called": self.data.tool_names_called,
            "has_error": self.data.has_error,
            "started_at": self.data.started_at,
            "finished_at": self.data.finished_at,
        }
        record.update(self.data.extra)
        _append_jsonl_record(record)
        with get_session() as session:
            session.add(
                StatRecordORM(
                    session_id=self.data.session_id,
                    query_text=self.data.query_text[:500],
                    intent=self.data.intent,
                    agent_mode=self.data.agent_mode,
                    prompt_tokens=self.data.prompt_tokens,
                    completion_tokens=self.data.completion_tokens,
                    total_tokens=self.data.total_tokens,
                    elapsed_ms=self.data.elapsed_ms,
                    llm_elapsed_ms=self.data.llm_elapsed_ms,
                    tool_call_count=len(self.data.tool_names_called),
                    tool_names_called=json.dumps(self.data.tool_names_called, ensure_ascii=False),
                    model_name=describe_agent_chain(self.data.agent_mode),
                    has_error=self.data.has_error,
                )
            )


def write_stats_event(
    event_type: str,
    *,
    session_id: str = "",
    user_id: str = "",
    query_text: str = "",
    intent: str = "",
    agent_mode: str = "",
    has_error: bool = False,
    persist_db: bool = False,
    **payload: Any,
) -> None:
    record = {
        "ts": local_now_iso(),
        "event_type": event_type,
        "session_id": session_id,
        "user_id": user_id,
        "query_text": query_text,
        "intent": intent,
        "agent_mode": agent_mode,
        "has_error": has_error,
    }
    record.update(payload)
    _append_jsonl_record(record)
    if not persist_db:
        return
    with get_session() as session:
        session.add(
            StatRecordORM(
                session_id=session_id[:36],
                query_text=query_text[:500],
                intent=intent[:50],
                agent_mode=agent_mode[:20],
                model_name=describe_agent_chain(agent_mode) if agent_mode else None,
                has_error=has_error,
            )
        )


def _append_jsonl_record(record: dict[str, Any]) -> None:
    with (settings.log_dir / "finstock_stats.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
