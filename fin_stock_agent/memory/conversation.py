from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from sqlalchemy import select

from fin_stock_agent.core.llm import get_llm
from fin_stock_agent.storage.database import get_session
from fin_stock_agent.storage.models import ConversationSummaryORM


class ConversationMemory:
    def __init__(self, user_id: str = "default-user", session_id: str = "default-session") -> None:
        self.user_id = user_id
        self.session_id = session_id
        self._rows: list[dict] = []

    def add_user(self, text: str) -> None:
        self._rows.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self._rows.append({"role": "assistant", "content": text})

    def all_rows(self) -> list[dict]:
        return list(self._rows)

    def to_lc_messages(self) -> list[BaseMessage]:
        out: list[BaseMessage] = []
        for row in self._rows:
            if row["role"] == "user":
                out.append(HumanMessage(content=row["content"]))
            else:
                out.append(AIMessage(content=row["content"]))
        return out

    def clear(self) -> None:
        self._rows.clear()

    def save_turn_summary(self, session_id: str, turn_idx: int, question: str, answer: str) -> str:
        summary = self._summarize_turn(question, answer)
        with get_session() as session:
            session.add(
                ConversationSummaryORM(
                    user_id=self.user_id,
                    session_id=session_id,
                    turn_idx=turn_idx,
                    question=(question or "")[:500],
                    summary=summary[:200],
                )
            )
        return summary

    def get_recent_summaries(self, user_id: str, limit: int = 10) -> list[str]:
        with get_session() as session:
            rows = session.execute(
                select(ConversationSummaryORM)
                .where(ConversationSummaryORM.user_id == user_id)
                .order_by(ConversationSummaryORM.created_at.desc())
                .limit(limit)
            ).scalars()
            return [row.summary for row in rows]

    def build_history_context(self, user_id: str) -> str:
        summaries = self.get_recent_summaries(user_id=user_id, limit=10)
        if not summaries:
            return "## Recent conversation summary\nNo persisted summaries yet."
        lines = ["## Recent conversation summary"]
        for summary in summaries:
            lines.append(f"- {summary}")
        return "\n".join(lines)

    def _summarize_turn(self, question: str, answer: str) -> str:
        prompt = (
            "Summarize the following user question and assistant answer in one Chinese sentence. "
            "Keep it under 50 Chinese characters.\n"
            f"Question: {question}\nAnswer: {answer}"
        )
        try:
            llm = get_llm(temperature=0)
            response = llm.invoke([HumanMessage(content=prompt)])
            if isinstance(response.content, str) and response.content.strip():
                return response.content.strip().replace("\n", " ")[:200]
        except Exception:
            pass
        return (question or "")[:50]

    def __len__(self) -> int:
        return len(self._rows)
