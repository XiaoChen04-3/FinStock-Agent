from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from sqlalchemy import select

from fin_stock_agent.core.config import get_config
from fin_stock_agent.core.llm import get_llm
from fin_stock_agent.memory.vector_store import SearchResult, get_vector_store
from fin_stock_agent.storage.database import get_session
from fin_stock_agent.storage.models import ConversationSummaryORM

logger = logging.getLogger(__name__)
_VECTOR_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="conversation-vector")


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
        vec_id = f"{session_id}_{turn_idx}"
        with get_session() as session:
            session.add(
                ConversationSummaryORM(
                    user_id=self.user_id,
                    session_id=session_id,
                    turn_idx=turn_idx,
                    question=(question or "")[:500],
                    summary=summary[:200],
                    vec_id=vec_id,
                )
            )
        _VECTOR_EXECUTOR.submit(self._upsert_vector, vec_id, summary[:200], session_id, turn_idx)
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

    def search_relevant_summaries(
        self,
        user_id: str,
        question: str,
        *,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[str]:
        cfg = get_config().memory.semantic_search
        try:
            results = get_vector_store().search(
                self._collection_name(user_id),
                question,
                top_k=top_k or cfg.conversation_top_k,
                threshold=threshold if threshold is not None else cfg.similarity_threshold,
            )
            if results:
                return [item.text for item in results]
        except Exception as exc:
            logger.warning("Conversation semantic search failed, falling back: %s", exc)
        return self.get_recent_summaries(user_id=user_id, limit=cfg.time_fallback_limit)

    def build_history_context(self, user_id: str) -> str:
        summaries = self.get_recent_summaries(user_id=user_id, limit=10)
        if not summaries:
            return "## 近期对话摘要\n暂无历史摘要。"
        lines = ["## 近期对话摘要"]
        for summary in summaries:
            lines.append(f"- {summary}")
        return "\n".join(lines)

    def _summarize_turn(self, question: str, answer: str) -> str:
        prompt = (
            "你是对话记忆摘要助手。请将以下用户提问与助手回答压缩为一句简体中文摘要。\n"
            "要求：\n"
            "1. 不超过 50 个汉字。\n"
            "2. 保留核心实体（股票/基金名称或代码）和关键数据（价格、涨跌幅、净值等）。\n"
            "3. 直接描述对话要点，不得包含\"用户问\"、\"助手答\"等标签。\n"
            "4. 若问答均无实质内容，返回原始问题的前 30 个字符。\n\n"
            f"用户提问：{question}\n"
            f"助手回答：{answer}"
        )
        try:
            llm = get_llm("conversation_summarizer", temperature=0)
            response = llm.invoke([HumanMessage(content=prompt)])
            if isinstance(response.content, str) and response.content.strip():
                return response.content.strip().replace("\n", " ")[:200]
        except Exception as exc:
            logger.warning("Turn summarization failed, using question prefix as fallback: %s", exc)
        return (question or "")[:50]

    def clear_user(self, user_id: str) -> None:
        with get_session() as session:
            rows = session.execute(
                select(ConversationSummaryORM).where(ConversationSummaryORM.user_id == user_id)
            ).scalars().all()
            for row in rows:
                session.delete(row)
        try:
            get_vector_store().delete_collection(self._collection_name(user_id))
        except Exception:
            logger.debug("Ignoring conversation collection delete failure for %s", user_id)

    def __len__(self) -> int:
        return len(self._rows)

    def _upsert_vector(self, vec_id: str, summary: str, session_id: str, turn_idx: int) -> None:
        try:
            get_vector_store().upsert(
                self._collection_name(self.user_id),
                vec_id,
                summary,
                {
                    "user_id": self.user_id,
                    "session_id": session_id,
                    "turn_idx": turn_idx,
                },
            )
        except Exception as exc:
            logger.warning("Conversation vector upsert failed: %s", exc)

    @staticmethod
    def _collection_name(user_id: str) -> str:
        return f"{user_id}_conversations"
