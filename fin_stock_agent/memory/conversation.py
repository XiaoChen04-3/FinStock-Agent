from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class ConversationMemory:
    """Manages multi-turn message history and converts to LangChain message objects."""

    def __init__(self) -> None:
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

    def __len__(self) -> int:
        return len(self._rows)
