from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.prebuilt import create_react_agent

from fin_stock_agent.core.llm import get_llm
from fin_stock_agent.prompts.react_prompt import REACT_SYSTEM_PROMPT
from fin_stock_agent.tools import get_all_tools


def _all_tools() -> list:
    return get_all_tools()


def build_react_agent(role: str = "react"):
    return create_react_agent(get_llm(role), _all_tools(), prompt=REACT_SYSTEM_PROMPT)


def extract_last_ai_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = message.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                return "\n".join(parts)
    return ""
