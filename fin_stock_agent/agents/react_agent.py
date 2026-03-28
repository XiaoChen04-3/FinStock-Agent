"""ReAct agent – used for simple single-intent queries and as PnE fallback."""
from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.prebuilt import create_react_agent

from fin_stock_agent.core.llm import get_llm
from fin_stock_agent.prompts.react_prompt import REACT_SYSTEM_PROMPT
from fin_stock_agent.tools.market import get_market_tools
from fin_stock_agent.tools.memory_tools import get_memory_tools
from fin_stock_agent.tools.portfolio import get_portfolio_tools
from fin_stock_agent.utils.datetime_utils import get_datetime_tools


def _all_tools():
    return [
        *get_datetime_tools(),   # datetime first so it's easy to find in tool list
        *get_market_tools(),
        *get_portfolio_tools(),
        *get_memory_tools(),
    ]


def build_react_agent():
    """Compile and return a ReAct agent graph."""
    return create_react_agent(get_llm(), _all_tools(), prompt=REACT_SYSTEM_PROMPT)


def extract_last_ai_text(messages: list[BaseMessage]) -> str:
    """Return the text content of the last AIMessage in a messages list."""
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            c = m.content
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                parts = [
                    block.get("text", "")
                    for block in c
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                return "\n".join(parts)
    return ""
