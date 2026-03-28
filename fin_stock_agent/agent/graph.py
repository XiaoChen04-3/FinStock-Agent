from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from finstock_agent.agent.prompts import SYSTEM_PROMPT
from finstock_agent.config import settings
from finstock_agent.tools.market import get_market_tools
from finstock_agent.tools.portfolio import get_portfolio_tools


def build_agent():
    if not settings.openai_api_key:
        raise RuntimeError(
            "未配置 OPENAI_API_KEY。请在 .env 中设置或使用 OpenAI 兼容网关的密钥。"
        )
    model = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or None,
        temperature=0.2,
    )
    tools = [*get_market_tools(), *get_portfolio_tools()]
    graph = create_react_agent(model, tools, prompt=SYSTEM_PROMPT)
    return graph


def last_ai_text(messages: list[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            c = m.content
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                parts = []
                for block in c:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                return "\n".join(parts)
    return ""
