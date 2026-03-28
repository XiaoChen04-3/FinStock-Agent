"""LLM factory – all agents share the same ChatOpenAI instance."""
from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from fin_stock_agent.core.settings import settings


@lru_cache(maxsize=1)
def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    if not settings.openai_api_key:
        raise RuntimeError(
            "未配置 OPENAI_API_KEY。请在 .env 中设置或使用 OpenAI 兼容网关的密钥。"
        )
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or None,
        temperature=temperature,
    )
