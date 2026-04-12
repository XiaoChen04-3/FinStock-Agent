from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage

from fin_stock_agent.core.settings import settings


@dataclass(frozen=True)
class LLMProfile:
    model: str
    temperature: float = 0.2
    extra_body: dict[str, Any] = field(default_factory=dict)


DEFAULT_LLM_ROLE = "default"

LLM_ROLE_PROFILES: dict[str, LLMProfile] = {
    DEFAULT_LLM_ROLE: LLMProfile(model=settings.openai_model, temperature=0.2),
    "query_enhancer": LLMProfile(model="qwen3.5-flash", temperature=0.0),
    "react": LLMProfile(model="qwen3.6-plus", temperature=0.2),
    "planner": LLMProfile(
        model="qwen3-max",
        temperature=0.2,
        extra_body={"enable_thinking": True},
    ),
    "replan": LLMProfile(
        model="qwen3-max",
        temperature=0.2,
        extra_body={"enable_thinking": True},
    ),
    "executor": LLMProfile(model="qwen3.6-plus", temperature=0.2),
    "finalize": LLMProfile(model="qwen3.6-plus", temperature=0.2),
    "daily_briefing": LLMProfile(model="qwen3.5-flash", temperature=0.2),
    "news_analysis": LLMProfile(model="qwen3.6-plus", temperature=0.2),
    "fund_analysis": LLMProfile(model="qwen3.5-flash", temperature=0.2),
    "agentic_news": LLMProfile(model="qwen3.6-plus", temperature=0.2),
    "report_synthesis": LLMProfile(model="qwen3.6-plus", temperature=0.2),
    "json_repair": LLMProfile(model="qwen3.5-flash", temperature=0.0),
}


def get_llm_profile(role: str = DEFAULT_LLM_ROLE) -> LLMProfile:
    if role not in LLM_ROLE_PROFILES:
        raise KeyError(f"Unknown llm role: {role}")
    return LLM_ROLE_PROFILES[role]


def role_uses_thinking(role: str) -> bool:
    return bool(get_llm_profile(role).extra_body.get("enable_thinking"))


def build_llm_kwargs(role: str = DEFAULT_LLM_ROLE, *, temperature: float | None = None) -> dict[str, Any]:
    profile = get_llm_profile(role)
    kwargs: dict[str, Any] = {
        "model": profile.model,
        "api_key": settings.openai_api_key,
        "base_url": settings.openai_base_url or None,
        "temperature": profile.temperature if temperature is None else temperature,
        "stream_usage": True,
    }
    if profile.extra_body:
        kwargs["extra_body"] = dict(profile.extra_body)
    return kwargs


@lru_cache(maxsize=32)
def _get_cached_llm(role: str, temperature_key: str):
    from langchain_openai import ChatOpenAI

    temperature = None if temperature_key == "__default__" else float(temperature_key)
    return ChatOpenAI(**build_llm_kwargs(role, temperature=temperature))


def get_llm(role: str = DEFAULT_LLM_ROLE, *, temperature: float | None = None):
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured in the environment.")
    temperature_key = "__default__" if temperature is None else repr(float(temperature))
    return _get_cached_llm(role, temperature_key)


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return ""


def _response_text(message: Any) -> str:
    if isinstance(message, (AIMessage, AIMessageChunk)):
        return _extract_text(message.content)
    return _extract_text(getattr(message, "content", ""))


def _usage_from_one(message: Any) -> dict[str, int]:
    usage = getattr(message, "usage_metadata", None) or {}
    if usage:
        prompt_tokens = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
        completion_tokens = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    metadata = getattr(message, "response_metadata", None) or {}
    token_usage = metadata.get("token_usage") or metadata.get("usage") or {}
    prompt_tokens = int(token_usage.get("prompt_tokens", token_usage.get("input_tokens", 0)) or 0)
    completion_tokens = int(token_usage.get("completion_tokens", token_usage.get("output_tokens", 0)) or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def merge_token_usage(*items: Any) -> dict[str, int]:
    merged = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for item in items:
        if item is None:
            continue
        if isinstance(item, dict) and {"prompt_tokens", "completion_tokens", "total_tokens"} & set(item.keys()):
            merged["prompt_tokens"] += int(item.get("prompt_tokens", 0) or 0)
            merged["completion_tokens"] += int(item.get("completion_tokens", 0) or 0)
            merged["total_tokens"] += int(item.get("total_tokens", 0) or 0)
            continue
        if isinstance(item, dict):
            nested = item.get("messages")
            if nested is not None:
                nested_usage = merge_token_usage(nested)
                merged["prompt_tokens"] += nested_usage["prompt_tokens"]
                merged["completion_tokens"] += nested_usage["completion_tokens"]
                merged["total_tokens"] += nested_usage["total_tokens"]
            continue
        if isinstance(item, list):
            nested_usage = merge_token_usage(*item)
            merged["prompt_tokens"] += nested_usage["prompt_tokens"]
            merged["completion_tokens"] += nested_usage["completion_tokens"]
            merged["total_tokens"] += nested_usage["total_tokens"]
            continue

        usage = _usage_from_one(item)
        merged["prompt_tokens"] += usage["prompt_tokens"]
        merged["completion_tokens"] += usage["completion_tokens"]
        merged["total_tokens"] += usage["total_tokens"]
    return merged


def invoke_text(role: str, messages: list[BaseMessage], *, config: dict[str, Any] | None = None) -> str:
    llm = get_llm(role)
    if role_uses_thinking(role):
        parts: list[str] = []
        for chunk in llm.stream(messages, config=config):
            text = _response_text(chunk)
            if text:
                parts.append(text)
        return "".join(parts).strip()
    response = llm.invoke(messages, config=config)
    return _response_text(response).strip()


def _strip_json_wrappers(text: str) -> str:
    raw = re.sub(r"```(?:json)?", "", text or "").strip()
    return raw


def _parse_json_value(text: str) -> Any:
    raw = _strip_json_wrappers(text)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start_obj = raw.find("{")
        end_obj = raw.rfind("}") + 1
        if start_obj >= 0 and end_obj > start_obj:
            try:
                return json.loads(raw[start_obj:end_obj])
            except json.JSONDecodeError:
                pass
        start_arr = raw.find("[")
        end_arr = raw.rfind("]") + 1
        if start_arr >= 0 and end_arr > start_arr:
            return json.loads(raw[start_arr:end_arr])
        raise


def _repair_json_value(raw_text: str, *, config: dict[str, Any] | None = None) -> Any:
    prompt = (
        "你是一个 JSON 格式修复专家。"
        "请将用户提供的内容修复为标准 JSON，"
        "并包裹在一个对象里，格式固定为 {\"payload\": <修复后的JSON值>}。"
        "只输出该 JSON 对象，不要输出解释。"
    )
    runnable = get_llm("json_repair").bind(response_format={"type": "json_object"})
    response = runnable.invoke(
        [
            HumanMessage(content=prompt),
            HumanMessage(content=raw_text),
        ],
        config=config,
    )
    repaired = _parse_json_value(_response_text(response))
    if isinstance(repaired, dict) and "payload" in repaired:
        return repaired["payload"]
    return repaired


def invoke_json(role: str, messages: list[BaseMessage], *, config: dict[str, Any] | None = None) -> Any:
    raw_text = invoke_text(role, messages, config=config)
    try:
        return _parse_json_value(raw_text)
    except json.JSONDecodeError:
        return _repair_json_value(raw_text, config=config)


def describe_agent_chain(agent_mode: str) -> str:
    if agent_mode == "react":
        return "query_enhancer=qwen3.5-flash; react=qwen3.6-plus"
    if agent_mode == "plan_execute":
        return (
            "query_enhancer=qwen3.5-flash; planner/replan=qwen3-max(enable_thinking=true,streaming,json-repair); "
            "executor/finalize=qwen3.6-plus"
        )
    return get_llm_profile(DEFAULT_LLM_ROLE).model
