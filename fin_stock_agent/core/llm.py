from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage

from fin_stock_agent.core.config import get_config
from fin_stock_agent.core.settings import settings


@dataclass(frozen=True)
class LLMProfile:
    model: str
    temperature: float = 0.2
    extra_body: dict[str, Any] = field(default_factory=dict)


DEFAULT_LLM_ROLE = "default"

_ROLE_MODEL_KEY = {
    DEFAULT_LLM_ROLE: "react_agent",
    "query_enhancer": "query_enhancer",
    "conversation_summarizer": "conversation_summarizer",
    "react": "react_agent",
    "planner": "planner",
    "replan": "replanner",
    "executor": "executor",
    "finalize": "finalizer",
    "daily_briefing": "news_filter",
    "news_analysis": "sentiment_analysis",
    "fund_analysis": "fund_trend",
    "agentic_news": "holding_correlation",
    "report_synthesis": "report_generation",
    "json_repair": "query_enhancer",
}

_ROLE_DEFAULTS: dict[str, dict[str, Any]] = {
    DEFAULT_LLM_ROLE: {"temperature": 0.2},
    "query_enhancer": {"temperature": 0.0},
    "conversation_summarizer": {"temperature": 0.0},
    "react": {"temperature": 0.2},
    "planner": {"temperature": 0.2, "extra_body": {"enable_thinking": True}},
    "replan": {"temperature": 0.2, "extra_body": {"enable_thinking": True}},
    "executor": {"temperature": 0.2},
    "finalize": {"temperature": 0.2},
    "daily_briefing": {"temperature": 0.2},
    "news_analysis": {"temperature": 0.2},
    "fund_analysis": {"temperature": 0.2},
    "agentic_news": {"temperature": 0.2},
    "report_synthesis": {"temperature": 0.2},
    "json_repair": {"temperature": 0.0},
}


def get_llm_profile(role: str = DEFAULT_LLM_ROLE) -> LLMProfile:
    if role not in _ROLE_MODEL_KEY:
        raise KeyError(f"Unknown llm role: {role}")
    model_key = _ROLE_MODEL_KEY[role]
    cfg = get_config()
    model_name = getattr(cfg.models, model_key)
    defaults = _ROLE_DEFAULTS.get(role, {})
    return LLMProfile(
        model=model_name,
        temperature=float(defaults.get("temperature", 0.2)),
        extra_body=dict(defaults.get("extra_body", {})),
    )


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


def clear_llm_cache() -> None:
    """清除 LLM 实例缓存，在配置重置后（如测试环境）调用以确保使用最新配置。"""
    _get_cached_llm.cache_clear()


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
            except json.JSONDecodeError as exc:
                logger.warning("JSON brace-extraction fallback also failed: %s | raw snippet: %.120s", exc, raw)
        start_arr = raw.find("[")
        end_arr = raw.rfind("]") + 1
        if start_arr >= 0 and end_arr > start_arr:
            return json.loads(raw[start_arr:end_arr])
        raise


def _repair_json_value(raw_text: str, *, config: dict[str, Any] | None = None) -> Any:
    prompt = (
        "你是 JSON 格式修复专家。\n"
        "任务：将用户提供的文本（可能含有 Markdown 代码块、多余说明文字或格式错误）修复为合法的 JSON，"
        "并将修复结果包裹在固定格式的外层对象中：{\"payload\": <修复后的JSON值>}。\n"
        "规则：\n"
        "1. 仅输出该 JSON 对象，严禁输出任何解释、注释或额外内容。\n"
        "2. 保持原始数据的类型和结构（对象/数组/字符串/数字）不变，不得增删字段。\n"
        "3. 若文本中包含多个 JSON 片段，以最外层或最完整的为准。\n"
        "4. 若无法识别任何有效的 JSON 结构，返回 {\"payload\": null}，不得抛出错误。"
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
    cfg = get_config()
    if agent_mode == "react":
        return f"query_enhancer={cfg.models.query_enhancer}; react={cfg.models.react_agent}"
    if agent_mode == "plan_execute":
        return (
            f"query_enhancer={cfg.models.query_enhancer}; "
            f"planner/replan={cfg.models.planner}(enable_thinking=true,streaming,json-repair); "
            f"executor/finalize={cfg.models.executor}"
        )
    return get_llm_profile(DEFAULT_LLM_ROLE).model
