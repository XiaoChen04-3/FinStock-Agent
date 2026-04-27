from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from fin_stock_agent.core.llm import invoke_json
from fin_stock_agent.memory.user_profile_file import (
    DEFAULT_USER_PROFILE_MD,
    UserProfileValidationError,
    get_user_profile_file_service,
)
from fin_stock_agent.prompts.memory_prompt import (
    USER_PROFILE_COMPRESSION_PROMPT,
    USER_PROFILE_EXTRACTION_PROMPT,
    USER_PROFILE_TEMPLATE,
)

logger = logging.getLogger(__name__)


class UserProfileExtractionOutput(BaseModel):
    should_update: bool = False
    reason: str = ""
    profile_md: str = Field(default_factory=lambda: DEFAULT_USER_PROFILE_MD)


class UserProfileCompressionOutput(BaseModel):
    profile_md: str


@dataclass(frozen=True)
class MemoryExtractionFailure:
    reason: str
    profile_md: str


def update_user_profile_from_turn(
    *,
    current_profile_md: str,
    question: str,
    answer: str,
    max_tokens: int,
    recent_summaries: list[str] | None = None,
    callbacks: list | None = None,
) -> UserProfileExtractionOutput:
    """使用 LLM 提取并合并可长期保留的用户画像信息。

    提取失败时只返回 ``should_update=False``，调用方跳过写入；这里不再使用规则匹配兜底。
    """
    service = get_user_profile_file_service()
    base_profile = service.normalize_markdown(current_profile_md or DEFAULT_USER_PROFILE_MD)
    recent = "\n".join(f"- {item}" for item in (recent_summaries or []) if item) or "暂无"
    prompt = USER_PROFILE_EXTRACTION_PROMPT.format(
        max_tokens=max_tokens,
        target_tokens=max(1, int(max_tokens * 0.9)),
        template=USER_PROFILE_TEMPLATE,
        current_profile_md=base_profile,
        question=question or "",
        answer=answer or "",
        recent_summaries=recent,
    )
    cfg = {"callbacks": callbacks} if callbacks else None
    try:
        payload = invoke_json(
            "memory_extractor",
            [
                SystemMessage(content="你负责维护简洁、安全、可长期保留的中文用户画像记忆。"),
                HumanMessage(content=prompt),
            ],
            config=cfg,
        )
        output = _coerce_extraction_output(payload, fallback_profile=base_profile)
        output.profile_md = service.normalize_markdown(output.profile_md or base_profile)
        if not output.should_update:
            output.profile_md = base_profile
            return output
        try:
            service.validate_markdown(output.profile_md)
        except UserProfileValidationError:
            compressed = _compress_profile(output.profile_md, max_tokens=max_tokens, callbacks=callbacks)
            service.validate_markdown(compressed)
            output.profile_md = compressed
        return output
    except Exception as exc:
        logger.warning("用户画像提取失败，保持 staged 画像不变：%s", exc)
        return UserProfileExtractionOutput(
            should_update=False,
            reason=f"画像提取失败：{type(exc).__name__}",
            profile_md=base_profile,
        )


def _coerce_extraction_output(payload: Any, *, fallback_profile: str) -> UserProfileExtractionOutput:
    if payload is None:
        return UserProfileExtractionOutput(should_update=False, reason="模型未返回画像更新", profile_md=fallback_profile)
    if isinstance(payload, UserProfileExtractionOutput):
        return payload
    if not isinstance(payload, dict):
        raise ValueError("记忆提取模型返回的 JSON 不是对象")
    try:
        return UserProfileExtractionOutput.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"记忆提取模型返回结构不符合要求：{exc}") from exc


def _compress_profile(profile_md: str, *, max_tokens: int, callbacks: list | None = None) -> str:
    service = get_user_profile_file_service()
    prompt = USER_PROFILE_COMPRESSION_PROMPT.format(max_tokens=max_tokens, profile_md=profile_md)
    cfg = {"callbacks": callbacks} if callbacks else None
    payload = invoke_json(
        "memory_extractor",
        [
            SystemMessage(content="你负责在不丢失关键事实的前提下压缩中文用户画像 Markdown。"),
            HumanMessage(content=prompt),
        ],
        config=cfg,
    )
    if not isinstance(payload, dict):
        raise ValueError("画像压缩模型返回的 JSON 不是对象")
    output = UserProfileCompressionOutput.model_validate(payload)
    return service.normalize_markdown(output.profile_md)
