from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, ValidationError

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


class ConfigValidationError(RuntimeError):
    """Raised when the YAML config is missing or invalid."""


class ModelsConfig(BaseModel):
    query_enhancer: str
    conversation_summarizer: str
    react_agent: str
    planner: str
    replanner: str
    executor: str
    finalizer: str
    news_filter: str
    sentiment_analysis: str
    fund_trend: str
    holding_correlation: str
    report_generation: str
    embedding: str


class SemanticSearchConfig(BaseModel):
    conversation_top_k: int
    digest_top_k: int
    time_fallback_limit: int
    similarity_threshold: float


class PlanLibraryConfig(BaseModel):
    reuse_threshold: float
    reference_threshold: float
    min_quality_score: float
    max_records_per_user: int


class VectorStoreConfig(BaseModel):
    backend: Literal["chromadb", "sqlite-vec"]
    max_total_records: int


class MemoryConfig(BaseModel):
    prompt_memory_max_chars: int
    semantic_search: SemanticSearchConfig
    plan_library: PlanLibraryConfig
    vector_store: VectorStoreConfig


class DailyReportConfig(BaseModel):
    news_fetch_limit: int
    briefing_top_n: int
    personalized_news_top_n: int
    nav_history_years: int
    recent_trading_days: int
    cache_ttl_hours: int
    report_summary_max_chars: int


class PlanExecuteConfig(BaseModel):
    max_plan_steps: int
    min_plan_steps: int
    max_errors_before_fallback: int
    context_max_chars: int


class TestApiConfig(BaseModel):
    host: str
    port: int


class ConcurrencyConfig(BaseModel):
    post_turn_workers: int
    daily_report_workers: int


class AppConfig(BaseModel):
    models: ModelsConfig
    memory: MemoryConfig
    daily_report: DailyReportConfig
    plan_execute: PlanExecuteConfig
    test_api: TestApiConfig
    concurrency: ConcurrencyConfig

    _instance: ClassVar["AppConfig | None"] = None
    _path: ClassVar[Path | None] = None

    @classmethod
    def default_path(cls) -> Path:
        return Path(__file__).resolve().parents[2] / "config.yaml"

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str] | None = None,
        *,
        exit_on_error: bool = True,
        force_reload: bool = False,
    ) -> "AppConfig":
        if cls._instance is not None and not force_reload:
            return cls._instance

        try:
            config_path = cls._resolve_path(path)
            raw_dict = cls._read_yaml(config_path)
            cls._apply_env_overrides(raw_dict)
            config = cls.model_validate(raw_dict)
            cls._validate_business_rules(config)
        except (ConfigValidationError, ValidationError) as exc:
            error = cls._normalize_error(exc)
            if exit_on_error:
                print(error, file=sys.stderr)
                raise SystemExit(1) from exc
            raise ConfigValidationError(error) from exc

        cls._instance = config
        cls._path = config_path
        return config

    @classmethod
    def get(cls) -> "AppConfig":
        return cls._instance or cls.load()

    @classmethod
    def reset_for_tests(cls) -> None:
        cls._instance = None
        cls._path = None
        try:
            from fin_stock_agent.core.llm import clear_llm_cache
            clear_llm_cache()
        except Exception:
            pass

    @classmethod
    def _resolve_path(cls, path: str | os.PathLike[str] | None) -> Path:
        raw = os.getenv("FINSTOCK_CONFIG") or path
        return Path(raw) if raw else cls.default_path()

    @classmethod
    def _read_yaml(cls, path: Path) -> dict[str, Any]:
        if yaml is None:
            raise ConfigValidationError("ConfigValidationError: pyyaml is required to load config.yaml")
        if not path.exists():
            raise ConfigValidationError(f"ConfigValidationError: config file not found: {path}")
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # pragma: no cover - parser-specific
            raise ConfigValidationError(f"ConfigValidationError: failed to parse {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigValidationError(f"ConfigValidationError: root of {path} must be a mapping")
        return raw

    @classmethod
    def _apply_env_overrides(cls, payload: dict[str, Any], prefix: str = "FINSTOCK", parents: tuple[str, ...] = ()) -> None:
        for key, value in list(payload.items()):
            path = parents + (key,)
            env_name = prefix + "_" + "_".join(part.upper() for part in path)
            if isinstance(value, dict):
                cls._apply_env_overrides(value, prefix=prefix, parents=path)
                continue
            env_value = os.getenv(env_name)
            if env_value is not None:
                payload[key] = env_value

    @classmethod
    def _validate_business_rules(cls, config: "AppConfig") -> None:
        positive_paths = {
            "memory.prompt_memory_max_chars": config.memory.prompt_memory_max_chars,
            "memory.semantic_search.conversation_top_k": config.memory.semantic_search.conversation_top_k,
            "memory.semantic_search.digest_top_k": config.memory.semantic_search.digest_top_k,
            "memory.semantic_search.time_fallback_limit": config.memory.semantic_search.time_fallback_limit,
            "memory.plan_library.max_records_per_user": config.memory.plan_library.max_records_per_user,
            "memory.vector_store.max_total_records": config.memory.vector_store.max_total_records,
            "daily_report.news_fetch_limit": config.daily_report.news_fetch_limit,
            "daily_report.briefing_top_n": config.daily_report.briefing_top_n,
            "daily_report.personalized_news_top_n": config.daily_report.personalized_news_top_n,
            "daily_report.nav_history_years": config.daily_report.nav_history_years,
            "daily_report.recent_trading_days": config.daily_report.recent_trading_days,
            "daily_report.cache_ttl_hours": config.daily_report.cache_ttl_hours,
            "daily_report.report_summary_max_chars": config.daily_report.report_summary_max_chars,
            "plan_execute.max_plan_steps": config.plan_execute.max_plan_steps,
            "plan_execute.min_plan_steps": config.plan_execute.min_plan_steps,
            "plan_execute.max_errors_before_fallback": config.plan_execute.max_errors_before_fallback,
            "plan_execute.context_max_chars": config.plan_execute.context_max_chars,
            "concurrency.post_turn_workers": config.concurrency.post_turn_workers,
            "concurrency.daily_report_workers": config.concurrency.daily_report_workers,
        }
        for path, value in positive_paths.items():
            if value <= 0:
                raise ConfigValidationError(f"ConfigValidationError: {path} = {value}, expected > 0")

        threshold_paths = {
            "memory.semantic_search.similarity_threshold": config.memory.semantic_search.similarity_threshold,
            "memory.plan_library.reuse_threshold": config.memory.plan_library.reuse_threshold,
            "memory.plan_library.reference_threshold": config.memory.plan_library.reference_threshold,
            "memory.plan_library.min_quality_score": config.memory.plan_library.min_quality_score,
        }
        for path, value in threshold_paths.items():
            if not 0.0 <= value <= 1.0:
                raise ConfigValidationError(f"ConfigValidationError: {path} = {value}, expected in [0, 1]")

        if config.plan_execute.min_plan_steps > config.plan_execute.max_plan_steps:
            raise ConfigValidationError(
                "ConfigValidationError: plan_execute.min_plan_steps must be <= plan_execute.max_plan_steps"
            )
        if not 1024 <= config.test_api.port <= 65535:
            raise ConfigValidationError(
                f"ConfigValidationError: test_api.port = {config.test_api.port}, expected in [1024, 65535]"
            )

    @classmethod
    def _normalize_error(cls, exc: ValidationError | ConfigValidationError) -> str:
        if isinstance(exc, ConfigValidationError):
            return str(exc)
        first = exc.errors()[0] if exc.errors() else {"loc": ("unknown",), "msg": str(exc)}
        loc = ".".join(str(part) for part in first.get("loc", ("unknown",)))
        return f"ConfigValidationError: {loc} - {first.get('msg', 'invalid value')}"


def get_config() -> AppConfig:
    return AppConfig.get()
