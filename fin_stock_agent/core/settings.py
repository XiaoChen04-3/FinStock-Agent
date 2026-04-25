from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv() -> bool:
        return False

load_dotenv()


def _strip(value: str | None) -> str:
    if value is None:
        return ""
    v = value.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        return v[1:-1]
    return v


def _bool(name: str, default: bool) -> bool:
    raw = _strip(os.getenv(name))
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


class Settings:
    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parents[2]
        self.env_file = self.project_root / ".env"
        self.data_dir = self.project_root / ".data"
        self.log_dir = self.project_root / "logs"
        self.data_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)

        self.tushare_token: str = _strip(os.getenv("TUSHARE_TOKEN"))
        self.openai_api_key: str = _strip(os.getenv("OPENAI_API_KEY"))
        self.openai_base_url: str = _strip(
            os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )
        self.openai_model: str = _strip(os.getenv("OPENAI_MODEL"))

        default_db_name = "finstock_test.db" if self._is_pytest_runtime() else "finstock.db"
        self.database_url: str = _strip(
            os.getenv(
                "DATABASE_URL",
                f"sqlite:///{(self.data_dir / default_db_name).as_posix()}",
            )
        )
        self.redis_url: str = _strip(os.getenv("REDIS_URL", "fakeredis://local"))
        self.app_timezone: str = _strip(os.getenv("APP_TIMEZONE", "Asia/Shanghai"))
        self.enable_news_fetch: bool = _bool("ENABLE_NEWS_FETCH", True)
        self.user_id_seed: str = _strip(os.getenv("USER_ID_SEED", "finstock-agent"))

    def is_configured(self) -> bool:
        return bool(self.openai_api_key and self.tushare_token)

    @staticmethod
    def _is_pytest_runtime() -> bool:
        return "pytest" in sys.modules or bool(os.getenv("PYTEST_CURRENT_TEST"))


settings = Settings()
