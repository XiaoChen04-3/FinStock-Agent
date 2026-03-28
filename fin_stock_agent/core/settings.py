from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _strip(value: str | None) -> str:
    if value is None:
        return ""
    v = value.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        return v[1:-1]
    return v


class Settings:
    def __init__(self) -> None:
        self.tushare_token: str = _strip(os.getenv("TUSHARE_TOKEN"))
        self.openai_api_key: str = _strip(os.getenv("OPENAI_API_KEY"))
        self.openai_base_url: str = _strip(
            os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )
        self.openai_model: str = _strip(os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        self.cache_path: Path = Path(
            _strip(os.getenv("FINSTOCK_CACHE_PATH")) or ".finstock_cache.sqlite"
        )


settings = Settings()
