import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _strip_optional_quotes(value: str | None) -> str:
    if value is None:
        return ""
    v = value.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        return v[1:-1]
    return v


class Settings:
    tushare_token: str = _strip_optional_quotes(os.getenv("TUSHARE_TOKEN"))
    openai_api_key: str = _strip_optional_quotes(os.getenv("OPENAI_API_KEY"))
    openai_base_url: str = _strip_optional_quotes(
        os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    openai_model: str = _strip_optional_quotes(
        os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )
    cache_path: Path = Path(
        _strip_optional_quotes(os.getenv("FINSTOCK_CACHE_PATH"))
        or ".finstock_cache.sqlite"
    )


settings = Settings()
