from __future__ import annotations

from fin_stock_agent.core.settings import settings


def local_profile_id() -> str:
    return settings.user_id_seed or "finstock-agent"
