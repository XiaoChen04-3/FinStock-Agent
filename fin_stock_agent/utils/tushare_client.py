from __future__ import annotations

import threading
import time
from typing import Any

import pandas as pd
import tushare as ts
from tenacity import retry, stop_after_attempt, wait_exponential

from fin_stock_agent.core.exceptions import TushareRequestError
from fin_stock_agent.core.settings import settings


class TushareClient:
    """Thin Tushare pro_api wrapper: rate-limiting and automatic retries."""

    def __init__(
        self,
        token: str | None = None,
        *,
        min_interval_sec: float = 0.35,
    ) -> None:
        tok = (token or settings.tushare_token or "").strip()
        if not tok:
            raise TushareRequestError(
                "未配置 TUSHARE_TOKEN。请在 .env 中设置或传入 token。"
            )
        self._pro = ts.pro_api(tok)
        self._min_interval = min_interval_sec
        self._last_call = 0.0
        self._lock = threading.Lock()

    def _throttle(self) -> None:
        with self._lock:
            now = time.time()
            wait = self._min_interval - (now - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        reraise=True,
    )
    def call(self, method: str, *, use_cache: bool = False, **kwargs: Any) -> pd.DataFrame:
        """Call a Tushare pro API method.

        ``use_cache`` is accepted for backward compatibility but has no effect.
        """
        self._throttle()
        try:
            fn = getattr(self._pro, method, None)
            if fn is None:
                raise TushareRequestError(f"未知的 Tushare 接口: {method}")
            df = fn(**kwargs)
        except TushareRequestError:
            raise
        except Exception as e:
            msg = str(e)
            if "积分" in msg or "permission" in msg.lower():
                msg = f"Tushare 权限或积分不足: {msg}"
            raise TushareRequestError(msg) from e

        return df if df is not None else pd.DataFrame()


_client_singleton: TushareClient | None = None
_client_lock = threading.Lock()


def get_client() -> TushareClient:
    global _client_singleton
    if _client_singleton is None:
        with _client_lock:
            if _client_singleton is None:
                _client_singleton = TushareClient()
    return _client_singleton
