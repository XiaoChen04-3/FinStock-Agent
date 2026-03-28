from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from typing import Any

import pandas as pd
import tushare as ts
from tenacity import retry, stop_after_attempt, wait_exponential

from fin_stock_agent.core.exceptions import TushareRequestError
from fin_stock_agent.core.settings import settings


class TushareClient:
    """Thin Tushare pro_api wrapper: throttle, retries, optional SQLite cache."""

    def __init__(
        self,
        token: str | None = None,
        *,
        min_interval_sec: float = 0.35,
        cache_enabled: bool = True,
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
        self._cache_enabled = cache_enabled
        self._cache_path = settings.cache_path

    def _throttle(self) -> None:
        with self._lock:
            now = time.time()
            wait = self._min_interval - (now - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()

    def _cache_key(self, method: str, kwargs: dict[str, Any]) -> str:
        payload = json.dumps({"m": method, "k": kwargs}, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> pd.DataFrame | None:
        if not self._cache_enabled:
            return None
        try:
            conn = sqlite3.connect(self._cache_path)
            cur = conn.execute(
                "SELECT payload FROM finstock_cache WHERE cache_key = ?", (key,)
            )
            row = cur.fetchone()
            conn.close()
            if not row:
                return None
            return pd.DataFrame(json.loads(row[0]))
        except Exception:
            return None

    def _cache_set(self, key: str, df: pd.DataFrame) -> None:
        if not self._cache_enabled or df is None:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self._cache_path)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS finstock_cache (
                    cache_key TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    created REAL NOT NULL
                )
                """
            )
            payload = df.to_json(orient="records", date_format="iso")
            conn.execute(
                "INSERT OR REPLACE INTO finstock_cache(cache_key, payload, created) VALUES (?,?,?)",
                (key, payload, time.time()),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        reraise=True,
    )
    def call(self, method: str, *, use_cache: bool = False, **kwargs: Any) -> pd.DataFrame:
        key = self._cache_key(method, kwargs)
        if use_cache:
            hit = self._cache_get(key)
            if hit is not None and not hit.empty:
                return hit

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

        if df is None:
            df = pd.DataFrame()
        if use_cache and not df.empty:
            self._cache_set(key, df)
        return df


_client_singleton: TushareClient | None = None
_client_lock = threading.Lock()


def get_client() -> TushareClient:
    global _client_singleton
    if _client_singleton is None:
        with _client_lock:
            if _client_singleton is None:
                _client_singleton = TushareClient()
    return _client_singleton
