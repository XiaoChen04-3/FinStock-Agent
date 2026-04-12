from __future__ import annotations

_cache = None


class _MemoryCache:
    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def get(self, key: str):
        return self._data.get(key)

    def setex(self, key: str, ttl: int, value):
        self._data[key] = value

    def delete(self, key: str) -> None:
        self._data.pop(key, None)


def get_cache():
    global _cache
    if _cache is not None:
        return _cache

    from fin_stock_agent.core.settings import settings

    if settings.redis_url.startswith("fakeredis://"):
        try:
            import fakeredis

            _cache = fakeredis.FakeStrictRedis(decode_responses=True)
        except Exception:
            _cache = _MemoryCache()
        return _cache

    try:
        import redis

        client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        client.ping()
        _cache = client
    except Exception:
        try:
            import fakeredis

            _cache = fakeredis.FakeStrictRedis(decode_responses=True)
        except Exception:
            _cache = _MemoryCache()
    return _cache
