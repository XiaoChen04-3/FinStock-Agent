from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class NewsItem(BaseModel):
    title: str
    summary: str = ""
    url: str
    source: str
    published_at: datetime | None = None


class NewsFetchResult(BaseModel):
    items: list[NewsItem]
    fetched_sources: list[str]
    degraded: bool = False
    message: str = ""
