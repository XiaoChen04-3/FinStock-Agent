from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class FundDailyStatus(BaseModel):
    ts_code: str
    name: str
    quantity: float
    avg_cost: float
    nav_today: float | None = None
    market_value: float | None = None
    unrealized_pnl: float | None = None
    unrealized_pnl_pct: float | None = None
    today_change_pct: float | None = None
    action: str = "hold"
    confidence: float = 0.5
    reason: str = ""
    trend: str = "insufficient_data"
    analysis_summary: str = ""
    three_year_return_pct: float | None = None
    key_risks: list[str] = Field(default_factory=list)
    related_news: list[str] = Field(default_factory=list)


class MarketFundIdea(BaseModel):
    theme: str
    fund_name: str
    ts_code: str
    action: str = "watch"
    confidence: float = 0.5
    reason: str = ""
    related_news: list[str] = Field(default_factory=list)


class DailyReport(BaseModel):
    user_id: str
    report_date: str
    generated_at: datetime
    recent_trading_days: list[str]
    total_market_value: float
    total_unrealized_pnl: float
    total_unrealized_pnl_pct: float
    today_portfolio_change_pct: float
    fund_statuses: list[FundDailyStatus]
    overall_summary: str
    market_context: str
    news_sentiment_label: str
    top_news: list[dict]
    market_fund_ideas: list[MarketFundIdea] = Field(default_factory=list)
    stage1_tokens: int = 0
    stage2_tokens: int = 0
    stage3_tokens: int = 0
    total_elapsed_ms: float = 0.0
    disclaimer: str = "AI generated summary for reference only."
