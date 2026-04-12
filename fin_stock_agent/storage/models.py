from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class TradeCalendarRecord(Base):
    __tablename__ = "trade_calendar"

    cal_date = Column(String(8), primary_key=True)
    is_open = Column(Boolean, default=False, nullable=False)
    exchange = Column(String(10), default="SSE")
    updated_at = Column(DateTime, default=datetime.utcnow)


class IndexLookupRecord(Base):
    __tablename__ = "index_lookup"

    ts_code = Column(String(20), primary_key=True)
    name = Column(String(100), nullable=False)
    market = Column(String(20), nullable=True)
    category = Column(String(50), nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow)


class FundLookupRecord(Base):
    __tablename__ = "fund_lookup"

    ts_code = Column(String(20), primary_key=True)
    name = Column(String(100), nullable=False)
    fund_type = Column(String(50), nullable=True)
    status = Column(String(10), nullable=True)
    market = Column(String(10), nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow)


class TradeRecordORM(Base):
    __tablename__ = "trade_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), index=True)
    ts_code = Column(String(20), nullable=False)
    name = Column(String(100), nullable=True)
    direction = Column(String(4), nullable=False)
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    fee = Column(Float, default=0.0)
    trade_date = Column(String(8), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_deleted = Column(Boolean, default=False)


class ConversationSummaryORM(Base):
    __tablename__ = "conversation_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), index=True)
    session_id = Column(String(36), index=True)
    turn_idx = Column(Integer, nullable=False)
    question = Column(String(500), nullable=True)
    summary = Column(String(200), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class UserMemoryProfileORM(Base):
    __tablename__ = "user_memory_profiles"

    user_id = Column(String(36), primary_key=True)
    risk_level = Column(String(32), nullable=True)
    investment_horizon = Column(String(32), nullable=True)
    preferred_assets_json = Column(Text, nullable=True)
    disliked_assets_json = Column(Text, nullable=True)
    focus_themes_json = Column(Text, nullable=True)
    answer_style_json = Column(Text, nullable=True)
    decision_constraints_json = Column(Text, nullable=True)
    watchlist_json = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow)


class UserMemoryEventORM(Base):
    __tablename__ = "user_memory_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), index=True)
    session_id = Column(String(36), index=True)
    turn_idx = Column(Integer, nullable=False)
    event_type = Column(String(50), nullable=False)
    summary = Column(String(300), nullable=False)
    payload_json = Column(Text, nullable=True)
    confidence = Column(Float, default=0.0)
    source_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class StatRecordORM(Base):
    __tablename__ = "stat_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), index=True)
    query_text = Column(String(500), nullable=True)
    intent = Column(String(50), nullable=True)
    agent_mode = Column(String(20), nullable=True)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    elapsed_ms = Column(Float, default=0.0)
    llm_elapsed_ms = Column(Float, default=0.0)
    tool_call_count = Column(Integer, default=0)
    tool_names_called = Column(Text, nullable=True)
    model_name = Column(String(100), nullable=True)
    has_error = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class NewsCacheORM(Base):
    __tablename__ = "news_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    url = Column(String(1000), unique=True)
    title = Column(String(500))
    summary = Column(Text, nullable=True)
    source = Column(String(20), nullable=True)
    published_at = Column(DateTime, nullable=True)
    fetched_at = Column(DateTime, default=datetime.utcnow)


class DailyReportORM(Base):
    __tablename__ = "daily_reports"
    __table_args__ = (UniqueConstraint("user_id", "report_date", name="uq_user_report_date"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), index=True)
    report_date = Column(String(10), nullable=False)
    report_json = Column(Text, nullable=False)
    model_name = Column(String(100), nullable=True)
    stage1_tokens = Column(Integer, default=0)
    stage2_tokens = Column(Integer, default=0)
    stage3_tokens = Column(Integer, default=0)
    elapsed_ms = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
