from __future__ import annotations

import uuid

from fin_stock_agent.agents import router
from fin_stock_agent.core.query_enhancer import EnhancedQuery, IntentType
from fin_stock_agent.memory.portfolio_memory import PortfolioMemory, TradeRecord, get_active
from fin_stock_agent.services.portfolio_service import PortfolioService
from fin_stock_agent.storage.database import init_db


def test_build_memory_restores_persisted_trades() -> None:
    init_db()
    user_id = f"portfolio-user-{uuid.uuid4()}"
    service = PortfolioService()

    service.add_trade(
        user_id,
        TradeRecord(
            ts_code="161725.OF",
            name="Test Fund",
            direction="buy",
            quantity=100.0,
            price=1.25,
            fee=1.5,
            trade_date="20260410",
        ),
    )
    service.add_trade(
        user_id,
        TradeRecord(
            ts_code="161725.OF",
            name="Test Fund",
            direction="buy",
            quantity=50.0,
            price=1.30,
            fee=0.0,
            trade_date="20260411",
        ),
    )

    memory = service.build_memory(user_id)

    assert len(memory) == 2
    assert [trade.ts_code for trade in memory.all_trades()] == ["161725.OF", "161725.OF"]
    assert memory.all_trades()[0].trade_date == "20260411"


def test_prep_session_hydrates_empty_memory_from_database(monkeypatch) -> None:
    init_db()
    user_id = f"session-user-{uuid.uuid4()}"
    service = PortfolioService()
    service.add_trade(
        user_id,
        TradeRecord(
            ts_code="110022.OF",
            name="Persisted Fund",
            direction="buy",
            quantity=88.0,
            price=2.05,
            fee=0.0,
            trade_date="20260411",
        ),
    )

    monkeypatch.setattr(
        router,
        "enhance_query",
        lambda question, resolver=None, callbacks=None: EnhancedQuery(
            original=question,
            rewritten=question,
            intent=IntentType.PORTFOLIO_QUERY,
            complexity="simple",
        ),
    )

    empty_memory = PortfolioMemory()
    mode, messages, enhanced, _ = router._prep_session(
        "看看我的持仓",
        user_id=user_id,
        session_id="test-session",
        memory=empty_memory,
        history_messages=[],
    )

    active_memory = get_active()

    assert mode == "react"
    assert enhanced.intent == IntentType.PORTFOLIO_QUERY
    assert len(messages) == 2
    assert len(active_memory) == 1
    assert active_memory.all_trades()[0].ts_code == "110022.OF"
