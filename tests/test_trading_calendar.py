from sqlalchemy import delete

from fin_stock_agent.init.trade_calendar import TradingCalendar
from fin_stock_agent.storage.database import get_session, init_db
from fin_stock_agent.storage.models import TradeCalendarRecord


def setup_module() -> None:
    init_db()
    with get_session() as session:
        session.execute(delete(TradeCalendarRecord))
        session.add_all(
            [
                TradeCalendarRecord(cal_date="20260409", is_open=True, exchange="SSE"),
                TradeCalendarRecord(cal_date="20260408", is_open=True, exchange="SSE"),
                TradeCalendarRecord(cal_date="20260407", is_open=True, exchange="SSE"),
                TradeCalendarRecord(cal_date="20260406", is_open=False, exchange="SSE"),
            ]
        )


def test_recent_trading_days() -> None:
    calendar = TradingCalendar()
    recent = calendar.get_recent_trading_days(3)
    assert recent[:3] == ["20260409", "20260408", "20260407"]
    assert calendar.get_latest_trading_day() == "20260409"
    assert calendar.is_trading_day("20260409") is True
    assert calendar.is_trading_day("20260406") is False
