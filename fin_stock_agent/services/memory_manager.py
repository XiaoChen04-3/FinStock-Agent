from __future__ import annotations

from fin_stock_agent.init.trade_calendar import TradingCalendar
from fin_stock_agent.memory.conversation import ConversationMemory
from fin_stock_agent.services.portfolio_service import PortfolioService
from fin_stock_agent.services.user_memory_service import UserMemoryService


class MemoryManager:
    def __init__(self, user_id: str, session_id: str) -> None:
        self.user_id = user_id
        self.session_id = session_id
        self.portfolio_service = PortfolioService()
        self.conversation_memory = ConversationMemory(user_id=user_id, session_id=session_id)
        self.trade_calendar = TradingCalendar()
        self.user_memory_service = UserMemoryService()

    def build_context_block(self) -> str:
        return "\n\n".join(
            [
                self.portfolio_service.build_portfolio_context(self.user_id),
                self.user_memory_service.build_profile_context(self.user_id),
                self.user_memory_service.build_recent_events_context(self.user_id),
                self.conversation_memory.build_history_context(self.user_id),
                f"## Trading calendar\nLatest trading day: {self.trade_calendar.get_latest_trading_day()}",
            ]
        )

    def after_turn(self, turn_idx: int, question: str, answer: str) -> None:
        self.conversation_memory.save_turn_summary(
            session_id=self.session_id,
            turn_idx=turn_idx,
            question=question,
            answer=answer,
        )
        self.user_memory_service.remember_turn(
            user_id=self.user_id,
            session_id=self.session_id,
            turn_idx=turn_idx,
            question=question,
            answer=answer,
        )
