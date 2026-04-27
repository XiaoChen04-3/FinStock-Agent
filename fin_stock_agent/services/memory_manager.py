from __future__ import annotations

import logging

from fin_stock_agent.core.config import get_config
from fin_stock_agent.init.trade_calendar import TradingCalendar
from fin_stock_agent.memory.conversation import ConversationMemory
from fin_stock_agent.services.daily_report_digest_service import DailyReportDigestService
from fin_stock_agent.services.portfolio_service import PortfolioService
from fin_stock_agent.services.user_memory_service import UserMemoryService

logger = logging.getLogger(__name__)


class MemoryManager:
    def __init__(self, user_id: str, session_id: str) -> None:
        self.user_id = user_id
        self.session_id = session_id
        self.portfolio_service = PortfolioService()
        self.conversation_memory = ConversationMemory(user_id=user_id, session_id=session_id)
        self.trade_calendar = TradingCalendar()
        self.user_memory_service = UserMemoryService()
        self.digest_service = DailyReportDigestService()

    def build_context_block(self, question: str = "") -> str:
        """组装注入每个 system prompt 的记忆上下文。

        用户画像层使用文件存储。单个进程生命周期内注入的是启动时快照；
        每轮对话后的画像提取只更新 pending Markdown，并在关闭时提交。
        """
        digest_block = self._build_digest_block(question)
        history_block = self._build_history_block(question)
        return "\n\n".join(
            [
                self._full_portfolio_context(),
                self._full_profile_context(),
                digest_block,
                history_block,
                f"## 交易日历\n最新交易日：{self.trade_calendar.get_latest_trading_day()}",
            ]
        )

    def build_prompt_memory_block(self, question: str = "") -> str:
        cfg = get_config()
        max_chars = min(cfg.memory.prompt_memory_max_chars, cfg.plan_execute.context_max_chars)
        sections = [
            ("portfolio", self._prompt_portfolio_context()),
            ("profile_core", self.user_memory_service.build_prompt_profile_context(self.user_id)),
            ("profile_extra", ""),
            ("history", self._prompt_history_context(question)),
        ]
        return self._trim_sections(sections, max_chars)

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

    def _full_portfolio_context(self) -> str:
        context = self.portfolio_service.build_portfolio_context(self.user_id)
        if "暂无持仓记录" in context or "æš‚æ— æŒä»“è®°å½•" in context:
            return "## 当前持仓\n当前用户暂无持仓，请从全市场视角进行分析。"
        return context

    def _full_profile_context(self) -> str:
        return self.user_memory_service.build_profile_context(self.user_id)

    def _build_digest_block(self, question: str) -> str:
        digests = self.digest_service.search_relevant_digests(self.user_id, question) if question else []
        if not digests:
            return self.digest_service.build_digest_context(self.user_id)
        lines = ["## 近期日报摘要"]
        for item in digests:
            lines.append(f"- {item}")
        return "\n".join(lines)

    def _build_history_block(self, question: str) -> str:
        summaries = self.conversation_memory.search_relevant_summaries(self.user_id, question) if question else []
        if not summaries:
            return self.conversation_memory.build_history_context(self.user_id)
        lines = ["## 近期对话摘要"]
        for item in summaries:
            lines.append(f"- {item}")
        return "\n".join(lines)

    def _prompt_portfolio_context(self) -> str:
        holdings = self.portfolio_service.get_holdings(self.user_id)
        if not holdings:
            return "## 持仓摘要\n当前用户暂无持仓。"
        lines = ["## 持仓摘要"]
        for row in holdings[:8]:
            pnl = row.get("unrealized_pnl")
            direction = "盈利" if (pnl or 0.0) >= 0 else "亏损"
            lines.append(
                f"- {row['name']}({row['ts_code']}): 份额={row['quantity']}, 均价={row['avg_cost']:.4f}, "
                f"盈亏方向={direction}"
            )
        return "\n".join(lines)

    def _prompt_history_context(self, question: str) -> str:
        summaries = self.conversation_memory.search_relevant_summaries(self.user_id, question) if question else []
        if not summaries:
            summaries = self.conversation_memory.get_recent_summaries(self.user_id, limit=3)
        if not summaries:
            return "## 近期摘要\n暂无近期对话摘要。"
        lines = ["## 近期摘要"]
        for item in summaries[:3]:
            lines.append(f"- {item}")
        return "\n".join(lines)

    def _trim_sections(self, sections: list[tuple[str, str]], max_chars: int) -> str:
        kept = [(name, text.strip()) for name, text in sections if text.strip()]
        if not kept:
            return ""
        assembled = "\n\n".join(text for _, text in kept)
        if len(assembled) <= max_chars:
            return assembled
        drop_order = ["history", "profile_extra", "profile_core", "portfolio"]
        working = list(kept)
        while len("\n\n".join(text for _, text in working)) > max_chars and working:
            if not drop_order:
                break
            name = drop_order[0]
            for index in range(len(working) - 1, -1, -1):
                if working[index][0] == name:
                    working.pop(index)
                    break
            else:
                drop_order.pop(0)
        result = "\n\n".join(text for _, text in working)
        if len(result) > max_chars:
            logger.warning("Prompt 记忆上下文超过字符预算，已截断")
            return result[:max_chars]
        return result
