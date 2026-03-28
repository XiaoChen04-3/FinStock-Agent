from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from fin_stock_agent.agents.router import stream_agent
from fin_stock_agent.core.settings import settings
from fin_stock_agent.core.tushare_permissions import describe_available_tiers
from fin_stock_agent.memory.conversation import ConversationMemory
from fin_stock_agent.memory.portfolio_memory import PortfolioMemory

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _get_portfolio_memory() -> PortfolioMemory:
    if "portfolio_memory" not in st.session_state:
        st.session_state["portfolio_memory"] = PortfolioMemory()
    return st.session_state["portfolio_memory"]


def _get_conversation_memory() -> ConversationMemory:
    if "conversation_memory" not in st.session_state:
        st.session_state["conversation_memory"] = ConversationMemory()
    return st.session_state["conversation_memory"]


def _get_tushare_score() -> int:
    """Return the user-configured Tushare score (0 = unrestricted)."""
    return int(st.session_state.get("tushare_score", 0) or 0)


def _get_thinking_history() -> list[list[str]]:
    """
    Returns a per-assistant-turn list of thinking lines.
    Index i corresponds to the i-th assistant message in ConversationMemory.
    """
    if "thinking_history" not in st.session_state:
        st.session_state["thinking_history"] = []
    return st.session_state["thinking_history"]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="FinStock-Agent", page_icon="📈", layout="wide")
    st.title("📈 FinStock-Agent")
    st.caption("LangGraph · Tushare · OpenAI 兼容 API  |  数据仅供学习研究，不构成投资建议")

    # Environment checks
    if not settings.tushare_token:
        st.warning("⚠️ 未检测到 TUSHARE_TOKEN，市场数据工具将无法使用。请在项目根目录配置 .env。")
    if not settings.openai_api_key:
        st.error("❌ 未检测到 OPENAI_API_KEY，无法调用大模型。请配置 .env 后刷新页面。")
        st.stop()

    pm = _get_portfolio_memory()
    cm = _get_conversation_memory()
    thinking_hist = _get_thinking_history()

    # ---------------------------------------------------------------------------
    # Sidebar
    # ---------------------------------------------------------------------------
    with st.sidebar:
        st.header("📋 持仓记忆")
        st.caption("直接在对话中告诉我你买了/卖了哪些股票，我会自动记录。")

        df_display = pm.to_dataframe()
        if df_display.empty:
            st.info("暂无持仓记录。")
        else:
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            st.caption(f"共 {len(pm)} 条交易记录")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ 清空持仓", use_container_width=True):
                pm.clear()
                st.rerun()
        with col2:
            if st.button("💬 清空对话", use_container_width=True):
                cm.clear()
                st.session_state["thinking_history"] = []
                st.rerun()

        # ── Tushare 积分配置 ─────────────────────────────────────────────────
        st.divider()
        st.subheader("🔑 Tushare 积分")
        score_input = st.number_input(
            "输入你的 Tushare 积分（选填，0 表示不限制）",
            min_value=0,
            max_value=99999,
            value=_get_tushare_score(),
            step=1000,
            help=(
                "根据积分自动过滤可用工具，避免调用无权限接口。\n"
                "120免费 | 2000基础 | 5000资金流向 | 6000ETF/基金 | 8000港股 | 10000美股"
            ),
            key="tushare_score",
        )
        if score_input > 0:
            st.caption(f"当前解锁：{describe_available_tiers(score_input)}")
        else:
            st.caption("未配置积分，默认开放全部工具。")

    # ---------------------------------------------------------------------------
    # Chat history display (with collapsible thinking for each assistant turn)
    # ---------------------------------------------------------------------------
    if len(cm) == 0:
        # seed the welcome message; it has no thinking history
        cm.add_assistant(
            "你好！我是 FinStock-Agent。\n\n"
            "你可以直接告诉我你买了或卖了哪些股票，我会自动记录到持仓记忆中。"
            "也可以问我 A 股行情、指数走势、板块涨跌，或让我计算你的持仓盈亏。\n\n"
            "股票代码请用 Tushare 格式，例如：600519.SH（茅台）、000001.SZ（平安银行）。"
        )
        # Seed an empty placeholder so the welcome message's index aligns
        if not thinking_hist:
            thinking_hist.append([])

    assistant_idx = 0
    for row in cm.all_rows():
        with st.chat_message(row["role"]):
            if row["role"] == "assistant":
                # Show collapsible thinking for this turn (if any was recorded)
                if assistant_idx < len(thinking_hist) and thinking_hist[assistant_idx]:
                    with st.expander("💭 查看思考过程", expanded=False):
                        st.markdown("\n\n".join(thinking_hist[assistant_idx]))
                st.markdown(row["content"])
                assistant_idx += 1
            else:
                st.markdown(row["content"])

    # ---------------------------------------------------------------------------
    # User input
    # ---------------------------------------------------------------------------
    prompt = st.chat_input("输入问题，例如：我昨天买了100股茅台600519.SH，价格1688元…")
    if not prompt:
        return

    cm.add_user(prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    # ---------------------------------------------------------------------------
    # Streaming agent invocation
    # ---------------------------------------------------------------------------
    history = cm.to_lc_messages()[:-1]   # all messages except the latest user one
    score = _get_tushare_score()

    with st.chat_message("assistant"):
        # ── Thinking process area (live, stays expanded after completion) ─────
        status_box = st.status("思考中…", expanded=True)
        thinking_lines: list[str] = []
        thinking_placeholder = status_box.empty()
        # Tracks whether the last appended line is an "⏳ pending tool" marker
        has_pending_tool: bool = False

        # ── Final answer area ─────────────────────────────────────────────────
        answer_placeholder = st.empty()
        answer_tokens: list[str] = []
        answer: str = ""

        def _render_thinking() -> None:
            thinking_placeholder.markdown("\n\n".join(thinking_lines))

        try:
            for ev_type, content in stream_agent(
                prompt,
                memory=pm,
                history_messages=history,
                tushare_score=score,
            ):
                # ── Agent mode label ─────────────────────────────────────────
                if ev_type == "mode":
                    status_box.update(
                        label=f"🤖 {content} 模式 — 思考中…", expanded=True
                    )

                # ── Generic thinking (intent recognition / plan progress) ────
                elif ev_type == "thinking":
                    thinking_lines.append(f"💭 {content}")
                    _render_thinking()

                # ── Tool call STARTED (shows ⏳ before result arrives) ────────
                elif ev_type == "tool_start":
                    # If a previous pending marker was never resolved (edge-case),
                    # pop it first so we don't stack multiple ⏳ lines.
                    if has_pending_tool and thinking_lines:
                        thinking_lines.pop()
                    thinking_lines.append(f"⏳ **{content}** — 调用中…")
                    has_pending_tool = True
                    _render_thinking()

                # ── Tool call COMPLETED: replace ⏳ with full card ────────────
                elif ev_type == "tool_interaction":
                    if has_pending_tool and thinking_lines:
                        thinking_lines.pop()
                        has_pending_tool = False
                    try:
                        info = json.loads(content)
                        name = info.get("name", "")
                        desc = info.get("description", "")
                        summary = info.get("summary", "")
                        desc_part = f" · *{desc}*" if desc else ""
                        thinking_lines.append(
                            f"🔧 **{name}**{desc_part}\n\n"
                            f"> 📊 {summary}"
                        )
                    except Exception:
                        thinking_lines.append(f"🔧 {content}")
                    _render_thinking()

                # ── Streaming answer tokens (ReAct) ──────────────────────────
                elif ev_type == "token":
                    # Clean up any dangling ⏳ before the answer starts
                    if has_pending_tool and thinking_lines:
                        thinking_lines.pop()
                        has_pending_tool = False
                        _render_thinking()
                    answer_tokens.append(content)
                    answer_placeholder.markdown("".join(answer_tokens) + "▌")

                # ── Complete answer delivered at once (PnE / fallback) ────────
                elif ev_type == "answer":
                    if has_pending_tool and thinking_lines:
                        thinking_lines.pop()
                        has_pending_tool = False
                        _render_thinking()
                    answer_tokens = [content]
                    answer_placeholder.markdown(content)

                # ── Error inside the agent ────────────────────────────────────
                elif ev_type == "error":
                    if has_pending_tool and thinking_lines:
                        thinking_lines.pop()
                        has_pending_tool = False
                    thinking_lines.append(f"❌ 错误：{content}")
                    _render_thinking()

            # Final render: remove cursor; keep thinking box open
            answer = "".join(answer_tokens)
            if answer:
                answer_placeholder.markdown(answer)
            status_box.update(
                label="✅ 完成",
                state="complete",
                expanded=True,  # always stays visible
            )

        except Exception as e:
            answer = f"❌ 调用失败：{e}"
            answer_placeholder.markdown(answer)
            status_box.update(label="❌ 出错", state="error", expanded=True)

    # Persist answer + thinking for this turn
    cm.add_assistant(answer)
    thinking_hist.append(list(thinking_lines))


if __name__ == "__main__":
    main()
