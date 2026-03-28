from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from fin_stock_agent.agents.router import stream_agent
from fin_stock_agent.core.settings import settings
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

    # ---------------------------------------------------------------------------
    # Sidebar: portfolio memory display
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
                st.rerun()

        st.divider()
        st.subheader("🤖 Agent 模式说明")
        st.markdown(
            """
- **ReAct 模式**：单一意图查询（行情、指数、记录交易）
- **Plan & Execute**：复杂多步分析（持仓对比大盘、多股分析等）
- 如 P&E 失败将自动降级至 ReAct
            """
        )

    # ---------------------------------------------------------------------------
    # Chat history display
    # ---------------------------------------------------------------------------
    if len(cm) == 0:
        cm.add_assistant(
            "你好！我是 FinStock-Agent。\n\n"
            "你可以直接告诉我你买了或卖了哪些股票，我会自动记录到持仓记忆中。"
            "也可以问我 A 股行情、指数走势、板块涨跌，或让我计算你的持仓盈亏。\n\n"
            "股票代码请用 Tushare 格式，例如：600519.SH（茅台）、000001.SZ（平安银行）。"
        )

    for row in cm.all_rows():
        with st.chat_message(row["role"]):
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

    with st.chat_message("assistant"):
        # ── Thinking process area (collapsible status box) ──────────────────
        status_box = st.status("思考中…", expanded=True)
        thinking_lines: list[str] = []
        thinking_placeholder = status_box.empty()

        # ── Final answer area (outside the status box) ───────────────────────
        answer_placeholder = st.empty()
        answer_tokens: list[str] = []
        answer: str = ""

        try:
            for ev_type, content in stream_agent(
                prompt, memory=pm, history_messages=history
            ):
                if ev_type == "mode":
                    status_box.update(label=f"🤖 {content} 模式 — 思考中…", expanded=True)

                elif ev_type == "tool_call":
                    thinking_lines.append(f"🔧 **调用工具** `{content}`")
                    thinking_placeholder.markdown("\n\n".join(thinking_lines))

                elif ev_type == "tool_result":
                    # Truncate long results for display
                    short = content[:150].replace("\n", " ")
                    thinking_lines.append(f"📊 工具返回：{short}…")
                    thinking_placeholder.markdown("\n\n".join(thinking_lines))

                elif ev_type == "thinking":
                    thinking_lines.append(f"💭 {content}")
                    thinking_placeholder.markdown("\n\n".join(thinking_lines))

                elif ev_type == "token":
                    answer_tokens.append(content)
                    # Show typing cursor while streaming
                    answer_placeholder.markdown("".join(answer_tokens) + "▌")

                elif ev_type == "answer":
                    # Complete answer delivered at once (PnE mode)
                    answer_tokens = [content]
                    answer_placeholder.markdown(content)

                elif ev_type == "error":
                    thinking_lines.append(f"❌ 错误：{content}")
                    thinking_placeholder.markdown("\n\n".join(thinking_lines))

            # Final render: remove cursor, collapse status box
            answer = "".join(answer_tokens)
            if answer:
                answer_placeholder.markdown(answer)
            status_box.update(
                label="✅ 完成",
                state="complete",
                expanded=False,
            )

        except Exception as e:
            answer = f"❌ 调用失败：{e}"
            answer_placeholder.markdown(answer)
            status_box.update(label="❌ 出错", state="error", expanded=True)

    cm.add_assistant(answer)


if __name__ == "__main__":
    main()
