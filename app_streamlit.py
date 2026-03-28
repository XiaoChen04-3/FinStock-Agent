from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from finstock_agent.agent.graph import build_agent, last_ai_text
from finstock_agent.config import settings
from finstock_agent.tools.portfolio import PORTFOLIO_CSV_TEMPLATE


@st.cache_resource
def _compiled_agent():
    return build_agent()


def _rows_to_lc_messages(rows: list[dict]) -> list[HumanMessage | AIMessage]:
    out: list[HumanMessage | AIMessage] = []
    for row in rows:
        if row["role"] == "user":
            out.append(HumanMessage(content=row["content"]))
        else:
            out.append(AIMessage(content=row["content"]))
    return out


def main() -> None:
    st.set_page_config(page_title="FinStock-Agent", layout="wide")
    st.title("FinStock-Agent")
    st.caption("LangGraph + Tushare + OpenAI 兼容 API · 数据仅供学习研究")

    if not settings.tushare_token:
        st.warning("未检测到 TUSHARE_TOKEN，工具调用将失败。请在项目根目录配置 .env。")
    if not settings.openai_api_key:
        st.error("未检测到 OPENAI_API_KEY，无法调用大模型。请配置 .env 后刷新页面。")
        st.stop()

    with st.sidebar:
        st.subheader("持仓流水")
        st.download_button(
            "下载 CSV 模板",
            data=PORTFOLIO_CSV_TEMPLATE.encode("utf-8"),
            file_name="portfolio_template.csv",
            mime="text/csv",
        )
        uploaded = st.file_uploader("上传持仓 CSV", type=["csv"])
        if uploaded is not None:
            st.session_state["portfolio_csv"] = uploaded.getvalue().decode(
                "utf-8", errors="replace"
            )
            st.success("已缓存上传内容。")
        attach_csv = st.checkbox("在提问中附带已上传的 CSV（用于盈亏计算）", value=True)
        if st.button("清除已上传 CSV"):
            st.session_state.pop("portfolio_csv", None)
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "你好，我可以帮你查 A 股行情/指数、申万行业涨跌，或用上传的买卖流水计算盈亏。"
                    "股票代码请用 Tushare 格式（如 600519.SH）。"
                ),
            }
        ]

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("输入问题…")
    if not prompt:
        return

    user_text = prompt
    csv_blob = st.session_state.get("portfolio_csv")
    if attach_csv and csv_blob:
        user_text = (
            f"{prompt}\n\n---\n用户上传的持仓流水 CSV（请用 calculate_portfolio_pnl 工具解析）：\n"
            f"{csv_blob}"
        )

    st.session_state.messages.append({"role": "user", "content": prompt})
    lc_messages = _rows_to_lc_messages(st.session_state.messages[:-1]) + [
        HumanMessage(content=user_text)
    ]

    graph = _compiled_agent()
    with st.spinner("查询与推理中…"):
        try:
            result = graph.invoke({"messages": lc_messages})
            answer = last_ai_text(result.get("messages", [])) or "（模型未返回文本）"
        except Exception as e:
            answer = f"调用失败：{e}"

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()


if __name__ == "__main__":
    main()
