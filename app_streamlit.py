from __future__ import annotations

import json
import logging
import sys
import uuid
from datetime import date, datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from fin_stock_agent.agents.router import stream_agent
from fin_stock_agent.app_bootstrap import ensure_app_bootstrap
from fin_stock_agent.app_runtime import get_local_user_id
from fin_stock_agent.core.logging_utils import configure_application_logging
from fin_stock_agent.init.name_resolver import NameResolver
from fin_stock_agent.memory.conversation import ConversationMemory
from fin_stock_agent.memory.portfolio_memory import PortfolioMemory, TradeRecord
from fin_stock_agent.reporting.daily_reporter import DailyReporter
from fin_stock_agent.reporting.fund_fetcher import TushareFundFetcher
from fin_stock_agent.reporting.report_tasks import ensure_report_generation, get_report_task_snapshot
from fin_stock_agent.services.portfolio_service import PortfolioService
from fin_stock_agent.services.user_memory_service import UserMemoryService

_logger = logging.getLogger(__name__)

WELCOME_MESSAGE = (
    "你好，我是 FinStock-Agent。你可以问我股票、基金、指数和宏观数据；"
    "持仓买卖请在“持仓录入”页操作，左侧会同步展示持仓与盈亏。"
)
INTERRUPTED_MESSAGE = "上一条回答在页面切换或刷新时被打断了，请重新提问。"


def _ensure_local_identity() -> tuple[str, str]:
    st.session_state["user_id"] = get_local_user_id()
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    return st.session_state["user_id"], st.session_state["session_id"]


def _ensure_chat_state(user_id: str, session_id: str) -> tuple[list[dict], list[list[str]]]:
    if (
        st.session_state.get("chat_user_id") != user_id
        or st.session_state.get("chat_session_id") != session_id
        or "chat_rows" not in st.session_state
        or "thinking_history" not in st.session_state
    ):
        st.session_state["chat_user_id"] = user_id
        st.session_state["chat_session_id"] = session_id
        st.session_state["chat_rows"] = []
        st.session_state["thinking_history"] = []
        st.session_state["pending_chat_turn"] = None
    return st.session_state["chat_rows"], st.session_state["thinking_history"]


def _recover_interrupted_chat() -> None:
    pending = st.session_state.get("pending_chat_turn")
    if not pending:
        return

    rows: list[dict] = st.session_state.get("chat_rows", [])
    row_index = pending.get("assistant_row_index", -1)
    if 0 <= row_index < len(rows) and rows[row_index].get("pending"):
        rows[row_index]["content"] = INTERRUPTED_MESSAGE
        rows[row_index]["pending"] = False
        rows[row_index]["meta"] = True
    st.session_state["pending_chat_turn"] = None


def _ensure_welcome_message(rows: list[dict], thinking_history: list[list[str]]) -> None:
    if rows:
        return
    rows.append({"role": "assistant", "content": WELCOME_MESSAGE, "pending": False})
    thinking_history.append([])


def _build_conversation_memory(user_id: str, session_id: str, rows: list[dict]) -> ConversationMemory:
    memory = ConversationMemory(user_id=user_id, session_id=session_id)
    for row in rows:
        if row.get("pending") or row.get("meta"):
            continue
        if row["role"] == "user":
            memory.add_user(row["content"])
        elif row["role"] == "assistant":
            memory.add_assistant(row["content"])
    return memory


def _get_portfolio_memory(user_id: str) -> PortfolioMemory:
    return PortfolioService().build_memory(user_id)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
        .metric-card {
            padding: 14px 16px;
            border-radius: 14px;
            background: linear-gradient(135deg, #f6f4ef, #eef3f8);
            border: 1px solid #d9e0ea;
        }
        .subtle-text { color: #5b6472; font-size: 0.92rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _extract_market_report_part(market_context: str) -> str:
    text = (market_context or "").replace("\r\n", "\n").strip()
    if not text:
        return ""

    for marker in ("**重点十条**", "重点十条"):
        if marker in text:
            head = text.split(marker, 1)[0].strip()
            if head:
                return head

    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    return paragraphs[0] if paragraphs else text


def _normalize_top_news(items: list[dict] | None, *, limit: int = 10) -> list[dict]:
    normalized: list[dict] = []
    for idx, row in enumerate(items or [], 1):
        title = str((row or {}).get("title") or "").strip()
        if not title:
            continue
        normalized.append(
            {
                "rank": row.get("rank") or idx,
                "title": title,
                "source": str(row.get("source") or "").strip(),
                "reason": str(row.get("reason") or "").strip(),
                "impact": row.get("impact"),
                "time": str(row.get("time") or row.get("published_at") or "").strip(),
            }
        )
        if len(normalized) >= limit:
            break
    return normalized


def _render_market_highlights(report) -> None:
    report_part = _extract_market_report_part(report.market_context)
    news_items = _normalize_top_news(report.top_news)

    with st.expander("当日市场要点与重点十条", expanded=True):
        st.markdown("#### 报告 Part")
        if report_part:
            st.markdown(report_part)
        else:
            st.info("暂无可展示的市场报告摘要。")

        st.markdown("#### 新闻 Part")
        if not news_items:
            st.info("暂无可展示的重点新闻。")
            return

        for item in news_items:
            st.markdown(f"{item['rank']}. **{item['title']}**")
            meta_parts = [part for part in [item["source"], item["time"]] if part]
            if meta_parts:
                st.caption(" | ".join(meta_parts))

            detail_parts: list[str] = []
            if item["impact"] not in (None, ""):
                detail_parts.append(f"影响力：{item['impact']}/5")
            if item["reason"]:
                detail_parts.append(item["reason"])
            if detail_parts:
                st.markdown("；".join(detail_parts))


def _render_sidebar(user_id: str, preload_snapshot) -> None:
    with st.sidebar:
        st.subheader("我的持仓")
        if preload_snapshot.state == "running":
            st.caption("启动中正在后台检查历史持仓和日报缓存。")
        elif preload_snapshot.state == "completed" and preload_snapshot.payload:
            payload = preload_snapshot.payload
            _rt = get_report_task_snapshot(user_id=user_id)
            st.caption(
                f"已加载历史数据：{payload.get('holding_count', 0)} 个持仓，"
                f"{payload.get('trade_count', 0)} 条交易，"
                f"{'今日日报生成中' if _rt.state == 'running' else ('已有今日日报' if _rt.report_exists else '今日日报待生成')}。"
            )
        elif preload_snapshot.state == "failed":
            st.caption("历史数据加载暂时不可用，请稍后刷新重试。")

        try:
            holdings = PortfolioService().get_holdings(user_id)
            if not holdings:
                st.info("暂无持仓记录。")
            else:
                import pandas as pd

                cols = ["name", "quantity", "avg_cost", "last_price", "unrealized_pnl"]
                df_h = pd.DataFrame(holdings)
                st.dataframe(df_h[[c for c in cols if c in df_h.columns]], width="stretch", hide_index=True)
        except Exception:
            _logger.exception("sidebar get_holdings failed for user %s", user_id)
            st.warning("持仓数据暂时不可用，请稍后重试。")


def _render_chat_tab(user_id: str, session_id: str) -> None:
    rows, thinking_history = _ensure_chat_state(user_id, session_id)
    _recover_interrupted_chat()
    _ensure_welcome_message(rows, thinking_history)
    portfolio_memory = _get_portfolio_memory(user_id)

    assistant_idx = 0
    for row in rows:
        with st.chat_message(row["role"]):
            if row["role"] == "assistant":
                if assistant_idx < len(thinking_history) and thinking_history[assistant_idx]:
                    with st.expander("查看思考过程", expanded=False):
                        st.markdown("\n\n".join(thinking_history[assistant_idx]))
                st.markdown(row["content"])
                if row.get("pending"):
                    st.caption("回答生成中，切页或刷新会中断当前这次生成。")
                assistant_idx += 1
            else:
                st.markdown(row["content"])

    prompt = st.chat_input("例如：沪深300最近表现如何？白酒板块相关基金有哪些？")
    if not prompt:
        return

    rows.append({"role": "user", "content": prompt})
    assistant_row_index = len(rows)
    thinking_index = len(thinking_history)
    rows.append({"role": "assistant", "content": "正在生成回答...", "pending": True})
    thinking_history.append([])
    st.session_state["pending_chat_turn"] = {
        "assistant_row_index": assistant_row_index,
        "thinking_index": thinking_index,
        "question": prompt,
    }

    with st.chat_message("user"):
        st.markdown(prompt)

    history_rows = rows[:-2]
    history = _build_conversation_memory(user_id, session_id, history_rows).to_lc_messages()

    with st.chat_message("assistant"):
        status_box = st.status("正在生成回答...", expanded=True)
        thinking_lines: list[str] = []
        thinking_placeholder = status_box.empty()
        answer_placeholder = st.empty()
        answer_parts: list[str] = []
        answer = ""
        had_stream_error = False

        def render_thinking() -> None:
            if thinking_lines:
                thinking_placeholder.markdown("\n\n".join(thinking_lines))
            else:
                thinking_placeholder.caption("准备分析中...")

        try:
            for event_type, content in stream_agent(
                prompt,
                user_id=user_id,
                session_id=session_id,
                memory=portfolio_memory,
                history_messages=history,
            ):
                if event_type == "mode":
                    thinking_lines.append(f"- 当前模式：{content}")
                    render_thinking()
                elif event_type == "thinking":
                    thinking_lines.append(f"- {content}")
                    render_thinking()
                elif event_type == "tool_start":
                    thinking_lines.append(f"- 正在调用工具：`{content}`")
                    render_thinking()
                elif event_type == "tool_interaction":
                    try:
                        info = json.loads(content)
                        tool_name = info.get("name", "") or "工具"
                        summary = info.get("summary", "") or "已返回结果"
                        thinking_lines.append(f"- {tool_name}：{summary}")
                    except Exception:
                        thinking_lines.append("- 工具已返回结果")
                    render_thinking()
                elif event_type == "token":
                    answer_parts.append(content)
                    answer_placeholder.markdown("".join(answer_parts) + "▌")
                elif event_type == "answer":
                    answer_parts = [content]
                    answer = content.strip()
                    answer_placeholder.markdown(content)
                    status_box.update(label="已完成", state="complete", expanded=False)
                elif event_type == "error":
                    had_stream_error = True
                    thinking_lines.append("- 分析过程出现问题，已停止本次生成。")
                    render_thinking()

            answer = "".join(answer_parts).strip() or "暂时未生成可展示的回答，请稍后重试。"
            answer_placeholder.markdown(answer)
            if answer and not had_stream_error:
                status_box.update(label="已完成", state="complete", expanded=False)
            elif had_stream_error:
                status_box.update(label="生成失败", state="error", expanded=True)
        except Exception:
            answer = "暂时无法完成回答，请稍后重试。"
            answer_placeholder.markdown(answer)
            status_box.update(label="生成失败", state="error", expanded=True)

        rows[assistant_row_index]["content"] = answer
        rows[assistant_row_index]["pending"] = False
        rows[assistant_row_index]["meta"] = False
        thinking_history[thinking_index] = list(thinking_lines)
        st.session_state["pending_chat_turn"] = None


def _render_report_tab(user_id: str) -> None:
    st.subheader("每日报告")

    reporter = DailyReporter()
    today_str = reporter.resolve_report_date()
    report = reporter.get_existing_report(user_id=user_id, date=today_str)
    task_snapshot = get_report_task_snapshot(user_id=user_id, report_date=today_str)

    col_btn, col_refresh, col_info = st.columns([1, 1, 4])
    with col_btn:
        if st.button("更新日报", width="stretch", key="report_refresh_btn"):
            ensure_report_generation(user_id=user_id, report_date=today_str, force=True)
            task_snapshot = get_report_task_snapshot(user_id=user_id, report_date=today_str)
    with col_refresh:
        if st.button("刷新状态", width="stretch", key="report_status_btn"):
            st.rerun()
    with col_info:
        if task_snapshot.state == "running":
            hint = "后台正在生成今日日报。"
            if report is not None:
                hint += f" 当前先展示 {report.generated_at.strftime('%H:%M:%S')} 的版本。"
            st.caption(hint)
        elif task_snapshot.state == "failed":
            st.caption("后台更新暂时不可用，请稍后重试。")
        elif report is not None:
            st.caption(
                f"当前日报生成于 {report.generated_at.strftime('%H:%M:%S')}。"
                " 点击“更新日报”可重新抓取新闻并刷新建议。"
            )

    if report is None:
        if task_snapshot.state != "failed":
            ensure_report_generation(user_id=user_id, report_date=today_str, force=False)
            st.info("今日日报不存在，已在后台开始抓取并分析。可稍后点击“刷新状态”查看结果。")
        else:
            st.warning("今日日报生成失败，请点击“更新日报”重试。")
        return

    metric_cols = st.columns(4)
    metric_cols[0].metric("总市值", f"{report.total_market_value:,.2f}")
    metric_cols[1].metric("浮动盈亏", f"{report.total_unrealized_pnl:,.2f}")
    metric_cols[2].metric("盈亏率", f"{report.total_unrealized_pnl_pct:.2%}")
    metric_cols[3].metric("生成时间", report.generated_at.strftime("%H:%M:%S"))

    if report.market_context or report.top_news:
        _render_market_highlights(report)

    import pandas as pd
    import plotly.express as px

    if report.fund_statuses:
        st.markdown("#### 持仓基金分析")
        df = pd.DataFrame(
            [
                {
                    "基金": item.name,
                    "代码": item.ts_code,
                    "建议": {"buy": "加仓", "hold": "持有", "sell": "卖出"}.get(item.action, item.action),
                    "置信度": f"{item.confidence:.0%}",
                    "市值": item.market_value,
                    "浮盈亏": item.unrealized_pnl,
                    "近3年收益": f"{item.three_year_return_pct:.1%}" if item.three_year_return_pct is not None else "暂无",
                    "趋势": item.trend,
                }
                for item in report.fund_statuses
            ]
        )
        st.dataframe(df, width="stretch", hide_index=True)
        chart = px.bar(
            pd.DataFrame([item.model_dump() for item in report.fund_statuses]),
            x="name",
            y="unrealized_pnl",
            color="action",
            title="各持仓浮动盈亏与操作建议",
            color_discrete_map={"buy": "#2ecc71", "hold": "#3498db", "sell": "#e74c3c"},
        )
        st.plotly_chart(chart, width="stretch")

        for item in report.fund_statuses:
            action_text = {"buy": "加仓", "hold": "持有", "sell": "卖出"}.get(item.action, item.action)
            with st.expander(f"{item.name} · {item.ts_code} · 建议{action_text}", expanded=False):
                st.markdown(item.reason or item.analysis_summary or "暂无详细分析。")
                if item.analysis_summary and item.analysis_summary != item.reason:
                    st.caption(item.analysis_summary)
                if item.related_news:
                    st.markdown("最相关新闻")
                    for title in item.related_news:
                        st.markdown(f"- {title}")
                if item.key_risks:
                    st.markdown("风险提示")
                    for risk in item.key_risks:
                        st.markdown(f"- {risk}")
    else:
        st.info("当前暂无持仓，日报仅展示市场主线、重点新闻与情绪分析。")

    st.markdown(report.overall_summary)
    st.caption(report.disclaimer)


def _render_holdings_input_tab(user_id: str) -> None:
    st.subheader("持仓录入")
    st.caption(
        "选择买入或卖出、基金与成交日期；份额或金额二选一。"
        "成交价按 Tushare `fund_nav` 在成交日或最近披露净值日自动填充。"
    )

    col_kw, col_btn = st.columns([4, 1])
    with col_kw:
        keyword = st.text_input(
            "基金名称关键字",
            key="fund_search_keyword",
            placeholder="例如：沪深300、债券、货币基金",
            label_visibility="collapsed",
        )
    with col_btn:
        st.write("")
        search_clicked = st.button("搜索基金", width="stretch")

    if search_clicked:
        if keyword.strip():
            try:
                results = NameResolver().search_funds(keyword.strip(), top_k=8)
                st.session_state["fund_search_results"] = results
            except Exception:
                st.error("基金搜索暂时不可用，请稍后重试。")
                st.session_state["fund_search_results"] = []
        else:
            st.warning("请输入关键字后再搜索。")

    search_results: list[dict] = st.session_state.get("fund_search_results", [])
    selected_fund: dict | None = None

    if search_results:
        options = {f"{r['name']} ({r['ts_code']})": r for r in search_results}
        chosen_label = st.selectbox("请选择基金", list(options.keys()), key="selected_fund_label")
        selected_fund = options.get(chosen_label)
    elif "fund_search_results" in st.session_state and not search_results:
        st.info("未找到匹配的基金，请尝试其他关键字。")

    if selected_fund:
        st.divider()
        st.markdown(f"**已选择：** {selected_fund['name']}  `{selected_fund['ts_code']}`")

        svc = PortfolioService()
        pos_map = {h["ts_code"]: h for h in svc.get_holdings(user_id)}
        code = selected_fund["ts_code"]
        avail_qty = float(pos_map.get(code, {}).get("quantity") or 0.0)

        col_a, col_b = st.columns(2)
        with col_a:
            direction = st.radio("方向", ["买入", "卖出"], horizontal=True, key="hi_direction")
        with col_b:
            trade_date_val: date = st.date_input("成交日期", value=datetime.today().date(), key="hi_trade_date")

        input_mode = st.radio("数量方式", ["按份额", "按金额"], horizontal=True, key="hi_input_mode")

        if input_mode == "按份额":
            qty_in = st.number_input("份额", min_value=0.0, step=100.0, format="%.4f", key="hi_quantity")
            amt_in = 0.0
        else:
            amt_in = st.number_input("金额（元）", min_value=0.0, step=1000.0, format="%.2f", key="hi_amount")
            qty_in = 0.0

        fee_input = st.number_input("手续费（元，可选）", min_value=0.0, step=1.0, format="%.2f", value=0.0, key="hi_fee")

        if direction == "卖出" and avail_qty <= 0:
            st.warning("当前该基金无可用持仓，无法卖出。请先买入或检查是否选错基金。")

        if st.button("确认提交", type="primary", width="stretch", key="hi_submit"):
            is_buy = direction == "买入"
            fetcher = TushareFundFetcher()
            ymd = trade_date_val.strftime("%Y%m%d")
            nav, nav_used = fetcher.fetch_unit_nav_on_or_before(code, ymd)

            err: str | None = None
            if nav is None or nav <= 0:
                err = (
                    f"无法从 Tushare 获取 {code} 在 {ymd} 附近的净值。"
                    " 请确认基金代码和 `TUSHARE_TOKEN` 配置。"
                )
            elif input_mode == "按份额" and qty_in <= 0:
                err = "份额必须大于 0。"
            elif input_mode == "按金额" and amt_in <= 0:
                err = "金额必须大于 0。"
            elif not is_buy:
                if avail_qty <= 0:
                    err = "没有可卖份额。"
                elif input_mode == "按份额" and qty_in > avail_qty + 1e-6:
                    err = f"卖出份额不能超过持仓（当前可卖 {avail_qty:.4f} 份）。"
                elif input_mode == "按金额":
                    sell_qty = amt_in / nav
                    if sell_qty > avail_qty + 1e-6:
                        err = (
                            f"按该日净值折算约 {sell_qty:.4f} 份，超过可卖 {avail_qty:.4f} 份。"
                            " 请减少金额或改用按份额卖出。"
                        )

            if err:
                st.error(err)
            else:
                final_qty = amt_in / nav if input_mode == "按金额" else qty_in
                try:
                    record = TradeRecord(
                        ts_code=code,
                        name=selected_fund["name"],
                        direction="buy" if is_buy else "sell",
                        quantity=final_qty,
                        price=nav,
                        fee=fee_input,
                        trade_date=ymd,
                    )
                    svc.add_trade(user_id, record)
                    used_note = f"净值日期 {nav_used}" if nav_used else ""
                    st.success(
                        f"已记录：**{direction}** {selected_fund['name']} × **{final_qty:.4f}** 份，"
                        f"成交净值 **{nav:.4f}**（{used_note}），成交日 {trade_date_val}"
                    )
                    st.session_state.pop("fund_search_results", None)
                    st.rerun()
                except Exception:
                    st.error("交易保存失败，请稍后重试。")

    st.divider()
    st.subheader("当前持仓")
    try:
        import pandas as pd
        from sqlalchemy import select

        service = PortfolioService()
        holdings = service.get_holdings(user_id)
        if not holdings:
            st.info("暂无持仓记录，请在上方录入您的基金持仓。")
        else:
            st.dataframe(pd.DataFrame(holdings), width="stretch", hide_index=True)

            with st.expander("管理交易记录"):
                trades = service.get_trade_history(user_id, limit=50)
                if trades:
                    trade_options = {
                        f"#{i + 1} {t.trade_date} {t.name or t.ts_code} {t.direction} ×{t.quantity:.2f} @{t.price:.4f}": i
                        for i, t in enumerate(trades)
                    }
                    to_delete_label = st.selectbox("选择要删除的交易", list(trade_options.keys()), key="hi_delete_sel")
                    if st.button("删除所选交易", key="hi_delete_btn"):
                        from fin_stock_agent.storage.database import get_session
                        from fin_stock_agent.storage.models import TradeRecordORM

                        idx = trade_options[to_delete_label]
                        with get_session() as sess:
                            orm_rows = sess.execute(
                                select(TradeRecordORM)
                                .where(
                                    TradeRecordORM.user_id == user_id,
                                    TradeRecordORM.is_deleted.is_(False),
                                )
                                .order_by(TradeRecordORM.trade_date.desc(), TradeRecordORM.id.desc())
                                .limit(50)
                            ).scalars().all()
                            row_ids = [r.id for r in orm_rows]
                        if idx < len(row_ids):
                            service.delete_trade(user_id, row_ids[idx])
                            st.success("已删除。")
                            st.rerun()
    except Exception:
        _logger.exception("holdings_input_tab get_holdings failed for user %s", user_id)
        st.warning("持仓数据暂时不可用，请稍后重试。")


def main() -> None:
    configure_application_logging()
    st.set_page_config(page_title="FinStock-Agent", page_icon="FA", layout="wide")
    _inject_styles()
    user_id, session_id = _ensure_local_identity()

    bootstrap = ensure_app_bootstrap(user_id=user_id, session_id=session_id)
    if not bootstrap.ready:
        return

    st.title("FinStock-Agent")
    _render_sidebar(user_id, bootstrap.preload_snapshot)
    tab_chat, tab_report, tab_holdings = st.tabs(["智能问答", "每日报告", "持仓录入"])
    with tab_chat:
        _render_chat_tab(user_id, session_id)
    with tab_report:
        _render_report_tab(user_id)
    with tab_holdings:
        _render_holdings_input_tab(user_id)


if __name__ == "__main__":
    main()
