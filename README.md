# FinStock-Agent

基于 **LangGraph**、**Tushare** 与 **OpenAI 兼容 API** 的股票数据对话助手，使用 **Streamlit** 提供界面。数据仅供学习研究，输出不构成投资建议。

## 功能概览

- 对话查询 A 股基础信息、日线行情、`daily_basic` 估值快照（视 Tushare 积分）
- 主要宽基指数区间表现、指数日线
- 申万一级行业涨跌排行（依赖 `sw_daily` / `index_classify` 权限与积分）
- 上传买卖流水 CSV，计算加权成本、实现盈亏与浮动盈亏（收盘价来自 Tushare `daily`）

## 环境准备

1. Python 3.10+（推荐 3.11）
2. [Tushare Pro](https://tushare.pro/) 账号与 `token`
3. 任意 OpenAI 兼容网关的 `api_key` 与 `base_url`（如 DeepSeek、通义等）

## 安装

```bash
cd FinStock-Agent
pip install -r requirements.txt
```

填写环境变量：

编辑 `.env`：`TUSHARE_TOKEN`、`OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL`。

## 验证 Tushare

```bash
python verify_tushare.py
```

应打印 `stock_basic` 样例行。

## 运行 Streamlit

```bash
streamlit run app_streamlit.py
```

在侧边栏可下载持仓 CSV 模板、上传流水；勾选「附带 CSV」后，下一条提问会把文件内容交给 Agent，模型应调用 `calculate_portfolio_pnl`。

## 项目结构

- `finstock_agent/`：配置、Tushare 封装（节流、重试、日线/指数 SQLite 缓存）
- `finstock_agent/tools/market.py`：行情与指数、行业类工具
- `finstock_agent/tools/portfolio.py`：持仓盈亏工具与 CSV 模板
- `finstock_agent/agent/graph.py`：`create_react_agent` 装配
- `app_streamlit.py`：Web 入口
- `tests/test_portfolio.py`：盈亏计算单元测试

## 测试

```bash
pytest tests -q
```

## 说明与限制

- Tushare 不同接口对积分要求不同；权限不足时工具会返回可读错误信息。
- 行业排行工具会对多个行业代码逐次请求，已限制单次最多 25 个以控制耗时。
- 默认缓存文件路径由 `FINSTOCK_CACHE_PATH` 指定（默认项目根目录下 `.finstock_cache.sqlite`）。
