# FinStock-Agent

FinStock-Agent 是一个面向中国市场场景的本地智能投研助手原型，整合了基金持仓管理、市场问答和每日报告三条能力链路。项目基于 `Streamlit` 提供界面，使用 `LangGraph/LangChain` 编排 Agent，使用 `Tushare` 获取结构化市场数据，并通过本地 `SQLite` 持久化交易、新闻缓存和日报结果。

## Features

- 智能问答：支持股票、基金、指数、宏观、技术指标、新闻、持仓相关问题
- 持仓管理：支持基金检索、买卖录入、净值回填、持仓与盈亏计算
- 每日报告：自动抓取财经快讯，生成市场摘要、重点新闻、持仓分析和可关注基金
- 本地优先：单用户本地运行，数据默认保存在项目目录内
- 可观测：记录工具调用、token 消耗和日报生成统计

## Stack

- UI: `Streamlit`
- Agent orchestration: `LangChain`, `LangGraph`
- Market data: `Tushare`
- Storage: `SQLite`, `SQLAlchemy`
- Async news ingestion: `aiohttp`

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Configure

在项目根目录创建 `.env`，最小配置如下：

```env
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
TUSHARE_TOKEN=your_token
```

常用可选项：

```env
DATABASE_URL=sqlite:///.data/finstock.db
APP_TIMEZONE=Asia/Shanghai
USER_ID_SEED=finstock-agent
```

### 3. Run

```bash
streamlit run app_streamlit.py
```

默认监听 `127.0.0.1:8501`。

## What You Can Do

### 智能问答

- 查询股票、基金、指数行情
- 做技术指标和宏观数据问答
- 结合本地持仓回答“我该关注什么”“我的组合怎么样”

### 每日报告

- 查看当日市场摘要
- 浏览重点新闻十条
- 查看持仓基金建议和主题基金候选

### 持仓录入

- 搜索基金
- 录入买入 / 卖出
- 自动回填净值并更新持仓盈亏

## Project Layout

```text
app_streamlit.py              Streamlit 入口
fin_stock_agent/
  agents/                     问答路由与 Agent 编排
  reporting/                  每日报告流水线
  tools/                      Agent 工具集
  services/                   持仓与上下文服务
  news/                       新闻抓取与缓存
  storage/                    数据库与缓存封装
  stats/                      运行统计
tests/                        回归测试
```

## Testing

```bash
python -m pytest -q
```