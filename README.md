# FinStock-Agent

FinStock-Agent 是一个面向中国市场的本地智能投研助手，整合了基金持仓管理、LLM 驱动的市场问答和自动化每日报告三条能力链路。项目基于 `Streamlit` 提供界面，使用 `LangGraph / LangChain` 编排双模式 Agent，使用 `Tushare` 获取结构化市场数据，并通过本地 `SQLite + ChromaDB` 持久化交易记录、新闻缓存、用户记忆和每日报告。

## 功能概览

| 模块 | 说明 |
|------|------|
| 智能问答 | 股票 / 基金 / 指数 / 宏观 / 技术指标 / 持仓 全品类问答，自动切换 ReAct 或 Plan-and-Execute 模式 |
| 持仓管理 | 基金检索、买卖录入、净值自动回填、持仓与浮动盈亏计算 |
| 每日报告 | 自动抓取财经快讯，生成市场摘要、重点十条新闻与个性化持仓建议 |
| 记忆系统 | 三层跨会话持久化：用户画像（包括用户偏好以及用户持仓） / 对话摘要 / 日报摘要，自动注入每次问答的 SystemMessage |
| 计划复用 | 相似问题自动检索历史执行计划，向量相似度双档位控制复用粒度 |
| 可观测性 | 每次问答记录意图、Agent 模式、工具调用、Token 消耗与耗时 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

**配置一**：复制 `config.example.yaml` 为 `config.yaml`，按需填写各 LLM 角色模型名、记忆预算、计划复用阈值等参数。

**配置二**：在项目根目录创建 `.env`，配置密钥：

```env
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-5.5
TUSHARE_TOKEN=your_token
```

### 3. 启动

```bash
streamlit run app_streamlit.py
```

默认监听 `127.0.0.1:8501`。

## 使用说明

### 智能问答

- 查询股票、基金、指数实时行情
- 技术指标（MA / MACD / RSI）与宏观数据（CPI / GDP / M2）
- 结合本地持仓回答"我的组合怎么样""我该关注什么"等个性化问题
- 问答界面实时展示意图识别、工具调用步骤与 Token 消耗

### 每日报告

- 当日市场要点与市场情绪摘要
- AI 遴选重点新闻十条（含影响力评分与理由）
- 持仓基金建议（加仓 / 持有 / 卖出 + 置信度 + 风险提示）
- 浮动盈亏可视化图表

### 持仓录入

- 按名称关键字搜索基金
- 录入买入 / 卖出，支持按份额或按金额
- 成交净值通过 Tushare 自动填充最近披露日
- 管理历史交易记录（可删除）

## 项目结构

```
app_streamlit.py              Streamlit 入口与 UI 渲染
config.example.yaml           配置模板（含多 LLM 角色、记忆预算、计划阈值）
fin_stock_agent/
  agents/                     Agent 路由、ReAct、Plan-and-Execute
  core/                       配置（Pydantic）、LLM 多角色工厂、查询增强、异常
  init/                       系统初始化、名称解析、交易日历、数据预加载
  memory/                     对话记忆（向量化摘要）、ChromaDB 抽象层
  news/                       新闻抓取与 DB 缓存
  reporting/                  每日报告流水线（Orchestrator + 5 个子 Agent）
  services/                   持仓服务、记忆聚合器、用户画像服务、日报摘要服务、计划复用库
  storage/                    SQLite 数据库、ORM 模型（含主库只读 Fallback）
  stats/                      运行统计（JSONL + DB 双写）
  tools/                      LangChain 工具集（行情 / 基金 / 宏观等）
  prompts/                    Agent Prompt 模板（ReAct / Plan / 报告各子 Agent）
  utils/                      工具函数（盈亏计算、Tushare 客户端等）
tests/                        pytest 测试套件
```

## 运行测试

```bash
python -m pytest -q
```

## License

MIT
