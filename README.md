# FinStock-Agent

基于 **LangGraph** + **Tushare** + **OpenAI 兼容 API** 的 A 股市场对话助手，使用 **Streamlit** 提供界面。

> 数据仅供学习研究，输出不构成投资建议。

---

## 功能

| 功能 | 说明 |
|------|------|
| 对话查行情 | 股票基础信息、日线行情、估值快照（视 Tushare 积分） |
| 指数 & 板块 | 宽基指数区间涨跌、申万一级行业涨跌排行 |
| 持仓记忆 | 在对话中提及买卖交易，Agent 自动识别并存入记忆 |
| 盈亏计算 | 基于持仓记忆，加权平均成本法计算实现/浮动盈亏 |
| 智能路由 | 简单问题 → ReAct；复杂多步分析 → Plan-and-Execute；P&E 失败自动降级 |

---

## 项目架构

```
fin_stock_agent/
  core/           # 核心层：settings、LLM 工厂、异常
  agents/         # Agent 层：ReAct、Plan-and-Execute、路由器
  tools/          # 工具层：market / portfolio / memory_tools
  memory/         # 记忆层：PortfolioMemory、ConversationMemory
  prompts/        # Prompt 层：react / plan / extraction 四套模板
  utils/          # 通用工具层：TushareClient、PnL 计算器
app_streamlit.py  # Streamlit 入口
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量


编辑 `.env`：

```
TUSHARE_TOKEN=你的token
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.deepseek.com/v1   # 或任意 OpenAI 兼容地址
OPENAI_MODEL=deepseek-chat
```

### 3. 验证 Tushare 连通性

```bash
python verify_tushare.py
```

### 4. 启动

```bash
streamlit run app_streamlit.py
```

---

## 使用持仓记忆

无需上传 CSV，直接在对话中告诉 Agent：

- 「我上周买了100股茅台(600519.SH)，价格1688元」
- 「今天卖出了30股，成交价1750元，手续费5块」
- 「帮我计算一下现在的盈亏情况」

Agent 会自动提取交易信息、存入记忆，并在需要时调用盈亏计算工具。

左侧边栏实时显示所有已记录的交易流水，可随时清空。

---

## 测试

```bash
pytest tests -q
```

---

## Tushare 积分说明

不同接口对积分要求不同，积分不足时工具会返回可读错误。推荐先用 `stock_basic`、`daily`、`index_daily` 等低积分接口验证流程，再根据账号权限开放更多功能。
