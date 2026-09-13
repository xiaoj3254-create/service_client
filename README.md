

# 多智能体客服系统

---

## 项目概述

基于 LangGraph 构建的多智能体客服系统，支持产品咨询、技术支持、账单处理、投诉处理等多种业务场景。系统采用模块化设计，每个智能体独立运行，通过条件路由将客户查询分发至对应的专业智能体。


## 项目结构

```
customer-service-ai-agent/
├── multi_agents/                  # 智能体模块
│   ├── __init__.py                # 智能体包初始化
│   ├── base_agent.py              # 基础智能体类（抽象基类 + 对话上下文管理）
│   ├── product_agent.py           # 产品专家智能体
│   ├── tech_agent.py              # 技术支持智能体
│   ├── billing_agent.py           # 账单专家智能体
│   ├── complaint_agent.py         # 投诉处理智能体
│   └── general_agent.py           # 综合客服智能体
├── tools/                         # 工具函数模块
│   ├── __init__.py                # 工具包初始化
│   └── query_tools.py            # 查询分类工具（LLM 驱动）
├── templates/                     # Web 界面模板
│   └── index.html                 # 主页面（聊天 UI + 会话侧边栏）
├── config.py                      # 基础配置（API、HTTP、日志）
├── multi_agent_customer_service.py # 主程序（LangGraph 工作流定义 + LLM 客户端）
├── session_manager.py             # 会话管理器（LangChain 标准接口）
├── web_app.py                     # Web 入口（Flask 路由）
├── chat_web_service.py            # 业务逻辑（LangGraph REST 调用、线程/会话管理）
├── langgraph.json                 # LangGraph 工作流配置
├── requirements.txt               # 项目依赖
├── .env                           # 环境变量配置
└── README_LangGraph_CLI.md        # LangGraph CLI 使用说明
```

## 主要特性

### 模块化智能体设计

- 每个智能体独立实现，职责单一
- 基于抽象基类 `BaseAgent`，便于扩展和维护
- 支持动态 LLM 注入（延迟初始化）

### 配置驱动的工作流

- 工作流定义在 `langgraph.json` 中
- 支持条件路由和直接连接
- 图入口函数 `make_graph()` 在 `multi_agent_customer_service.py` 中硬编码编译

### 专业业务领域

| 智能体 | 文件 | 职责 |
|---|---|---|
| 产品专家 | `multi_agents/product_agent.py` | 产品信息查询、规格推荐 |
| 技术支持 | `multi_agents/tech_agent.py` | 故障诊断、兼容性问题 |
| 账单专员 | `multi_agents/billing_agent.py` | 支付、退款、发票 |
| 投诉处理 | `multi_agents/complaint_agent.py` | 客户投诉、建议反馈 |
| 综合客服 | `multi_agents/general_agent.py` | 物流、营业时间等一般咨询 |

### 智能对话上下文管理

- **会话隔离**：每次新建对话创建独立 LangGraph 线程，互不干扰
- **历史对话缓存**：通过 `persisted_dialogue` 状态字段由 checkpointer 持久化，跨进程有效
- **上下文感知**：智能体优先从图状态读取历史对话，生成连贯回答
- **多轮对话**：同一会话内支持连续对话，自动续写上下文
- **护栏机制**：非客服范围查询（越狱、闲聊等）直接拦截，不进入业务智能体

## 安装和配置

### 1. 创建虚拟环境并安装依赖

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
pip install -r requirements.txt
pip install -U "langgraph-cli[inmem]"
pip install colorama              # Windows 彩色日志必需
```

> **Windows 注意**：`langgraph-cli` 安装后需将 `langgraph.exe` 路径加入 `PATH`，或直接使用 `.\.venv\Scripts\langgraph.exe`。

### 2. 环境变量配置

复制 `env_example.txt` 为 `.env` 并填写：

```env
# OpenAI 兼容 API 配置
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
OPENAI_MODEL=Qwen/Qwen3-8B

# HTTP 配置
HTTP_TIMEOUT=30
HTTP_MAX_RETRIES=3

# 日志配置
LOG_LEVEL=INFO
```

### 3. 图结构自检

```bash
python multi_agent_customer_service.py
```

输出 `✅ LangGraph工作流图构建完成` 即表示通过。

## 运行说明

### 方式 1：LangGraph Studio UI

```bash
langgraph dev
```

启动后自动打开 LangSmith Studio，默认端口 2024。
浏览器访问：`https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`

### 方式 2：Web 应用（Flask + LangGraph）

需启动两个服务：

```bash
# 终端 1：LangGraph 服务
($env:PYTHONUTF8=1)
langgraph dev --port 2024 --host 127.0.0.1 --no-browser

# 终端 2：Flask Web 应用
python web_app.py
```

浏览器访问 `http://localhost:5000`，界面功能：

- 实时聊天：输入问题，获得智能回复
- 智能体信息：显示当前处理问题的专家和查询类型
- 会话管理：新建对话、查看历史、切换/删除会话
- 数据导出：导出对话记录用于分析

### 方式 3：直接 API 调用

接口文档：`http://127.0.0.1:2024/docs`（页面内嵌 JS，需网络代理）
完整 API 参考：[LangGraph API 文档](https://langchain-ai.github.io/langgraph/cloud/reference/api/api_ref.html)

## 工作流程

```
客户查询
  ↓
classify_query（分类节点）
  ├── 调用 LLM 进行查询分类
  ├── 用户消息写入 persisted_dialogue
  ├── 若 out_of_scope → 直接返回固定拒答
  └── 根据分类结果路由 ↓
      ├── product_info      → 产品专家
      ├── technical_support → 技术支持
      ├── billing           → 账单专员
      ├── complaint         → 投诉处理
      └── general_inquiry   → 综合客服
           ↓
      业务智能体处理
      ├── 读取 persisted_dialogue 获取上下文
      ├── 匹配本地知识库
      ├── 构建 messages → 调用 LLM
      └── AI 回复写入 persisted_dialogue
           ↓
final_response（最终响应节点）
  └── 加上智能体名称前缀，工作流结束
```

### 状态管理

系统使用 `AgentState` 管理工作流状态：

| 字段 | 类型 | 说明 |
|---|---|---|
| `customer_query` | `str` | 客户查询内容 |
| `query_type` | `str` | 查询类型分类 |
| `current_agent` | `str` | 当前处理智能体 |
| `response` | `str` | 智能体回复 |
| `tools_used` | `List[str]` | 使用的工具列表 |
| `session_id` | `str` | 会话唯一标识 |
| `conversation_history` | `List` | 对话历史（内存） |
| `persisted_dialogue` | `List` | 由 checkpointer 持久化的对话（跨进程有效） |
| `memory` | `BaseChatMessageHistory` | LangChain 记忆组件 |

## 关于 LLM 服务

系统使用硅基流动（SiliconFlow）作为默认 LLM 服务商，兼容 OpenAI API 格式，可无缝替换为其他服务商。

### 优势

- **国内服务**：访问速度快，延迟低
- **价格实惠**：相比其他 API 服务更经济
- **模型丰富**：支持多种开源模型
- **API 兼容**：完全兼容 OpenAI API 格式

### 推荐模型

| 模型 | 特点 |
|---|---|
| `Qwen/Qwen3-8B` | 性价比高，适合一般应用（默认） |
| `Qwen/Qwen2.5-14B-Instruct` | 性能更好，适合复杂任务 |
| `meta-llama/Llama-3.1-8B-Instruct` | 通用性强，稳定性好 |
| `mistralai/Mistral-7B-Instruct` | 推理能力强 |

> 切换模型只需修改 `.env` 中的 `OPENAI_MODEL` 和 `OPENAI_BASE_URL`。

## 扩展指南

### 添加新的智能体

1. 在 `multi_agents/` 目录下创建新文件，继承 `BaseAgent` 并实现 `process()` 方法
2. 在 `multi_agents/__init__.py` 中导入新智能体
3. 在 `multi_agent_customer_service.py` 的 `initialize_agents()` 中注册
4. 在 `make_graph()` 中添加节点和条件边

### 添加新的工具函数

1. 在 `tools/` 目录下创建新文件
2. 使用 `@tool` 装饰器定义工具
3. 在 `tools/__init__.py` 中导入

### 修改工作流程

- 图结构在 `multi_agent_customer_service.py` 的 `make_graph()` 中硬编码编译
- `langgraph.json` 中的 `workflow` 段为配置化描述（与代码中的图定义对应）
- 修改节点/边后重启服务即可生效

## 技术架构

| 组件 | 技术 | 说明 |
|---|---|---|
| 工作流编排 | LangGraph | 状态图、条件路由、checkpointer 持久化 |
| LLM 集成 | LangChain Core | 消息格式、工具装饰器 |
| Web 服务 | Flask | 路由、会话、SSE 流式 |
| LLM 服务 | 硅基流动 API | OpenAI 兼容接口 |
| 会话管理 | LangChain SessionManager | 多会话并发、历史缓存 |

## 相关文档

- [README_LangGraph_CLI.md](README_LangGraph_CLI.md) — LangGraph CLI 部署指南
- [langgraph.json](langgraph.json) — 工作流配置文件
- [LangGraph CLI 官方文档](https://docs.langchain.com/langgraph-platform/cli#configuration-file)
- [LangGraph Studio](https://smith.langchain.com/)
