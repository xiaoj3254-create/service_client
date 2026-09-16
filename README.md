# 多智能体客服系统

基于 LangGraph 构建的多智能体客服系统，通过查询分类器将用户请求路由到 5 个领域智能体（产品 / 技术 / 账单 / 投诉 / 综合）之一，并支持多轮对话、多模态图片输入、SSE 流式响应与会话历史持久化。

## 技术栈

- **语言**：Python 3.11+（开发环境 3.13）
- **工作流编排**：LangGraph ≥ 1.0、LangChain Core ≥ 1.3
- **LLM**：OpenAI 兼容 API（默认 SiliconFlow 的 Qwen3-8B，可切换任意 OpenAI 兼容端点，如 mimo-v2.5）
- **Web 框架**：Flask ≥ 2.3（路由 + Jinja2 模板）
- **配置加载**：python-dotenv
- **HTTP 客户端**：requests
- **数据校验**：pydantic ≥ 2.7
- **会话持久化**：LangGraph checkpointer（TTL 12 小时，每 10 分钟清理）
- **知识存储**：LangGraph store（TTL 7 天）

## 系统架构

```
用户请求
   │
   ▼
┌──────────────────┐
│ classify_query   │  ← 6 类标签路由：product / tech / billing / complaint / general / out_of_scope
│ (含 out_of_scope │     识别失败一律走 out_of_scope（护栏，拒答不放行到业务智能体）
│   护栏)          │
└─────────┬────────┘
          │ 条件边
          ▼
┌────────────────────────────────────────────┐
│  ProductAgent │ TechAgent │ BillingAgent │ │
│  ComplaintAgent │ GeneralAgent             │
│  (BaseAgent 模板方法 + _get_system_prompt  │
│   + _match_data 子类实现)                   │
└────────────────────┬───────────────────────┘
                     │
                     ▼
          ┌──────────────────┐
          │  final_response │  ← 合成最终回复，写入 state.response / persisted_dialogue
          └──────────────────┘
```

### 关键设计

- **classify_query 护栏**：分类失败或越界话题一律路由到 `final_response` 返回固定拒答，绝不放行到业务智能体（保守拒答）。
- **BaseAgent 模板方法**：[base_agent.py](multi_agents/base_agent.py) 定义 `process()` 公共流程，5 个子类只需实现 `_get_system_prompt()` 与 `_match_data()`，避免重复代码。
- **按需实例化**：单次路由只创建实际需要的那一个智能体实例并缓存，避免重复构造。
- **跨进程对话续聊**：对话历史持久化在 LangGraph checkpointer（图状态字段 `persisted_dialogue`），跨 LangGraph 工作进程仍可续聊；进程内 `LangChainSessionManager` 仅作回退。

## 目录结构

```
customer-service-ai-agent-main/
├── multi_agent_customer_service.py   # 主工作流：StateGraph、classify_query_node、final_response_node、make_graph
├── chat_web_service.py               # Flask 业务层：LangGraph REST 调用、线程管理、会话列表、SSE 流式
├── web_app.py                        # Flask 路由入口：/ 与 /api/* 路由、图片校验
├── config.py                         # 配置加载（从 .env 读取）
├── session_manager.py                # LangChainSessionManager：进程内会话管理（回退用）
├── langgraph.json                    # LangGraph 平台配置（graph_id、节点、边、checkpointer TTL）
├── requirements.txt                  # Python 依赖
├── env_example.txt                   # 环境变量示例
├── .env                              # 实际环境变量（不要提交到仓库）
├── multi_agents/
│   ├── __init__.py
│   ├── base_agent.py                 # BaseAgent 抽象基类（模板方法）
│   ├── product_agent.py              # 产品专家
│   ├── tech_agent.py                 # 技术支持
│   ├── billing_agent.py              # 账单专家
│   ├── complaint_agent.py            # 投诉处理
│   └── general_agent.py              # 综合客服
├── tools/
│   ├── __init__.py
│   └── query_tools.py                # classify_query 工具、normalize_classifier_label、_looks_like_refusal
└── templates/
    └── index.html                    # 前端：侧栏会话列表 + 搜索、图片上传、SSE 流式接收
```

## 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

> 还需安装 LangGraph CLI（`langgraph dev` 命令）：
> ```bash
> pip install "langgraph-cli[inmem]"
> ```

### 2. 配置环境变量

复制 `env_example.txt` 为 `.env` 并填入实际值：

```bash
cp env_example.txt .env
```

关键字段：

| 变量 | 说明 | 默认值 |
| --- | --- | --- |
| `OPENAI_API_KEY` | LLM 服务 API Key（必填） | — |
| `OPENAI_BASE_URL` | OpenAI 兼容端点 | `https://api.siliconflow.cn/v1` |
| `OPENAI_MODEL` | 模型名 | `Qwen/Qwen3-8B` |
| `HTTP_TIMEOUT` | HTTP 超时秒数 | `30` |
| `HTTP_MAX_RETRIES` | HTTP 重试次数 | `3` |
| `LOG_LEVEL` | 日志级别 | `INFO` |
| `FLASK_PORT` | Flask 端口 | `5000` |
| `FLASK_DEBUG` | Flask debug 模式（生产请置 `False`） | `False` |
| `FLASK_SECRET_KEY` | Flask session 密钥（生产必填强随机值） | 未设置时自动生成临时随机值 |
| `LANGGRAPH_API_URL` | LangGraph API 地址 | `http://127.0.0.1:2024` |
| `LANGGRAPH_GRAPH_NAME` | 图名 | `customer_service` |

### 3. 启动服务

系统由两个独立进程组成，需分别启动：

**进程 1：LangGraph API（端口 2024）**

```powershell
# Windows PowerShell：必须先设置 UTF-8，否则 langgraph dev 加载 .env 会因 GBK 编码报错
$env:PYTHONUTF8=1
langgraph dev --port 2024 --host 0.0.0.0
```

**进程 2：Flask Web 应用（端口 5000）**

```powershell
$env:PYTHONUTF8=1
python web_app.py
```

### 4. 访问

打开浏览器访问 [http://localhost:5000](http://localhost:5000) 即可使用。

## HTTP API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 主页（渲染 [templates/index.html](templates/index.html)） |
| POST | `/api/chat` | 同步聊天，返回 AI 回复与新 thread_id |
| POST | `/api/chat/stream` | SSE 流式聊天，`text/event-stream` |
| GET | `/api/sessions` | 获取会话列表（并发拉取线程状态） |
| GET | `/api/sessions/{session_id}` | 获取指定会话详情 |
| DELETE | `/api/sessions/{session_id}` | 删除会话（远程线程 + 本地） |
| POST | `/api/sessions/{session_id}/clear` | 清空会话并新建线程 |
| POST | `/api/new_session` | 在 Flask session 侧新建会话 |
| GET | `/api/health` | 健康检查 |
| GET | `/api/test` | 测试 LangGraph API 连通性 |

### 聊天请求示例

```json
POST /api/chat
{
  "message": "我的订单还没收到，怎么办？",
  "session_id": "default",
  "images": ["data:image/jpeg;base64,..."]
}
```

- `message`：用户文本，必填。
- `session_id`：会话 ID。`default` 或非法 ID 会新建线程；合法 ID 会复用历史会话续聊。
- `images`：可选，data URL 列表，最多 4 张，单张解码后不超过 5MB，支持 PNG/JPEG/WebP/GIF。

## 核心特性

- **多智能体路由**：classify_query 输出 6 类标签之一，条件边将请求路由到对应领域智能体或直接拒答。
- **out_of_scope 护栏**：识别失败、越界话题一律走 `final_response` 返回固定拒答，避免错误放行到业务智能体。
- **多轮对话**：图状态字段 `persisted_dialogue` 由 LangGraph checkpointer 持久化，跨进程续聊有效；TTL 12 小时，每 10 分钟清理一次。
- **多模态图片输入**：前端上传图片转 base64 data URL，后端校验后透传给 OpenAI 兼容端点（如 mimo-v2.5 图片理解）。
- **SSE 流式响应**：`/api/chat/stream` 以 `text/event-stream` 推送 token 级增量。
- **侧栏会话管理**：左侧栏列出历史会话，支持搜索、切换、删除、清空。
- **并发会话拉取**：[chat_web_service.py](chat_web_service.py) 使用 `ThreadPoolExecutor(max_workers=8)` 并发拉取多个线程状态。
- **延迟初始化 LLM**：首次调用时创建 `OpenAICompatibleClient` 并缓存，API Key 缺失时回退到错误提示模式。

## 注意事项

### Windows 编码问题

- 运行 `langgraph dev` 前必须设置 `PYTHONUTF8=1`，否则系统默认 GBK 编码会导致 `UnicodeDecodeError`。
- 所有包含中文字符的 Python 脚本应以 UTF-8 读取。

### 测试请求的编码陷阱

- **PowerShell 的 `Invoke-WebRequest` 默认用 ASCII 编码 POST body**，发送中文会乱码，导致分类器识别失败。
- 测试中文接口时请使用：
  - Python `requests` 库（会正确以 UTF-8 编码），或
  - 浏览器 / Postman / curl（`curl --data-binary` + UTF-8 文件）

### 生产部署

- `FLASK_DEBUG` 必须置 `False`。
- `FLASK_SECRET_KEY` 必须设置为强随机值（至少 32 字节），不要使用弱默认值。
- 不要将 `.env` 提交到版本控制。
- LangGraph API（`langgraph dev`）仅用于开发；生产请使用 `langgraph build` + 容器部署。

### 诊断日志

[tools/query_tools.py](tools/query_tools.py)、[chat_web_service.py](chat_web_service.py)、[multi_agent_customer_service.py](multi_agent_customer_service.py) 中可能含有临时诊断日志（`[诊断]` 前缀），生产环境可在 `LOG_LEVEL=INFO` 下自动过滤，或按需删除 `logger.info("[诊断]...")` 行。

## 相关文档

- [README_LangGraph_CLI.md](README_LangGraph_CLI.md)：LangGraph CLI 的使用说明。
