# 多智能体客服系统

基于 **LangGraph** 构建的多智能体客服系统，支持产品咨询、技术支持、账单处理、投诉处理等业务场景。
采用模块化设计，每个智能体职责单一，通过条件路由将客户查询分发至对应的专业智能体。

**技术栈**：LangGraph · LangChain Core · Flask · 任意 OpenAI 兼容 LLM

---

## 目录

- [核心特性](#核心特性)
- [checkpointer 与图工厂签名](#checkpointer-与图工厂签名)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [运行方式](#运行方式)
- [工作流程](#工作流程)
- [状态管理](#状态管理)
- [配置项参考](#配置项参考)
- [HTTP 接口](#http-接口)
- [关于 LLM 服务](#关于-llm-服务)
- [扩展指南](#扩展指南)
- [技术架构](#技术架构)
- [相关文档](#相关文档)

---

## 核心特性

### 模块化智能体设计

- 5 个专家智能体各自独立实现，职责单一
- 基于抽象基类 `BaseAgent` 提供**统一处理骨架**，新增智能体只需实现 3 个钩子（见[扩展指南](#添加新的智能体)）
- 支持动态 LLM 注入（延迟初始化），便于测试时替换为桩实现

### 配置驱动的工作流

- 图结构在 `multi_agent_customer_service.py` 的 `make_graph()` 中定义并编译（**零参数工厂**，签名受平台约束）
- `langgraph.json` 中的 `workflow` 段为**描述性元数据**（与代码中的图定义对应），供平台识别与文档化
- 支持条件路由（按分类结果分发）与直接连接（各智能体汇入 `final_response`）

### 专业业务领域

| 智能体 | 文件 | 职责 |
|---|---|---|
| 产品专家 | `multi_agents/product_agent.py` | 产品规格、价格比较、功能推荐 |
| 技术支持 | `multi_agents/tech_agent.py` | 故障诊断、系统优化、硬件问题 |
| 账单专员 | `multi_agents/billing_agent.py` | 退款、发票、支付方式 |
| 投诉处理 | `multi_agents/complaint_agent.py` | 投诉受理、补偿措施、服务改进 |
| 综合客服 | `multi_agents/general_agent.py` | 营业时间、联系方式等一般咨询 |

### 智能对话上下文管理

- **会话隔离**：每次新建对话创建独立 LangGraph 线程，互不干扰
- **历史对话持久化**：通过 `persisted_dialogue` 状态字段由 checkpointer 承载
- **上下文感知**：智能体优先从图状态读取历史对话，生成连贯回答
- **多轮对话**：同一会话内支持连续对话，自动续写上下文

### 多图上传

- 单条消息最多附带 **4 张图片**，单张压缩后不超过 **5 MB**
- 前端 canvas 压缩（无第三方依赖）→ data URL 传输，**不落服务器磁盘**
- 后端做**魔术字节校验**：比对声明的 MIME 与真实字节头，不一致即拒绝
- 图片随 `persisted_dialogue` 持久化，刷新页面后历史气泡仍能回显

### 护栏机制（fail-closed）

非客服范围查询（越狱、闲聊、提示词注入等）直接拦截，**不进入业务智能体**。

> **关于护栏兜底**：分类输出无法识别时的兜底标签为 `out_of_scope`（保守拒答），而非 `general_inquiry`。
> 这样可确保「识别不了就不放行」，避免分类链路异常时把越界或拒答请求误路由到业务智能体。
> 拦截结果可通过 `tools_used` 中的 `out_of_scope_refusal` 标记识别。

> **关于 checkpointer（重要）**：`make_graph()` 是 **LangGraph 平台的图工厂入口，签名必须零参数**——
> 平台按「参数个数」判定工厂类型，1 个未标注 `ServerRuntime` 的参数会被当成 config 并收到
> `RunnableConfig` 字典；用 `*args, **kwargs` 更会直接让图加载失败。详见下方
> [checkpointer 与图工厂签名](#checkpointer-与图工厂签名)。
>
> - **平台托管**（`langgraph dev` / `up`）：`make_graph()` **不挂载** checkpointer，
>   由平台注入它自己支持删除/回滚/ttl 的持久化实现。
> - **独立运行**：用 `set_checkpointer()` 预设，或直接 `python multi_agent_customer_service.py`
>   （`__main__` 中已显式挂载进程内 `InMemorySaver`，因此具备多轮对话记忆）。

### checkpointer 与图工厂签名

`make_graph()` **必须保持零参数**，这不是代码风格偏好，而是平台的硬性约束：

| `make_graph` 参数个数 | 平台行为 |
| --- | --- |
| **0 个**（当前实现） | 合法，直接以 `value()` 调用 |
| 1 个 | 未标注 `ServerRuntime` 时**一律视为 config**，把 `RunnableConfig` 字典传进来 |
| 2 个 | 其中一个必须标注 `ServerRuntime`，否则抛 `ValueError` |
| ≥3 个 / 使用 `*args`、`**kwargs` | 抛 `ValueError`，**图直接加载失败** |

> 历史上 `make_graph(checkpointer=None)` 正是踩了「1 个参数 = config 工厂」这一条，
> 导致平台把 config 字典塞进 `checkpointer` 并透传给 `compile()`，
> 抛 `TypeError: Invalid checkpointer provided ... Received dict`。

需要自定义 saver 时用 `set_checkpointer()`（模块级槽位，绕开签名限制）：

```python
import multi_agent_customer_service as m
from langgraph.checkpoint.sqlite import SqliteSaver

m.set_checkpointer(SqliteSaver.from_conn_string("checkpoints.db"))
app = m.make_graph()

m.set_checkpointer(None)    # 清除，恢复平台托管形态
```

> **平台托管场景不要调用 `set_checkpointer()`**：图内自带 saver 会让平台跳过注入，
> 导致 `DELETE /threads/<id>` 与 `multitask_strategy='rollback'` 的清理失效。

---

## 项目结构

```
customer-service-ai-agent/
├── multi_agents/                    # 智能体模块
│   ├── __init__.py                  # 智能体包初始化（统一导出）
│   ├── base_agent.py                # 抽象基类：统一处理骨架 + 上下文管理 + 匹配骨架
│   ├── product_agent.py             # 产品专家智能体
│   ├── tech_agent.py                # 技术支持智能体
│   ├── billing_agent.py             # 账单专家智能体
│   ├── complaint_agent.py           # 投诉处理智能体
│   └── general_agent.py             # 综合客服智能体
├── tools/                           # 工具函数模块
│   ├── __init__.py                  # 工具包初始化
│   └── query_tools.py               # 查询分类工具（LLM 驱动 + 标签归一化）
├── templates/                       # Web 界面模板
│   └── index.html                   # 主页面（聊天 UI + 会话侧边栏 + 图片上传）
├── config.py                        # 基础配置（API / HTTP / 日志 / 会话上限）+ setup_logging()
├── multi_agent_customer_service.py  # 主程序（LangGraph 工作流定义 + LLM 客户端）
├── session_manager.py               # 会话管理器（LangChain 标准接口 + TTL/LRU 淘汰）
├── web_app.py                       # Web 入口（Flask 路由 + 图片校验 + 调试端点门控）
├── chat_web_service.py              # 业务逻辑（LangGraph REST 调用、线程/会话管理）
├── langgraph.json                   # LangGraph 工作流配置
├── requirements.txt                 # 项目依赖
├── env_example.txt                  # 环境变量模板（复制为 .env）
├── README_LangGraph_CLI.md          # LangGraph CLI 使用说明
├── 项目审查报告.md                   # 代码审查结论与分级修复记录
└── 运行修复记录.md                   # 历次运行问题的根因与修复过程
```

---

## 快速开始

### 1. 创建虚拟环境并安装依赖

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# 安装依赖（langgraph-cli[inmem] 与 colorama 已包含在 requirements.txt 中）
pip install -r requirements.txt
```

> **Windows 注意**：若 `langgraph` 命令不可用，直接将 `.\.venv\Scripts\langgraph.exe` 加入 `PATH`，
> 或使用完整路径调用。
>
> `colorama` 已列入 `requirements.txt`：`langgraph-api` 使用 structlog 的彩色控制台渲染器，
> 在 Windows 上缺失该包会导致启动失败。

### 2. 环境变量配置

复制 `env_example.txt` 为 `.env` 并按需填写：

```env
# ===== LLM 服务（OpenAI 兼容）=====
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
OPENAI_MODEL=Qwen/Qwen3-8B

# ===== HTTP 客户端 =====
HTTP_TIMEOUT=30
HTTP_MAX_RETRIES=3

# ===== 日志 =====
LOG_LEVEL=INFO

# ===== Flask Web 应用 =====
FLASK_HOST=127.0.0.1
FLASK_PORT=5000
FLASK_DEBUG=False
FLASK_SECRET_KEY=请填写一个足够随机的密钥

# ===== 会话上限（避免长跑进程内存无限增长）=====
SESSION_MAX_COUNT=500
SESSION_TTL_HOURS=24
SESSIONS_LIST_LIMIT=50

# ===== 调试端点（生产环境务必设为 false）=====
ENABLE_DEBUG_ENDPOINTS=true

# ===== LangGraph HTTP API（一般无需修改）=====
# LANGGRAPH_API_URL=http://127.0.0.1:2024
# LANGGRAPH_GRAPH_NAME=customer_service
```

> **安全提示**：未配置 `FLASK_SECRET_KEY` 时，程序会生成**进程级随机密钥**并在日志中告警。
> 这会导致重启后所有会话失效，因此生产环境请显式配置。

### 3. 图结构自检

```bash
python multi_agent_customer_service.py
```

输出包含 `LangGraph工作流图构建完成（使用 set_checkpointer 指定的 saver）` 即表示通过。

---

## 运行方式

### 方式 1：LangGraph Studio UI

```bash
# Windows 下建议先设置编码，避免源码含中文时 GBK 解码报错
$env:PYTHONUTF8=1
langgraph dev --port 2024 --host 127.0.0.1 --no-browser
```

启动后访问 Studio：
`https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`

### 方式 2：Web 应用（Flask + LangGraph）

需启动两个服务：

```bash
# 终端 1：LangGraph 服务（端口与 langgraph.json 的 http.port 保持一致）
$env:PYTHONUTF8=1
langgraph dev --port 2024 --host 127.0.0.1 --no-browser

# 终端 2：Flask Web 应用
$env:PYTHONUTF8=1
python web_app.py
```

浏览器访问 `http://localhost:5000`，界面功能：

- **实时聊天**：输入问题获得智能回复，支持多图上传
- **智能体信息**：显示当前处理问题的专家与查询类型
- **会话管理**：新建对话、查看历史、侧栏搜索、切换/删除/清空会话
- **数据导出**：导出对话记录用于分析

### 方式 3：Studio 本地服务直连

```bash
python multi_agent_customer_service.py    # 独立运行，__main__ 中挂载默认 InMemorySaver
```

---

## 工作流程

```
客户查询
  ↓
classify_query（分类节点）
  ├── 调用 LLM 进行查询分类，并对输出做标签归一化
  ├── 用户消息写入 persisted_dialogue
  ├── 若 out_of_scope → 直接返回固定拒答（fail-closed）
  └── 根据分类结果路由 ↓
      ├── product_info      → product_agent（产品专家）
      ├── technical_support → tech_agent（技术支持）
      ├── billing           → billing_agent（账单专员）
      ├── complaint         → complaint_agent（投诉处理）
      └── general_inquiry   → general_agent（综合客服）
           ↓
      业务智能体处理（统一走 BaseAgent.run_agent 骨架）
      ├── 读取 persisted_dialogue 获取上下文
      ├── 匹配本地知识库
      ├── 构建 messages（含图片时构造多模态消息）→ 调用 LLM
      └── 结果写入 state（response / current_agent / tools_used）
           ↓
final_response（最终响应节点）
  └── 加上智能体名称前缀，工作流结束
```

> **分类兜底原则**：`classify_query_node` 内任何异常（LLM 失败、工具调用失败、输出无法识别）
> 统一兜底为 `out_of_scope`，绝不默认放行到业务智能体。
> 同时空查询分支会显式设置 `next_agent = "final_response"`——
> 否则条件边会取默认值 `general_agent`，绕过护栏。

### 状态管理

系统使用 `AgentState` 管理工作流状态：

| 字段 | 类型 | 说明 |
|---|---|---|
| `customer_query` | `str` | 客户查询内容 |
| `query_type` | `str` | 查询类型分类结果 |
| `next_agent` | `str` | 条件边的路由依据（由分类节点设置） |
| `current_agent` | `str` | 当前处理智能体名称 |
| `response` | `str` | 智能体回复内容 |
| `tools_used` | `List[str]` | 使用记录（`query_classification`、`<智能体名>_processing`、`out_of_scope_refusal`） |
| `session_id` | `str` | 会话唯一标识 |
| `conversation_history` | `List` | 对话历史（内存） |
| `persisted_dialogue` | `List` | 由 checkpointer 持久化的对话（跨进程有效） |
| `customer_images` | `Optional[List[str]]` | 客户上传的图片（data URL 列表，最多 4 张） |
| `memory` | `BaseChatMessageHistory` | LangChain 记忆组件 |

> **注意**：护栏拦截通过 `tools_used` 含 `out_of_scope_refusal` 与 `current_agent == "智能客服"` 判定，
> 状态中**没有** `blocked` 字段。

---

## 配置项参考

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `OPENAI_API_KEY` | 空 | LLM 服务密钥，缺失时系统降级为模拟响应 |
| `OPENAI_BASE_URL` | `https://api.siliconflow.cn/v1` | OpenAI 兼容接口地址 |
| `OPENAI_MODEL` | `Qwen/Qwen3-8B` | 模型名称 |
| `HTTP_TIMEOUT` | `30` | HTTP 请求超时（秒） |
| `HTTP_MAX_RETRIES` | `3` | HTTP 重试次数 |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `FLASK_HOST` | `127.0.0.1` | Flask 绑定地址（**默认仅本机**） |
| `FLASK_PORT` | `5000` | Flask 端口 |
| `FLASK_DEBUG` | `False` | 调试模式（**请勿在生产开启**） |
| `FLASK_SECRET_KEY` | 随机生成 | Flask 会话密钥，未配置时每次启动随机 |
| `SESSION_MAX_COUNT` | `500` | memory 后端最大会话数，超出按最久未活动淘汰 |
| `SESSION_TTL_HOURS` | `24` | 会话空闲过期时间（小时） |
| `SESSIONS_LIST_LIMIT` | `50` | 会话列表默认返回条数上限 |
| `ENABLE_DEBUG_ENDPOINTS` | `true` | 是否启用 `/api/test` 调试端点（生产设 `false`） |
| `LANGGRAPH_API_URL` | `http://127.0.0.1:2024` | LangGraph 服务地址 |

> **生产部署清单**：
> 1. 显式配置 `FLASK_SECRET_KEY`（否则重启后会话全部失效）
> 2. 设置 `ENABLE_DEBUG_ENDPOINTS=false` 关闭调试端点
> 3. 保持 `FLASK_DEBUG=False`；若必须绑定 `0.0.0.0`，**切勿**同时开启 debug
> 4. 确认 `langgraph.json` 的 `http.port` 与 `LANGGRAPH_API_URL` 端口一致

---

## HTTP 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/` | 聊天主页面 |
| `POST` | `/api/chat` | 发送消息（支持 `images` 字段，最多 4 张 data URL） |
| `POST` | `/api/chat/stream` | SSE 形式返回。**当前为轮询式**：整轮运行完成后一次性下发，非 token 级增量流 |
| `GET` | `/api/sessions` | 会话列表，支持 `?limit=` / `?offset=` 分页 |
| `GET` | `/api/sessions/<id>` | 会话详情（含历史对话） |
| `DELETE` | `/api/sessions/<id>` | 删除会话 |
| `POST` | `/api/sessions/<id>/clear` | 清空会话历史（保留线程） |
| `GET` | `/api/health` | 健康检查 |
| `GET` | `/api/test` | LangGraph 连通性测试（受 `ENABLE_DEBUG_ENDPOINTS` 门控，关闭时返回 403） |

请求示例：

```bash
curl -X POST http://127.0.0.1:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "你们的手机有哪些型号？", "session_id": "default"}'
```

响应中的 `thread_id` 即 LangGraph 线程 ID。**前端应采纳该 ID 用于后续消息**，
以保证同一会话的消息写入同一线程；首次发送时可传 `session_id: "default"` 由后端创建线程。

---

## 关于 LLM 服务

系统默认使用硅基流动（SiliconFlow），兼容 OpenAI API 格式，可无缝替换为其他服务商。

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
> 若要接入多模态模型（图片理解），需选择支持视觉输入的模型。

---

## 扩展指南

### 添加新的智能体

得益于 `BaseAgent` 提供统一处理骨架，新增智能体只需关注业务差异：

1. 在 `multi_agents/` 下创建新文件，继承 `BaseAgent` 并实现 3 个钩子 + 1 个类属性：

```python
from multi_agents.base_agent import BaseAgent

class RefundAgent(BaseAgent):
    _KIND_LABEL = "退款政策信息"          # 知识片段在 prompt 中的标题

    def __init__(self):
        super().__init__(
            name="退款专员",
            role="退款流程处理",
            expertise=["退款政策", "到账时间", "材料要求"],
        )
        self.refund_database = { ... }     # 知识库

    def _match_info(self, query: str) -> str:
        """从知识库匹配相关片段（可复用基类的 _match_by_category）"""
        return self._match_by_category(
            query, self.refund_database,
            fallback_keywords=["退款", "到账", "退货"],
            fallback_prefix="相关政策",
        )

    def _base_system_prompt(self) -> str:
        return f"你是{self.name}，专门负责{self.role}。"

    def _llm_error_reply(self) -> str:
        return "抱歉，处理您的退款问题时遇到系统错误，请稍后重试。"

    def process(self, state):
        return self.run_agent(state)       # 统一骨架
```

2. 在 `multi_agents/__init__.py` 中导入并加入 `__all__`
3. 在 `multi_agent_customer_service.py` 的 `_AGENT_CLASSES` 映射中注册
4. 在 `make_graph()` 中添加节点与条件边分支
5. **同步更新 `langgraph.json`** 的 `workflow.nodes` 与 `conditional.classify_query`
6. 在 `tools/query_tools.py` 的 `_CLASS_LABELS` 中加入新的分类标签

> **不要重写整个 `process()`**。基类的 `run_agent()` 已统一处理：
> 读取对话上下文 → 匹配知识 → 组装 SystemMessage 列表 → 构造（多模态）HumanMessage →
> 调用 LLM → 写回 `response` / `current_agent` / `tools_used`。

### 添加新的工具函数

1. 在 `tools/` 目录下创建新文件
2. 使用 `@tool` 装饰器定义工具
3. 在 `tools/__init__.py` 中导入

### 修改工作流程

- 图结构在 `multi_agent_customer_service.py` 的 `make_graph()` 中定义并编译
- `langgraph.json` 中**有两套图定义**：`graphs` 指向真实实现，`workflow` 段为描述性元数据
- **修改节点/边后两处都要同步**，否则文档会漂移（历史上曾出现 `workflow` 段缺少 4 个 agent 节点与 `out_of_scope` 边的情况）
- ⚠️ **`make_graph()` 的零参数签名不可改动**，否则图无法被平台加载（见 [checkpointer 与图工厂签名](#checkpointer-与图工厂签名)）
- 修改后重启服务生效

---

## 技术架构

| 组件 | 技术 | 说明 |
|---|---|---|
| 工作流编排 | LangGraph | 状态图、条件路由、checkpointer 持久化 |
| LLM 集成 | LangChain Core | 消息格式、工具装饰器 |
| Web 服务 | Flask | 路由、SSE 传输、图片校验 |
| LLM 服务 | OpenAI 兼容 API | 默认硅基流动，可替换 |
| 会话管理 | LangChain SessionManager | 多会话并发、TTL/LRU 淘汰 |
| 日志 | Python logging | 统一由 `config.setup_logging()` 配置 |

---

## 相关文档

- [README_LangGraph_CLI.md](README_LangGraph_CLI.md) — LangGraph CLI 部署指南
- [langgraph.json](langgraph.json) — 工作流配置文件
- [项目审查报告.md](项目审查报告.md) — 代码审查结论、P0/P1/P2 分级与修复记录
- [运行修复记录.md](运行修复记录.md) — 历次运行问题的根因分析与修复过程
- [LangGraph CLI 官方文档](https://docs.langchain.com/langgraph-platform/cli#configuration-file)
- [LangGraph Studio](https://smith.langchain.com/)
