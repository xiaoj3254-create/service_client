# 使用 LangGraph CLI 部署多智能体客服系统

基于 [LangGraph CLI 官方文档](https://docs.langchain.com/langgraph-platform/cli#configuration-file)，系统支持通过 LangGraph CLI 进行本地开发与生产部署。

## 主要优势

- **标准化部署**：使用官方推荐的 LangGraph CLI 工具
- **Docker 支持**：自动构建和运行 Docker 容器
- **生产就绪**：支持生产环境的部署和扩展
- **配置驱动**：通过 `langgraph.json` 配置文件管理所有设置

## 配置文件结构

### langgraph.json 配置说明

> 图入口函数为 `make_graph`（非 `build_workflow`），定义在 `multi_agent_customer_service.py` 中。

```json
{
  "dependencies": ["."],
  "graphs": {
    "customer_service": "./multi_agent_customer_service.py:make_graph"
  },
  "env": "./.env",
  "python_version": "3.13",
  "base_image": "langchain/langgraph-api:0.2",
  "workflow": {
    "name": "多智能体客服系统",
    "description": "基于LangGraph的多智能体客服系统，支持产品咨询、技术支持、账单处理、投诉处理等",
    "entry_point": "classify_query",
    "nodes": {
      "classify_query": {
        "type": "function",
        "description": "分类客户查询类型",
        "function": "classify_query_node"
      },
      "general_agent": {
        "type": "agent",
        "description": "综合客服智能体",
        "agent": "GeneralAgent",
        "expertise": ["信息查询", "基础服务", "问题转接"]
      },
      "final_response": {
        "type": "function",
        "description": "生成最终响应",
        "function": "final_response_node"
      }
    },
    "edges": {
      "conditional": {
        "classify_query": {
          "product_info": "product_agent",
          "technical_support": "tech_agent",
          "billing": "billing_agent",
          "complaint": "complaint_agent",
          "general_inquiry": "general_agent"
        }
      },
      "direct": [
        ["product_agent", "final_response"],
        ["tech_agent", "final_response"],
        ["billing_agent", "final_response"],
        ["complaint_agent", "final_response"],
        ["general_agent", "final_response"]
      ]
    },
    "end_point": "final_response"
  },
  "store": {
    "ttl": {
      "refresh_on_read": true,
      "sweep_interval_minutes": 60,
      "default_ttl": 10080
    }
  },
  "checkpointer": {
    "ttl": {
      "strategy": "delete",
      "sweep_interval_minutes": 10,
      "default_ttl": 43200
    }
  },
  "http": {
    "port": 8123,
    "host": "0.0.0.0"
  }
}
```

### 配置字段速查

| 字段 | 说明 | 当前值 |
|---|---|---|
| `graphs.customer_service` | 图入口（文件路径:函数名） | `make_graph` |
| `python_version` | 运行时 Python 版本 | `3.13` |
| `base_image` | Docker 基础镜像 | `langchain/langgraph-api:0.2` |
| `http.port` / `http.host` | API 服务监听地址 | `8123` / `0.0.0.0` |
| `store.ttl.default_ttl` | Store 默认 TTL（分钟，10080 = 7 天） | `10080` |
| `checkpointer.ttl.default_ttl` | Checkpoint TTL（分钟，43200 = 30 天） | `43200` |

## 安装和配置

### 1. 安装 LangGraph CLI

```bash
# 基础版本
pip install langgraph-cli

# 开发版本（支持 dev 命令，本地开发必需）
pip install -U "langgraph-cli[inmem]"
```

> **Windows 注意**：安装后需将 `langgraph.exe` 所在路径加入 `PATH`，或使用全路径调用。
> 若使用项目虚拟环境，安装后可直接 `.\.venv\Scripts\langgraph.exe dev`。
>
> Windows 环境还需额外安装 `colorama`，否则 LangGraph 启动时彩色日志渲染器会崩溃：
> ```bash
> pip install colorama
> ```

### 2. 验证安装

```bash
langgraph --help
```

### 3. 环境配置

确保项目根目录下有以下文件：

| 文件 | 用途 |
|---|---|
| `langgraph.json` | LangGraph CLI 配置文件 |
| `.env` | 环境变量（API 密钥、模型配置等） |
| `requirements.txt` | Python 依赖 |
| `multi_agents/` | 智能体模块目录 |
| `tools/` | 工具函数模块目录 |

**环境变量配置示例（`.env`）**：

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

## 使用方法

### 开发模式（dev）

```bash
# 基本启动（默认端口 2024，自动打开浏览器）
langgraph dev

# 指定端口和主机，不自动打开浏览器
langgraph dev --port 2024 --host 127.0.0.1 --no-browser

# 启用断点调试
langgraph dev --debug-port 5678 --wait-for-client
```

> `dev` 模式支持热重载：修改代码后服务自动重启。

### 构建 Docker 镜像

```bash
# 构建镜像
langgraph build -t my-customer-service:latest

# 多平台构建
langgraph build --platform linux/amd64,linux/arm64 -t my-customer-service:latest

# 不拉取最新基础镜像
langgraph build --no-pull -t my-customer-service:latest
```

### 启动服务（up）

```bash
# 启动 LangGraph API 服务器
langgraph up

# 指定端口
langgraph up -p 8000

# 等待服务就绪
langgraph up --wait

# 使用本地构建的镜像
langgraph up --image my-customer-service:latest
```

### 生成 Dockerfile

```bash
# 生成 Dockerfile
langgraph dockerfile -c langgraph.json Dockerfile
```

## API 端点

启动服务后，接口文档默认位于 `http://127.0.0.1:2024/docs`（页面内嵌 JS，需网络代理）。

完整 API 参考也可查阅 [LangGraph API 官方文档](https://langchain-ai.github.io/langgraph/cloud/reference/api/api_ref.html)。

### Assistants 管理

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/assistants` | 创建助手 |
| POST | `/assistants/search` | 搜索助手 |
| GET | `/assistants/{assistant_id}` | 获取助手详情 |
| DELETE | `/assistants/{assistant_id}` | 删除助手 |
| PATCH | `/assistants/{assistant_id}` | 更新助手 |

### Threads 管理

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/threads` | 创建线程 |
| POST | `/threads/search` | 搜索线程 |
| GET | `/threads/{thread_id}` | 获取线程详情 |
| DELETE | `/threads/{thread_id}` | 删除线程 |
| PATCH | `/threads/{thread_id}` | 更新线程 |

### Thread Runs 执行

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/threads/{thread_id}/runs` | 在线程上提交运行 |
| POST | `/threads/{thread_id}/runs/stream` | 流式运行 |
| GET | `/threads/{thread_id}/runs/{run_id}` | 获取运行状态 |
| POST | `/threads/{thread_id}/runs/{run_id}/cancel` | 取消运行 |

### Stateless Runs 执行

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/runs` | 无状态执行 |
| POST | `/runs/stream` | 无状态流式执行 |
| POST | `/runs/batch` | 批量执行 |

### 系统状态

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/info` | 服务器信息 |
| GET | `/ok` | 健康检查 |
| GET | `/metrics` | 系统指标 |

## 监控和调试

### 开发模式

- **热重载**：代码修改后自动重启服务
- **调试支持**：支持断点调试（`--debug-port`）
- **实时日志**：查看详细的执行日志
- **Studio 集成**：自动连接到 LangGraph Studio

### 生产模式

- **Docker 容器**：隔离的运行环境
- **健康检查**：自动健康状态监控
- **日志管理**：结构化的日志输出
- **性能监控**：内置的性能指标

## Docker 部署

### 构建镜像

```bash
# 构建生产镜像
langgraph build -t customer-service:latest

# 查看构建的镜像
docker images | grep customer-service
```

### 运行容器

```bash
docker run -d \
  --name customer-service \
  -p 2024:2024 \
  -e OPENAI_API_KEY=your_key \
  customer-service:latest
```

### Docker Compose

创建 `docker-compose.yml`：

```yaml
version: '3.8'
services:
  customer-service:
    build: .
    ports:
      - "2024:2024"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./.env:/app/.env
    restart: unless-stopped
```

```bash
docker-compose up -d
```

## 相关资源

- [LangGraph CLI 官方文档](https://docs.langchain.com/langgraph-platform/cli#configuration-file)
- [LangGraph Platform 概述](https://docs.langchain.com/langgraph-platform/)
- [使用 langgraph 创建模板项目](https://docs.langchain.com/langgraph-platform/local-server)
- [LangGraph Studio](https://smith.langchain.com/)
