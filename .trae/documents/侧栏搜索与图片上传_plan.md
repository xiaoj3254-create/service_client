# 侧栏历史对话搜索 + 客户图片上传 实施计划

## Repository Research（现状调研）

### 现有会话列表链路（搜索功能基础）

- 前端 `templates/index.html :: refreshSessionList()`（L1686）GET `/api/sessions`，拿到数组后交给 `displaySessionList(sessions)`（L1741）渲染，**前端目前不缓存列表数据**。
- 后端 `web_app.py :: get_sessions()`（L106）→ `chat_web_service.py :: fetch_sessions_list()`（L291）。
- `fetch_sessions_list()` 已经**逐个线程** GET `/threads/{tid}/state`，并调用 `conversation_history_from_state_data()`（L61）解析出完整对话列表，但只取了 `last_user_question_from_history()`（最后一条用户消息）和 `message_count` 返回——**完整历史已在服务端内存中，只是没返回**，增加搜索字段无需额外 HTTP 调用。

### 现有发消息链路（图片功能基础）

- 前端 `sendMessage()`（L1880）POST JSON `{message, session_id}` 到 `/api/chat`。
- `web_app.py :: chat()`（L63）→ `chat_web_service.py :: run_chat_sync()`（L414）→ POST `/threads/{tid}/runs`，input 含 `messages`、`customer_query`、`session_id`。
- 图节点：`classify_query_node`（L220）→ 条件边（读 `next_agent`）→ 业务智能体 `agent.process(state)` → `final_response_node`（L376）。
- 五个智能体构建消息的模式完全一致（各 2 行 `messages.append(HumanMessage(content=...))`，行号 86/88）：先知识库上下文，再用户查询。
- `OpenAICompatibleClient.invoke()`（L81）将 LangChain 消息转为 dict 时，`msg.content` 原样透传——**若 content 是 list（多模态格式），天然兼容**，只需验证无需特殊处理。

### 模型能力（已确认）

当前 `.env` 使用 `mimo-v2.5`（base_url `https://api.xiaomimimo.com/v1`）。经官方文档确认，该模型**原生支持图片理解**，采用 OpenAI 兼容多模态格式：

```json
{"role": "user", "content": [
  {"type": "text", "text": "请看这张报错截图"},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,...."}}
]}
```

### 关键约束

- 分类节点 `classify_query` 用独立 LLM 调用做意图分类。计划中**分类仍只用文本**（省视觉 token；用户通常会配文字描述），图片只传给业务智能体。
- `persisted_dialogue` 由 LangGraph checkpointer 持久化，历史会话渲染依赖它（`append_turn_from_state` L44）。图片需要随用户轮次存入，才能在"加载历史会话"时重新显示。
- 发送按钮就绪逻辑 `updateComposerSendReady()`（L1463）目前要求文本非空，需要改为"文本或图片有其一即可发送"。

---

## Files and Modules（改动文件）

### 功能一：侧栏搜索

| 文件 | 改动 |
|---|---|
| `chat_web_service.py` | `fetch_sessions_list()` 返回值每项增加 `search_text`（该会话全部用户消息拼接） |
| `templates/index.html` | 侧栏加搜索框 HTML/CSS；缓存 sessions 数组；输入时前端过滤渲染；无结果提示 |

### 功能二：图片上传

| 文件 | 改动 |
|---|---|
| `multi_agent_customer_service.py` | `AgentState` 增加 `customer_image` 字段；`classify_query_node` 初始化该字段并把图片写入 `persisted_dialogue` 用户轮次 |
| `multi_agents/base_agent.py` | 新增 `_build_human_message(text, state)` 辅助方法：有图片时构造多模态 HumanMessage，无图片时纯文本 |
| `multi_agents/product_agent.py` 等 5 个智能体 | 最后一条 `HumanMessage(content=...)` 改用 `self._build_human_message(..., state)`（各 1~2 行） |
| `chat_web_service.py` | `run_chat_sync()` 增加 `image` 参数 → 放入 run input 的 `customer_image`；`append_turn_from_state()` 透传轮次中的 `image` 字段 |
| `web_app.py` | `chat()` 解析并校验 `image` 字段（data URL 前缀 + 大小上限），传给业务层 |
| `templates/index.html` | 输入工具栏加图片按钮 + 隐藏 file input；选图后 canvas 压缩 + 缩略图预览（可删除）；随消息发送 base64；气泡/历史中渲染图片；发送就绪逻辑联动 |

---

## Implementation Steps（实施步骤，按依赖顺序）

### 第一阶段：后端图片链路（图状态 → LLM）

1. **`multi_agent_customer_service.py`**
   - `AgentState` 增加 `customer_image: Optional[str]`（data URL 或 None）。
   - `classify_query_node`：初始化 `state["customer_image"] = state.get("customer_image")`（无则 None）；写用户轮次到 `persisted_dialogue` 时，若有图片则在条目里增加 `"image": <data_url>`。
   - 护栏（out_of_scope）分支无需图片。
2. **`multi_agents/base_agent.py`**
   - 导入 `HumanMessage`；新增：
     ```python
     def _build_human_message(self, text: str, state: dict):
         image = (state or {}).get("customer_image")
         if image:
             return HumanMessage(content=[
                 {"type": "text", "text": text},
                 {"type": "image_url", "image_url": {"url": image}},
             ])
         return HumanMessage(content=text)
     ```
3. **5 个智能体文件**：把"当前查询"那条 `HumanMessage(content=customer_query)`（及知识库上下文那条，统一替换更简单）改为 `self._build_human_message(<原文本>, state)`。
4. **`OpenAICompatibleClient.invoke()`**：检查 list 类型 content 的透传；SystemMessage 当前被转成 user 文本（保持不动），仅确认多模态 dict 原样进入 payload；必要时加注释说明。

### 第二阶段：后端 HTTP 层

5. **`chat_web_service.py :: run_chat_sync()`**
   - 函数签名增加 `image: Optional[str] = None`；run input 增加 `"customer_image": image`（无图时不传或传 None）。
   - `stream_chat_events()` 同步增加参数（保持两个入口一致，避免遗留半截）。
6. **`chat_web_service.py :: append_turn_from_state()`**
   - 轮次 dict 中若有 `image` 键，透传到 history entry（历史接口才能返回图片）。
7. **`web_app.py :: chat()`（及 `chat_stream()`）**
   - 读取 `image = data.get('image')`；校验：必须以 `data:image/` 开头；base64 解码后大小 ≤ 5MB；非法返回 400。
   - 透传给 `run_chat_sync(message, client_session_id, image=image)`。

### 第三阶段：后端搜索字段

8. **`chat_web_service.py :: fetch_sessions_list()`**
   - 已解析的 `parsed_hist` 中取全部 `is_user=True` 的 content，用空格/换行拼接为 `search_text`（限长，如前 1000 字符），随 session 项返回。

### 第四阶段：前端搜索 UI

9. **`templates/index.html` 侧栏 HTML**：在"最近对话"标题下方插入搜索框（input + 清除 × 按钮），样式沿用现有圆角/边框变量。
10. **前端 JS**：
    - 新增模块级变量 `allSessions = []`；`refreshSessionList()` 拉到数据后先缓存再渲染。
    - `displaySessionList` 接受数组参数不变；新增 `filterSessions(keyword)`：对 `search_text`（回退 `last_user_question`）做大小写不敏感包含匹配。
    - 搜索框 `input` 事件实时过滤；空值恢复全部；无匹配显示"未找到匹配的对话"；清除按钮清空并恢复。

### 第五阶段：前端图片 UI

11. **composer HTML/CSS**：
    - 工具栏左侧（hint 之前）加图片按钮（回形针/图片 SVG）+ 隐藏 `<input type="file" accept="image/*">`。
    - textarea 上方加图片预览条容器（缩略图 + 右上角 × 删除），默认隐藏。
12. **选图与压缩 JS**：
    - 新增 `pendingImage = null`（data URL）。
    - 选中后校验类型（jpg/png/webp/gif）与原始大小（≤ 10MB）；用 `Image` + `canvas` 压缩：最长边 ≤ 1280px，JPEG quality 0.8（PNG 透明图保留 png）；输出 data URL 存入 `pendingImage`，显示缩略图。
    - × 按钮清除 `pendingImage` 并隐藏预览；发送成功后同样清空。
13. **发送联动**：
    - `updateComposerSendReady()`：`canSend = (有文本 || pendingImage) && !isTyping`。
    - `sendMessage()`：允许纯图片（文本为空时用占位文案如"（图片消息）"作为 message 以兼容分类器）；body 增加 `image: pendingImage || undefined`；用户气泡渲染文本+图片；成功后清预览。
14. **气泡/历史渲染**：
    - `addMessage(role, content, tsOpt, imageUrl)` 增加可选图片参数：在正文上方/下方插入可点击放大的缩略图（点击新窗口打开 data URL）。
    - `displaySessionHistory` 读取后端 history entry 的 `image` 字段并传入。
    - `chat_web_service` 侧栏预览 `last_user_question` 对纯图片消息显示"（图片消息）"。

### 第六阶段：验证

15. 见下方 Validation。

---

## Dependencies and Considerations（依赖与注意事项）

- 不引入任何新 Python 依赖；前端压缩用浏览器原生 canvas API，不引第三方库。
- 图片走 base64 data URL 内嵌 JSON（现有 `/api/chat` 是 JSON 契约，LangGraph run input 也是 JSON），不引入 multipart 上传，改动面最小。
- 分类器保持纯文本：纯图片消息前端补占位文案"（图片消息）"，同时建议用户附文字说明；避免每张图都产生视觉 token 分类成本。
- 五个智能体改动点已确认完全同构（10 处 HumanMessage，行号统一），可用相同模式逐个替换。
- `stream_chat_events()` 虽当前主界面未使用 SSE，仍同步加 image 参数，避免两个入口行为分叉。

---

## Validation（验证方式）

1. `.\.venv\Scripts\python.exe multi_agent_customer_service.py` 图自检通过。
2. 重启 LangGraph 服务与 Flask（注意 Windows 需 `$env:PYTHONUTF8=1`）。
3. **搜索**：造 3 个以上历史对话，搜索关键词能正确过滤；清空恢复；无结果有提示；不影响点击加载/删除会话。
4. **图片端到端**：新建对话 → 上传一张报错截图 + 文字"这个报错怎么解决" → 技术支持智能体返回与截图内容相关的回复（验证视觉链路真的生效，而非只收到文字）。
5. **纯图片消息**：不输入文字直接发图，不报错且有正常回复。
6. **历史回显**：刷新页面/切换会话后，含图对话仍显示缩略图，点击可看大图。
7. **限制校验**：超大文件（>10MB）被拒绝并有提示；非图片文件无法选择/被拒。
8. 回归：纯文本对话行为完全不变；新建对话隔离仍正常。

---

## Risks（风险与对策）

| 风险 | 对策 |
|---|---|
| base64 图片撑大 JSON / checkpointer 存储 | 前端 canvas 压缩（≤1280px、JPEG 0.8）；后端 5MB 解码后大小硬限制；checkpointer 本就有 TTL 清理（当前配置 12h） |
| 若日后换成不支持视觉的模型，多模态消息可能报错 | 业务智能体 LLM 调用已有 try/except 兜底错误文案；在计划验证步骤 4 中明确检验"回复确实与图片相关"；后续可加模型能力开关环境变量 |
| LangGraph inmem 服务对单次 run input 有 body 大小限制 | 5MB 解码 ≈ 6.7MB base64，在常见限制内；如遇 413 再下调压缩尺寸 |
| 搜索仅匹配用户消息，可能漏匹配（AI 回复含关键词但用户问题不含） | 本期按"找对话"场景取用户消息拼接（$search_text$）；如需覆盖 AI 回复，后续可在同一字段中追加全部文本，无需改前端 |
| 多图需求 | 本期支持单图（客服场景足够）；状态字段与 UI 预留为数组的扩展空间在代码注释中标注 |
