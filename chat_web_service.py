#!/usr/bin/env python3
"""
客服 Web 相关业务逻辑：LangGraph REST 调用、线程/运行、状态解析、会话列表拼装等。
与 Flask 路由解耦，便于单测与复用。
"""

from __future__ import annotations

import json
import logging
import os
import time
import datetime as _dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 配置（可被环境变量覆盖）
# -----------------------------------------------------------------------------

LANGGRAPH_API_URL: str = os.getenv("LANGGRAPH_API_URL", "http://127.0.0.1:2024").rstrip("/")
LANGGRAPH_GRAPH_NAME: str = os.getenv("LANGGRAPH_GRAPH_NAME", "customer_service")

# 连接 LangGraph 的超时（秒）
LANGGRAPH_CONNECT_TIMEOUT: float = float(os.getenv("LANGGRAPH_CONNECT_TIMEOUT", "10"))
# 两个 SSE 事件之间的最大静默时间（秒）。注意这是「每次读取」的超时，不是整个请求的总时长：
# run 若在服务端排队（例如被僵尸 run 占住 worker），连接虽已建立但永不出数据，
# 靠它主动报错，避免浏览器一直转圈无反馈。默认 120s，兼顾冷启动慢的情况。
LANGGRAPH_SILENCE_TIMEOUT: float = float(
    os.getenv("LANGGRAPH_SILENCE_TIMEOUT", os.getenv("LANGGRAPH_STREAM_READ_TIMEOUT", "120"))
)

# 助手ID缓存（创建后只读，可安全跨请求复用）
_assistant_id: Optional[str] = None

# 注意：_current_thread_id 全局变量已移除，避免多用户并发时线程ID互相覆盖。
# 线程ID现在由 ensure_thread_exists 返回，作为局部变量在请求生命周期内传递。


def get_assistant_id() -> Optional[str]:
    return _assistant_id


# -----------------------------------------------------------------------------
# 线程 state → 对话列表 / 侧栏预览
# -----------------------------------------------------------------------------

def append_turn_from_state(conversation_history: List[Dict[str, Any]], msg: Dict[str, Any]) -> None:
    """从状态中的单条消息追加到会话历史列表；仅在状态里带有 timestamp 时写入条目。"""
    content = msg.get("content", "") or ""
    if not content:
        return
    is_user = bool(msg.get("is_user", False))
    entry: Dict[str, Any] = {
        "is_user": is_user,
        "content": content,
        "role": "user" if is_user else "assistant",
    }
    ts = msg.get("timestamp")
    if ts is not None and ts != "":
        entry["timestamp"] = ts
    # 客户上传的图片列表（data URL）随轮次透传，供前端历史回显
    images = msg.get("images")
    if images:
        entry["images"] = images
    conversation_history.append(entry)


def conversation_history_from_state_data(state_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从 LangGraph 线程 state JSON 解析对话列表。"""
    conversation_history: List[Dict[str, Any]] = []
    if not isinstance(state_data, dict):
        return conversation_history

    if "values" in state_data and isinstance(state_data["values"], dict):
        values = state_data["values"]

        source_turns = None
        filled_from_turn_list = False
        pd_raw = values.get("persisted_dialogue")
        ch_raw = values.get("conversation_history")
        if isinstance(pd_raw, list) and len(pd_raw) > 0:
            source_turns = pd_raw
        elif isinstance(ch_raw, list) and len(ch_raw) > 0:
            source_turns = ch_raw

        if source_turns is not None:
            filled_from_turn_list = True
            for msg in source_turns:
                if isinstance(msg, dict):
                    append_turn_from_state(conversation_history, msg)

        elif "messages" in values:
            for message in values["messages"]:
                role = message.get("role", "user")
                content = message.get("content", "")
                if content:
                    is_user = role == "user"
                    row: Dict[str, Any] = {
                        "is_user": is_user,
                        "content": content,
                        "role": role,
                    }
                    mt = message.get("timestamp")
                    if mt is not None and mt != "":
                        row["timestamp"] = mt
                    conversation_history.append(row)

        # 已有 persisted_dialogue / conversation_history 时不再追加 values.response：
        # 助手正文已在轮次里；final_response_node 还可能给 response 加前缀导致去重失败、出现双线助手气泡。
        if "response" in values and values["response"]:
            response_content = values["response"]
            if not filled_from_turn_list:
                if not any(
                    msg["content"] == response_content and not msg["is_user"]
                    for msg in conversation_history
                ):
                    conversation_history.append({
                        "is_user": False,
                        "content": response_content,
                        "role": "assistant"
                    })

    elif "messages" in state_data:
        for message in state_data["messages"]:
            role = message.get("role", "user")
            content = message.get("content", "")
            if content:
                is_user = role == "user"
                row = {
                    "is_user": is_user,
                    "content": content,
                    "role": role,
                }
                mt = message.get("timestamp")
                if mt is not None and mt != "":
                    row["timestamp"] = mt
                conversation_history.append(row)

    return conversation_history


def last_user_question_from_history(conversation_history: List[Dict[str, Any]]) -> str:
    """取最后一条用户消息的纯文本（用于侧栏预览）。"""
    for msg in reversed(conversation_history):
        if not msg.get("is_user"):
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = str(content) if content is not None else ""
        s = content.strip()
        if s:
            return s
    return ""


def extract_ai_response(thread_state: Dict[str, Any]) -> str:
    """从线程状态中提取 AI 回复文本。"""
    try:
        if "values" in thread_state and isinstance(thread_state["values"], dict):
            values = thread_state["values"]

            if "response" in values and values["response"]:
                return str(values["response"])

            if "messages" in values:
                for message in values["messages"]:
                    if message.get("role") == "assistant":
                        content = message.get("content", "")
                        if content:
                            return str(content)

        return "抱歉，我无法理解您的问题。"

    except Exception as e:
        logger.exception("提取AI回复时出错")
        return "抱歉，处理您的请求时出现了错误。"


# -----------------------------------------------------------------------------
# 助手 / 线程
# -----------------------------------------------------------------------------

def ensure_assistant_exists() -> Tuple[bool, Optional[str]]:
    """确保 LangGraph 助手存在，不存在则创建。返回 (成功标志, assistant_id)。"""
    global _assistant_id

    try:
        search_response = requests.post(
            f"{LANGGRAPH_API_URL}/assistants/search",
            json={
                "graph_id": LANGGRAPH_GRAPH_NAME,
                "limit": 1
            },
            timeout=10
        )

        if search_response.status_code == 200:
            assistants = search_response.json()
            if assistants:
                _assistant_id = assistants[0]["assistant_id"]
                logger.info("找到现有助手: %s", _assistant_id)
                return True, _assistant_id

        create_response = requests.post(
            f"{LANGGRAPH_API_URL}/assistants",
            json={
                "graph_id": LANGGRAPH_GRAPH_NAME,
                "name": "Customer Service Assistant",
                "description": "Multi-agent customer service system"
            },
            timeout=10
        )

        if create_response.status_code == 200:
            result = create_response.json()
            _assistant_id = result["assistant_id"]
            logger.info("创建新助手: %s", _assistant_id)
            return True, _assistant_id
        else:
            logger.error("创建助手失败: %s", create_response.status_code)
            return False, None

    except Exception as e:
        logger.exception("确保助手存在时出错")
        return False, None


def ensure_thread_exists(client_session_id: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """
    确保有可用的 LangGraph 线程。
    - client_session_id 是已存在的合法线程 ID：复用该线程（继续/切换历史会话）。
    - client_session_id 为空、'default' 或非法：一律【新建】线程，
      不复用全局缓存，保证每次"新建对话"相互隔离。
    返回 (成功标志, thread_id)。
    """
    sid = client_session_id
    if sid and sid != 'default':
        try:
            thread_response = requests.get(
                f"{LANGGRAPH_API_URL}/threads/{sid}",
                timeout=5
            )
            if thread_response.status_code == 200:
                return True, sid
            else:
                logger.warning("会话ID %s 不是有效的LangGraph线程ID，将创建新线程", sid)
        except Exception as e:
            logger.warning("验证会话ID %s 时出错: %s，将创建新线程", sid, e)

    try:
        response = requests.post(
            f"{LANGGRAPH_API_URL}/threads",
            json={},
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            thread_id = result["thread_id"]
            logger.info("创建新线程: %s", thread_id)
            return True, thread_id
        else:
            logger.error("创建线程失败: %s", response.status_code)
            return False, None

    except Exception as e:
        logger.exception("确保线程存在时出错")
        return False, None


def _normalize_created_at(created_at: Any) -> float:
    if isinstance(created_at, str):
        try:
            dt = _dt.datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            return dt.timestamp()
        except Exception:
            return time.time()
    if isinstance(created_at, (int, float)) and created_at > 0:
        return float(created_at)
    return time.time()


def _message_count_from_state_data(state_data: Dict[str, Any]) -> int:
    if "values" in state_data and isinstance(state_data["values"], dict):
        values = state_data["values"]
        if "conversation_history" in values and values["conversation_history"]:
            return len(values["conversation_history"])
        if "messages" in values:
            return len(values["messages"])
        if "response" in values and values["response"]:
            return 1
    if "messages" in state_data:
        return len(state_data["messages"])
    return 0


def _fetch_thread_state_info(thread_id: str) -> Tuple[int, str, str]:
    """拉取单个线程的 state 并解析出 (message_count, last_user_question, search_text)。"""
    message_count = 0
    last_user_question = ""
    search_text = ""
    try:
        state_response = requests.get(
            f"{LANGGRAPH_API_URL}/threads/{thread_id}/state",
            timeout=5
        )
        if state_response.status_code == 200:
            state_data = state_response.json()
            parsed_hist = conversation_history_from_state_data(state_data)
            last_user_question = last_user_question_from_history(parsed_hist)
            message_count = _message_count_from_state_data(state_data)
            # 全部用户消息拼接，供侧栏搜索做前端包含匹配（限长 1000 字符）
            user_texts = [
                str(m.get("content", "")).strip()
                for m in parsed_hist
                if m.get("is_user") and str(m.get("content", "")).strip()
            ]
            search_text = "\n".join(user_texts)[:1000]
    except Exception:
        message_count = 0
        search_text = ""
    return message_count, last_user_question, search_text


def fetch_sessions_list() -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    拉取线程列表并拼装前端会话项。
    成功返回 (sessions, None)，失败返回 (None, error_message)。
    """
    try:
        response = requests.post(f"{LANGGRAPH_API_URL}/threads/search", json={})

        if response.status_code != 200:
            logger.error("获取线程列表失败: %s", response.status_code)
            return None, f'获取会话列表失败: {response.status_code}'

        threads = response.json()
        sessions: List[Dict[str, Any]] = []

        # 并发拉取各线程 state，避免串行 N 次请求导致的长延迟
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_map = {
                executor.submit(_fetch_thread_state_info, t.get("thread_id", "")): t
                for t in threads
            }
            for future in as_completed(future_map):
                thread = future_map[future]
                thread_id = thread.get("thread_id", "")
                created_at = _normalize_created_at(thread.get("created_at", time.time()))
                try:
                    message_count, last_user_question, search_text = future.result()
                except Exception:
                    message_count, last_user_question, search_text = 0, "", ""
                sessions.append({
                    "session_id": thread_id,
                    "created_at": created_at,
                    "message_count": message_count,
                    "last_user_question": last_user_question,
                    "search_text": search_text,
                })

        return sessions, None

    except Exception as e:
        logger.exception("获取会话列表时出错")
        return None, f'服务器错误: {str(e)}'


def fetch_session_detail(session_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """获取单个线程详情 + 对话历史。成功返回 (payload, None)。"""
    try:
        response = requests.get(f"{LANGGRAPH_API_URL}/threads/{session_id}")

        if response.status_code != 200:
            logger.error("获取线程详情失败: %s", response.status_code)
            return None, f'获取会话详情失败: {response.status_code}'

        thread_data = response.json()
        conversation_history: List[Dict[str, Any]] = []

        try:
            state_response = requests.get(
                f"{LANGGRAPH_API_URL}/threads/{session_id}/state",
                timeout=5
            )
            if state_response.status_code == 200:
                state_data = state_response.json()
                conversation_history = conversation_history_from_state_data(state_data)
            else:
                logger.warning("获取线程状态失败: %s", state_response.status_code)
        except Exception as e:
            logger.exception("获取线程状态时出错")

        session_data = {
            "session_id": session_id,
            "created_at": thread_data.get("created_at", time.time()),
            "conversation_history": conversation_history
        }
        return session_data, None

    except Exception as e:
        logger.exception("获取会话详情时出错")
        return None, f'服务器错误: {str(e)}'


def delete_remote_thread(thread_id: str) -> Tuple[bool, int]:
    """删除 LangGraph 线程。成功为任意 2xx（DELETE 常为 204 No Content）。"""
    response = requests.delete(f"{LANGGRAPH_API_URL}/threads/{thread_id}", timeout=10)
    ok = 200 <= response.status_code < 300
    return ok, response.status_code


def clear_thread_and_create_new(thread_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    删除旧线程并在服务端新建线程。
    成功返回 (new_thread_id, None)，失败返回 (None, error)。
    """
    ok, status = delete_remote_thread(thread_id)
    if not ok:
        return None, f'清空会话失败: {status}'

    new_thread_response = requests.post(
        f"{LANGGRAPH_API_URL}/threads",
        json={},
        timeout=10
    )

    if new_thread_response.status_code != 200:
        return None, '创建新线程失败'

    new_thread_id = new_thread_response.json()["thread_id"]
    return new_thread_id, None


# -----------------------------------------------------------------------------
# 一次聊天运行（阻塞轮询）
# -----------------------------------------------------------------------------

def run_chat_sync(
    user_message: str,
    client_session_id: Optional[str] = None,
    images: Optional[List[str]] = None,
) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[str]]:
    """
    在当前线程上提交一轮用户消息并等待完成。
    client_session_id: 前端传入的会话 ID（可为 LangGraph 线程 ID）。
    images: 可选的客户图片 data URL 列表（"data:image/...;base64,..."），随输入传给工作流。
    返回 (ai_text, error_text, http_status_optional, thread_id)。
    """
    if not user_message.strip():
        return None, '消息不能为空', 400, None

    asst_ok, assistant_id = ensure_assistant_exists()
    if not asst_ok:
        return None, '无法创建或找到助手', 500, None

    thread_ok, thread_id = ensure_thread_exists(client_session_id)
    if not thread_ok:
        return None, '无法创建线程', 500, None

    assert assistant_id and thread_id

    # 工作流输入（含可选客户图片列表）
    run_input: Dict[str, Any] = {
        "messages": [
            {
                "role": "user",
                "content": user_message.strip()
            }
        ],
        "customer_query": user_message.strip(),
        "session_id": thread_id
    }
    if images:
        run_input["customer_images"] = images

    logger.info("run_input customer_query=%r session_id=%s", user_message.strip()[:80], thread_id)

    try:
        run_resp = requests.post(
            f"{LANGGRAPH_API_URL}/threads/{thread_id}/runs",
            json={
                "assistant_id": assistant_id,
                "input": run_input
            },
            timeout=30
        )

        if run_resp.status_code != 200:
            logger.error("创建运行失败: %s", run_resp.status_code)
            return None, f'调用失败: {run_resp.status_code}', run_resp.status_code, thread_id

        result = run_resp.json()
        run_id = result["run_id"]

        run_status = "running"
        max_wait_time = 120
        wait_start = time.time()

        while run_status in ["running", "pending"]:
            if time.time() - wait_start > max_wait_time:
                logger.warning("运行超时，已等待 %s 秒", max_wait_time)
                return None, '运行超时', 500, thread_id

            time.sleep(0.5)
            status_response = requests.get(
                f"{LANGGRAPH_API_URL}/threads/{thread_id}/runs/{run_id}",
                timeout=10
            )
            if status_response.status_code != 200:
                logger.error("获取运行状态失败: %s", status_response.status_code)
                break

            run_data = status_response.json()
            run_status = run_data.get("status", "unknown")

            if run_status in ["completed", "success"]:
                thread_response = requests.get(
                    f"{LANGGRAPH_API_URL}/threads/{thread_id}/state",
                    timeout=10
                )
                if thread_response.status_code == 200:
                    thread_state = thread_response.json()
                    ai_response = extract_ai_response(thread_state)
                    return ai_response, None, None, thread_id
                else:
                    logger.error("获取线程状态失败: %s", thread_response.status_code)
                    return None, '无法获取线程状态', 500, thread_id

            if run_status in ["failed", "cancelled"]:
                logger.error("运行失败: %s", run_status)
                return None, f'运行失败: {run_status}', 500, thread_id

        return None, '运行超时', 500, thread_id

    except Exception as e:
        logger.exception("聊天处理错误")
        return None, f'内部错误: {str(e)}', 500, thread_id


def _sse(payload: Dict[str, Any]) -> str:
    """把字典编码为一行 SSE data（ensure_ascii=False，便于日志直读中文）。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _iter_langgraph_sse(response) -> Iterator[Tuple[str, Any]]:
    """
    解析 LangGraph /runs/stream 返回的 SSE 流，产出 (event_name, data_obj)。

    SSE 事件块以空行分隔，字段形如 `event: custom` / `data: {...}`。
    自行按 utf-8 解码 bytes，避免 decode_unicode 猜错编码导致中文乱码。
    """
    event_name: Optional[str] = None
    data_lines: List[str] = []

    def _flush(evt: Optional[str], lines: List[str]) -> Tuple[str, Any]:
        raw = "\n".join(lines)
        try:
            return (evt or "message"), json.loads(raw)
        except ValueError:
            return (evt or "message"), raw

    for raw_line in response.iter_lines():
        if raw_line is None:
            continue
        line = raw_line.decode("utf-8", errors="ignore") if isinstance(raw_line, bytes) else raw_line

        if line == "":
            if data_lines:
                yield _flush(event_name, data_lines)
            event_name, data_lines = None, []
            continue
        if line.startswith(":"):
            # 注释行（心跳），忽略
            continue
        if line.startswith("event:"):
            event_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())

    # 流末尾若没有以空行收尾，补发最后一个事件
    if data_lines:
        yield _flush(event_name, data_lines)


def stream_chat_events(
    user_message: str,
    client_session_id: Optional[str] = None,
    images: Optional[List[str]] = None,
) -> Iterable[str]:
    """
    以 SSE 转发 LangGraph 的流式运行结果，实现真正的逐 token 推流。

    走 POST /threads/{tid}/runs/stream，stream_mode 取：
    - "custom" ：graph 节点内用 get_stream_writer() 写入的 token 片段
    - "updates"：节点增量状态，用于取最终完整回复 / current_agent / query_type 兜底

    推送事件格式：
    - {"type": "token", "content": "..."}                 逐块正文
    - {"type": "done", "content": "完整正文", "session_id": ..., "thread_id": ...,
       "agent": ..., "query_type": ...}                   收尾
    - {"error": "..."}                                    错误
    - data: [DONE]                                        流结束标记

    健壮性：/runs/stream 不可用、或一个 token 都没收到时，自动退回
    _stream_chat_events_by_polling（原轮询实现），保证行为不退化。
    """
    if not user_message.strip():
        yield _sse({'error': '消息不能为空'})
        yield "data: [DONE]\n\n"
        return

    # 立即吐一个事件：werkzeug 只有在生成器「首次 yield」时才会写响应头。
    # 若不在最前面发一个，下面「助手校验 → 线程校验 → 提交 run」期间后端一旦阻塞
    # （典型场景：服务端 run 在队列里排队），浏览器会一直处于「请求 pending、毫无反馈」，
    # 看起来像前端卡死。先发一个事件即可让响应头立刻返回，前端马上有反馈。
    yield _sse({"type": "status", "content": "正在接入客服…"})

    asst_ok, assistant_id = ensure_assistant_exists()
    if not asst_ok:
        yield _sse({'error': '无法创建或找到助手'})
        yield "data: [DONE]\n\n"
        return

    thread_ok, thread_id = ensure_thread_exists(client_session_id)
    if not thread_ok:
        yield _sse({'error': '无法创建线程'})
        yield "data: [DONE]\n\n"
        return

    assert assistant_id is not None and thread_id is not None

    # 工作流输入（含可选客户图片列表）
    stream_run_input: Dict[str, Any] = {
        "messages": [{"role": "user", "content": user_message.strip()}],
        "customer_query": user_message.strip(),
        "session_id": thread_id
    }
    if images:
        stream_run_input["customer_images"] = images

    sid = thread_id
    streamed_any = False
    # run 是否已成功提交：提交成功后绝不再回退轮询重跑（重跑会再排一个 run，加重服务端排队）
    run_submitted = False
    accumulated: List[str] = []
    fallback_text: Optional[str] = None
    final_agent = ""
    final_query_type = ""

    try:
        response = requests.post(
            f"{LANGGRAPH_API_URL}/threads/{sid}/runs/stream",
            json={
                "assistant_id": assistant_id,
                "input": stream_run_input,
                "stream_mode": ["custom", "updates"],
            },
            timeout=(LANGGRAPH_CONNECT_TIMEOUT, LANGGRAPH_SILENCE_TIMEOUT),
            stream=True,
        )

        if response.status_code != 200:
            # 提交阶段即失败：此时服务端没有我们的 run 在跑，回退轮询是安全的
            logger.warning("runs/stream 返回 %s，回退到轮询式实现", response.status_code)
            response.close()
            yield from _stream_chat_events_by_polling(user_message, sid, assistant_id, stream_run_input)
            return

        # 从这里开始服务端已有我们的 run，后续任何失败都不再重跑
        run_submitted = True

        try:
            for event_name, payload in _iter_langgraph_sse(response):
                if event_name == "custom":
                    if not isinstance(payload, dict):
                        continue
                    evt_type = payload.get("type")
                    if evt_type == "status":
                        # 阶段性状态（如「正在识别您的问题类型…」）：原样透传给前端展示。
                        # 不计入正文，也不影响 streamed_any（否则会出现空气泡）
                        yield _sse({"type": "status", "content": payload.get("content") or ""})
                    elif evt_type == "token":
                        piece = payload.get("content")
                        if piece:
                            streamed_any = True
                            accumulated.append(str(piece))
                            yield _sse({"type": "token", "content": piece})
                elif event_name.startswith("update"):
                    # payload 形如 {node_name: state_delta}
                    if not isinstance(payload, dict):
                        continue
                    for node_name, delta in payload.items():
                        if not isinstance(delta, dict):
                            continue
                        resp_text = delta.get("response")
                        if node_name == "final_response":
                            # final_response 会给正文加【xx's Response】前缀：
                            # 仅在拿不到更干净的 agent 原始回复时才用它兜底
                            if resp_text and not fallback_text:
                                fallback_text = str(resp_text)
                            continue
                        if resp_text:
                            fallback_text = str(resp_text)
                            final_agent = delta.get("current_agent") or final_agent
                            final_query_type = delta.get("query_type") or final_query_type
        finally:
            response.close()

    except Exception as e:
        # 注意：requests 在 iter_lines 内部会把 ReadTimeout 包装成
        # ConnectionError("Read timed out.")，因此按「类型 + 文案」双重判定，否则会漏判。
        is_timeout = isinstance(e, requests.exceptions.Timeout) or "timed out" in str(e).lower()

        if is_timeout and not streamed_any:
            # 静默超时：连接已建立但服务端长时间不发任何数据。
            # 最常见原因是 run 在服务端队列里排队（例如被僵尸 run 占住 worker）。
            logger.warning(
                "流式静默超时：%.0fs 内未收到任何事件（%s）",
                LANGGRAPH_SILENCE_TIMEOUT, type(e).__name__,
            )
            yield _sse({
                'error': f'客服响应超时（{int(LANGGRAPH_SILENCE_TIMEOUT)} 秒内无响应，可能正在排队），请稍后重试'
            })
            yield "data: [DONE]\n\n"
            return

        logger.exception("流式转发出错")
        if not streamed_any and not run_submitted:
            # run 从未提交成功 → 回退轮询是安全的
            logger.warning("流式未产出任何内容且 run 未提交，回退到轮询式实现")
            yield from _stream_chat_events_by_polling(user_message, sid, assistant_id, stream_run_input)
            return
        if not streamed_any:
            # run 已提交但没有任何产出：不再重跑（重跑会再排一个 run，加重服务端排队）
            logger.warning("run 已提交但无任何产出，直接报错不再重跑")
            yield _sse({'error': '客服处理超时或服务繁忙，请稍后重试'})
            yield "data: [DONE]\n\n"
            return
        # 已推给前端部分正文：用已累积内容收尾，绝不重跑（否则前端会看到重复文本）
        yield _sse({
            "type": "done",
            "content": "".join(accumulated),
            "session_id": sid,
            "thread_id": sid,
            "agent": final_agent,
            "query_type": final_query_type,
            "truncated": True,
        })
        yield "data: [DONE]\n\n"
        return

    full_text = "".join(accumulated)
    if not full_text:
        if not fallback_text:
            # 既无 token 也无最终文本：退回轮询实现再取一次完整回复
            logger.warning("流式通道无任何输出，回退到轮询式实现")
            yield from _stream_chat_events_by_polling(user_message, sid, assistant_id, stream_run_input)
            return
        # 有完整文本但没有流式片段（例：LLM 走了非流式回退）：整段补发一次
        full_text = fallback_text
        yield _sse({"type": "token", "content": full_text})

    yield _sse({
        "type": "done",
        "content": full_text,
        "session_id": sid,
        "thread_id": sid,
        "agent": final_agent,
        "query_type": final_query_type,
    })
    yield "data: [DONE]\n\n"


def _stream_chat_events_by_polling(
    user_message: str,
    thread_id: str,
    assistant_id: str,
    run_input: Dict[str, Any],
) -> Iterable[str]:
    """
    兜底实现：提交 run 后轮询运行状态，结束后一次性取回完整回复。

    仅在 /runs/stream 不可用、或流式通道一个 token 都没产出时调用，
    保证「真流式不可用时功能不退化」。事件格式与 stream_chat_events 一致。
    """
    tid = thread_id
    # 告知前端已切入兼容模式，避免等待期毫无反馈
    yield _sse({"type": "status", "content": "正在切换兼容模式处理…"})

    try:
        response = requests.post(
            f"{LANGGRAPH_API_URL}/threads/{tid}/runs",
            json={"assistant_id": assistant_id, "input": run_input},
            timeout=30,
        )

        if response.status_code != 200:
            yield _sse({'error': f'流式调用失败: {response.status_code}'})
            yield "data: [DONE]\n\n"
            return

        run_id = response.json().get("run_id")
        if not run_id:
            yield _sse({'error': '无法获取运行ID'})
            yield "data: [DONE]\n\n"
            return

        run_status = "running"
        max_wait_time = 120
        wait_start = time.time()

        while run_status in ["running", "pending"]:
            if time.time() - wait_start > max_wait_time:
                yield _sse({'error': '运行超时'})
                yield "data: [DONE]\n\n"
                return

            time.sleep(0.5)
            status_response = requests.get(
                f"{LANGGRAPH_API_URL}/threads/{tid}/runs/{run_id}",
                timeout=10
            )
            if status_response.status_code != 200:
                yield _sse({'error': f'获取运行状态失败: {status_response.status_code}'})
                break

            run_status = status_response.json().get("status", "unknown")

            if run_status in ["completed", "success"]:
                thread_response = requests.get(
                    f"{LANGGRAPH_API_URL}/threads/{tid}/state",
                    timeout=10
                )
                if thread_response.status_code == 200:
                    ai_response = extract_ai_response(thread_response.json())
                    if ai_response:
                        yield _sse({"type": "token", "content": ai_response})
                    yield _sse({
                        "type": "done",
                        "content": ai_response or "",
                        "session_id": tid,
                        "thread_id": tid,
                    })
                else:
                    yield _sse({'error': '无法获取线程状态'})
                break

            if run_status in ["failed", "cancelled"]:
                yield _sse({'error': f'运行失败: {run_status}'})
                break
        else:
            yield _sse({'error': '运行超时'})

    except Exception as e:
        logger.exception("轮询式流式处理错误")
        yield _sse({'error': f'流式处理错误: {str(e)}'})

    yield "data: [DONE]\n\n"


def langgraph_connectivity_test() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    探测 LangGraph 服务与搜索接口。
    成功返回 (result_dict, None)，失败返回 (None, error_message)。
    """
    try:
        health_check_status = 0
        try:
            ok_response = requests.get(f"{LANGGRAPH_API_URL}/ok", timeout=5)
            health_check_status = ok_response.status_code
        except requests.exceptions.RequestException as e:
            logger.warning("LangGraph GET /ok 失败: %s", e)

        threads_response = requests.post(f"{LANGGRAPH_API_URL}/threads/search", json={}, timeout=10)
        assistants_response = requests.post(f"{LANGGRAPH_API_URL}/assistants/search", json={}, timeout=10)

        if not (200 <= health_check_status < 300) and (200 <= threads_response.status_code < 300):
            logger.warning("LangGraph GET /ok 未成功，但 threads/search 正常，健康检查标记为通过")
            health_check_status = 200

        return ({
            'status': 'test_completed',
            'health_check': health_check_status,
            'threads_search': threads_response.status_code,
            'assistants_search': assistants_response.status_code,
            'details': {
                'ok_response': 'OK' if 200 <= health_check_status < 300 else (health_check_status or 'unreachable'),
                'threads_response': threads_response.text if threads_response.status_code != 200 else 'OK',
                'assistants_response': assistants_response.text if assistants_response.status_code != 200 else 'OK'
            }
        }, None)

    except Exception as e:
        logger.exception("测试 LangGraph API 时出错")
        return None, f'测试失败: {str(e)}'
