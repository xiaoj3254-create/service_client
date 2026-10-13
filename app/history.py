"""线程 state 解析的纯函数（自 chat_web_service.py 迁移，保持既有测试语义）。

- conversation_history_from_state_data：从线程 state 解析对话列表
- last_user_question_from_history：侧栏预览
- extract_ai_response：最终回复提取
- message_count_from_state_data：消息条数统计
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


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
    """从线程 state（values 字典或 REST JSON）解析对话列表。

    解析优先级：persisted_dialogue > conversation_history > messages。
    """
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
        # 助手正文已在轮次里，避免去重失败导致双线助手气泡。
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

    except Exception:
        logger.exception("提取AI回复时出错")
        return "抱歉，处理您的请求时出现了错误。"


def message_count_from_state_data(state_data: Dict[str, Any]) -> int:
    """从线程 state 解析消息条数（与 conversation_history_from_state_data 优先级对齐）。"""
    if not isinstance(state_data, dict):
        return 0
    if "values" in state_data and isinstance(state_data["values"], dict):
        values = state_data["values"]
        if isinstance(values.get("persisted_dialogue"), list):
            return len(values["persisted_dialogue"])
        if "conversation_history" in values and values["conversation_history"]:
            return len(values["conversation_history"])
        if "messages" in values:
            return len(values["messages"])
        if "response" in values and values["response"]:
            return 1
    if "messages" in state_data:
        return len(state_data["messages"])
    return 0
