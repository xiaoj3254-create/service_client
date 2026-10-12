"""进程内运行 LangGraph 图：初始化、线程保障、流式/阻塞回复、统计回写。

架构变化（2026-10 FastAPI 迁移）：
- 不再经 langgraph dev REST（原 chat_web_service.py 的轮询/降级/僵尸 run 防御随之删除）；
- checkpointer：有 DATABASE_URL 用 AsyncPostgresSaver（跨进程持久），否则进程内 InMemorySaver；
- 会话归属由 app_threads 注册表承载（metadata.user_id 语义平移）。
"""

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from app.config import AINVOKE_TIMEOUT, DATABASE_URL, SESSION_LIST_LIMIT
from app.db import OWNERSHIP_ERROR, get_threads_store, new_thread_id
from app.history import (
    conversation_history_from_state_data,
    last_user_question_from_history,
    message_count_from_state_data,
)

logger = logging.getLogger(__name__)

_graph: Any = None
_saver_cm: Any = None
_graph_lock = asyncio.Lock()


async def get_graph() -> Any:
    """进程级单例图：checkpointer 按部署模式惰性创建。"""
    global _graph, _saver_cm
    if _graph is not None:
        return _graph
    async with _graph_lock:
        if _graph is not None:
            return _graph
        from multi_agent_customer_service import make_graph

        if DATABASE_URL:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            _saver_cm = AsyncPostgresSaver.from_conn_string(DATABASE_URL)
            saver = await _saver_cm.__aenter__()
            await saver.setup()  # 首次启动建 checkpoint 表
            _graph = make_graph(checkpointer=saver)
            logger.info("已启用 AsyncPostgresSaver（持久化到 Postgres）")
        else:
            # make_graph(None) → 进程内 InMemorySaver（仅开发/测试语义成立）
            _graph = make_graph(checkpointer=None)
            logger.warning("未配置 DATABASE_URL，使用进程内存储（仅限开发/测试，单 worker）")
        return _graph


async def close_graph() -> None:
    global _graph, _saver_cm
    if _saver_cm is not None:
        try:
            await _saver_cm.__aexit__(None, None, None)
        except Exception:
            logger.exception("关闭 checkpointer 连接失败")
        _saver_cm = None
    _graph = None


async def ensure_thread(client_session_id: Optional[str], user_id: str) -> str:
    """
    确保有可用的线程，并按用户隔离（与原 ensure_thread_exists 语义一致）：
    - client_session_id 是当前用户名下线程 → 复用；
    - 空 / 'default' / 不存在 / 归属他人 → 新建（绝不透传他人线程）。
    返回 thread_id。
    """
    store = get_threads_store()
    sid = client_session_id
    if sid and sid != 'default':
        row = await store.get(sid, user_id)
        if row:
            return sid
        logger.warning("会话ID %s 不是当前用户的有效线程，将创建新线程", sid)

    tid = new_thread_id()
    await store.create(tid, user_id)
    logger.info("创建新线程: %s (user_id=%s)", tid, user_id)
    return tid


def build_run_input(user_message: str, images: Optional[List[str]], thread_id: str) -> Dict[str, Any]:
    """工作流输入；customer_images 无图时也显式写 None，防止上一轮图片残留。"""
    return {
        "messages": [{"role": "user", "content": user_message}],
        "customer_query": user_message,
        "customer_images": images,
        "session_id": thread_id,
    }


def _config(thread_id: str) -> Dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


async def update_thread_stats(thread_id: str) -> None:
    """从图 state 回写线程统计（消息数/预览/搜索文本），供侧栏列表与搜索。"""
    try:
        graph = await get_graph()
        snap = await graph.aget_state(_config(thread_id))
        values = snap.values or {}
        state_data = {"values": values}
        hist = conversation_history_from_state_data(state_data)
        count = message_count_from_state_data(state_data)
        last_q = last_user_question_from_history(hist)
        search_text = "\n".join(
            str(m.get("content", "")).strip()
            for m in hist
            if m.get("is_user") and str(m.get("content", "")).strip()
        )[:1000]
        await get_threads_store().update_stats(
            thread_id,
            message_count=count,
            last_user_question=last_q,
            search_text=search_text,
        )
    except Exception:
        logger.exception("回写线程统计失败: %s", thread_id)


def _sse(payload: Dict[str, Any]) -> str:
    """把字典编码为一行 SSE data（ensure_ascii=False）。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


_DONE = "data: [DONE]\n\n"


async def stream_reply(
    user_message: str,
    thread_id: str,
    images: Optional[List[str]],
    user_id: str,
) -> AsyncIterator[str]:
    """
    以 SSE 转发进程内图的流式运行，事件协议与前端既有实现一致：
    - {"type": "status"|"token", "content": ...}
    - {"type": "done", "content": 完整正文, "session_id", "thread_id", "agent", "query_type"}
    - {"error": "..."} 与 data: [DONE] 结束标记
    """
    yield _sse({"type": "status", "content": "正在接入客服…"})

    accumulated: List[str] = []
    fallback_text: Optional[str] = None
    final_agent = ""
    final_query_type = ""

    try:
        graph = await get_graph()
        async for mode, chunk in graph.astream(
            build_run_input(user_message, images, thread_id),
            config=_config(thread_id),
            stream_mode=["custom", "updates"],
        ):
            if mode == "custom" and isinstance(chunk, dict):
                evt_type = chunk.get("type")
                if evt_type == "status":
                    yield _sse({"type": "status", "content": chunk.get("content") or ""})
                elif evt_type == "token":
                    piece = chunk.get("content")
                    if piece:
                        accumulated.append(str(piece))
                        yield _sse({"type": "token", "content": piece})
            elif mode == "updates" and isinstance(chunk, dict):
                # payload 形如 {node_name: state_delta}
                for node_name, delta in chunk.items():
                    if not isinstance(delta, dict):
                        continue
                    resp_text = delta.get("response")
                    if node_name == "final_response":
                        # final_response 只是路由收尾：仅在拿不到业务智能体原始回复时兜底
                        if resp_text and not fallback_text:
                            fallback_text = str(resp_text)
                        continue
                    if resp_text:
                        fallback_text = str(resp_text)
                        final_agent = delta.get("current_agent") or final_agent
                        final_query_type = delta.get("query_type") or final_query_type

    except Exception as e:
        logger.exception("进程内图流式运行出错")
        if accumulated:
            # 已推给前端部分正文：用已累积内容收尾，绝不重跑
            yield _sse({
                "type": "done",
                "content": "".join(accumulated),
                "session_id": thread_id,
                "thread_id": thread_id,
                "agent": final_agent,
                "query_type": final_query_type,
                "truncated": True,
            })
            yield _DONE
            return
        yield _sse({"error": "客服处理超时或服务繁忙，请稍后重试"})
        logger.warning("流式零产出失败（thread=%s）: %s", thread_id, type(e).__name__)
        yield _DONE
        return

    full_text = "".join(accumulated) or (fallback_text or "")
    if not full_text:
        # 图正常结束但零输出：直接读线程 state 取最终回复（结果就在 checkpoint 里）
        try:
            graph = await get_graph()
            snap = await graph.aget_state(_config(thread_id))
            from app.history import extract_ai_response
            full_text = extract_ai_response({"values": snap.values or {}})
            if full_text == "抱歉，我无法理解您的问题。":
                full_text = ""  # state 里确实没有回复
        except Exception:
            logger.exception("读取线程状态时出错")
            full_text = ""
        if not full_text:
            yield _sse({"error": "客服暂时没有生成回复，请稍后重试"})
            yield _DONE
            return
        yield _sse({"type": "token", "content": full_text})

    await update_thread_stats(thread_id)
    yield _sse({
        "type": "done",
        "content": full_text,
        "session_id": thread_id,
        "thread_id": thread_id,
        "agent": final_agent,
        "query_type": final_query_type,
    })
    yield _DONE


async def run_reply(
    user_message: str,
    thread_id: str,
    images: Optional[List[str]],
    user_id: str,
) -> str:
    """非流式单轮：ainvoke 直接跑完整图，返回最终回复文本。"""
    graph = await get_graph()
    try:
        result = await asyncio.wait_for(
            graph.ainvoke(build_run_input(user_message, images, thread_id), config=_config(thread_id)),
            timeout=AINVOKE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning("非流式运行超时（%ss）: thread=%s", AINVOKE_TIMEOUT, thread_id)
        raise

    await update_thread_stats(thread_id)
    return str((result or {}).get("response") or "")


async def fetch_session_detail_state(thread_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """获取当前用户线程详情 + 对话历史；越权返回 None 并携带 OWNERSHIP_ERROR 语义由路由层处理。"""
    store = get_threads_store()
    row = await store.get(thread_id, user_id)
    if not row:
        return None

    graph = await get_graph()
    conversation_history: List[Dict[str, Any]] = []
    try:
        snap = await graph.aget_state(_config(thread_id))
        conversation_history = conversation_history_from_state_data({"values": snap.values or {}})
    except Exception:
        logger.exception("获取线程状态时出错")

    return {
        "session_id": thread_id,
        "created_at": row.get("created_at"),
        "conversation_history": conversation_history,
    }


async def delete_thread(thread_id: str, user_id: str) -> bool:
    """删除当前用户线程：先删注册表行（归属校验），再清理 checkpoint 数据。"""
    store = get_threads_store()
    if not await store.delete(thread_id, user_id):
        logger.warning("拒绝删除不属于当前用户的会话 %s", thread_id)
        return False
    try:
        graph = await get_graph()
        await graph.checkpointer.adelete_thread(thread_id)
    except Exception:
        # 注册表已删，会话不再可见；checkpoint 清理失败仅记录，不阻断
        logger.exception("清理线程 checkpoint 失败: %s", thread_id)
    return True


async def clear_thread_and_create_new(thread_id: str, user_id: str) -> Optional[str]:
    """删除旧线程并新建；越权返回 None（路由层转 OWNERSHIP_ERROR）。"""
    if not await delete_thread(thread_id, user_id):
        return None
    new_tid = new_thread_id()
    await get_threads_store().create(new_tid, user_id)
    return new_tid


async def fetch_sessions_list(user_id: str) -> List[Dict[str, Any]]:
    """当前用户的会话列表（注册表直查，updated_at desc，无需 N+1 拉 state）。"""
    return await get_threads_store().list_for_user(user_id, SESSION_LIST_LIMIT)


__all__ = [
    "OWNERSHIP_ERROR",
    "ensure_thread",
    "run_reply",
    "stream_reply",
    "fetch_sessions_list",
    "fetch_session_detail_state",
    "delete_thread",
    "clear_thread_and_create_new",
    "close_graph",
]
