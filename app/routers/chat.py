"""聊天路由：非流式 /api/chat 与流式 /api/chat/stream（SSE）。"""

import json
import logging
from typing import Any, Dict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app import graph_service
from app.deps import get_or_create_user_id
from app.validation import validate_image_data, validate_query_length

logger = logging.getLogger(__name__)

router = APIRouter()


async def _parse_payload(request: Request) -> Any:
    """解析 JSON 请求体；空 body 视为 {}，非法 JSON 返回 None（由调用方回 400）。"""
    body = await request.body()
    if not body:
        return {}
    try:
        return json.loads(body)
    except ValueError:
        return None


def _rate_limited(request: Request, user_id: str) -> bool:
    # 双窗口限流：用户维度（正常浏览器会话）+ IP 维度独立窗口。
    # 匿名 cookie 可被脚本丢弃（每次请求获得新 user_id），仅按 user_id 计数形同虚设；
    # IP 窗口依赖 uvicorn --proxy-headers 还原的真实客户端 IP（见 Dockerfile CMD）。
    # 两个窗口都判定（不短路），保证计数器行为一致。
    client = request.client.host if request.client else "unknown"
    user_ok = request.app.state.limiter.allow(f"chat:{user_id}:{client}")
    ip_ok = request.app.state.ip_limiter.allow(f"chat-ip:{client}")
    return not (user_ok and ip_ok)


@router.post("/api/chat")
async def chat(request: Request):
    """非流式聊天：跑完整个图后一次性返回最终回复。"""
    user_id = get_or_create_user_id(request)
    if _rate_limited(request, user_id):
        return JSONResponse({"error": "请求过于频繁，请稍后再试"}, status_code=429)

    data = await _parse_payload(request)
    if data is None:
        return JSONResponse({"error": "请求体格式错误"}, status_code=400)
    if not isinstance(data, dict):
        return JSONResponse({"error": "请求体格式错误"}, status_code=400)

    user_message = (data.get("message") or "").strip()
    err = validate_query_length(user_message)
    if err:
        return JSONResponse({"error": err}, status_code=400)
    if not user_message:
        return JSONResponse({"error": "消息不能为空"}, status_code=400)

    image_data, img_err = validate_image_data(data.get("images"))
    if img_err:
        return JSONResponse({"error": img_err}, status_code=400)

    try:
        thread_id = await graph_service.ensure_thread(data.get("session_id"), user_id)
        ai_text = await graph_service.run_reply(user_message, thread_id, image_data, user_id)
    except TimeoutError:
        return JSONResponse({"error": "运行超时"}, status_code=500)
    except Exception:
        logger.exception("聊天处理错误")
        return JSONResponse({"error": "服务器内部错误，请稍后重试"}, status_code=500)

    return {"response": ai_text, "session_id": thread_id, "thread_id": thread_id}


@router.post("/api/chat/stream")
async def chat_stream(request: Request):
    """流式聊天（SSE）：逐 token 推送，事件协议与前端既有实现一致。"""
    user_id = get_or_create_user_id(request)
    if _rate_limited(request, user_id):
        return JSONResponse({"error": "请求过于频繁，请稍后再试"}, status_code=429)

    data = await _parse_payload(request)
    if data is None or not isinstance(data, dict):
        return JSONResponse({"error": "请求体格式错误"}, status_code=400)

    user_message = (data.get("message") or "").strip()
    err = validate_query_length(user_message)
    if err:
        return JSONResponse({"error": err}, status_code=400)

    image_data, img_err = validate_image_data(data.get("images"))
    if img_err:
        return JSONResponse({"error": img_err}, status_code=400)

    if not user_message:
        # 保持 SSE 语义：错误也以事件流返回
        async def empty_msg():
            yield 'data: {"error": "消息不能为空"}\n\n'
            yield "data: [DONE]\n\n"
        return StreamingResponse(empty_msg(), media_type="text/event-stream")

    try:
        thread_id = await graph_service.ensure_thread(data.get("session_id"), user_id)
    except Exception:
        logger.exception("创建线程失败")
        return JSONResponse({"error": "服务器内部错误，请稍后重试"}, status_code=500)

    return StreamingResponse(
        graph_service.stream_reply(user_message, thread_id, image_data, user_id),
        media_type="text/event-stream",
        headers={
            # 反缓冲：nginx 端也配了 proxy_buffering off，这里双重保险
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )
