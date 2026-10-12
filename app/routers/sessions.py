"""会话路由：列表 / 详情 / 删除 / 清空（全部按 user_id 归属校验）。"""

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app import graph_service
from app.db import OWNERSHIP_ERROR
from app.deps import get_or_create_user_id

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/sessions")
async def get_sessions(request: Request):
    """获取会话列表（仅当前用户的会话）。"""
    user_id = get_or_create_user_id(request)
    try:
        sessions = await graph_service.fetch_sessions_list(user_id)
        return {"sessions": sessions or []}
    except Exception:
        logger.exception("获取会话列表时出错")
        return JSONResponse({"error": "服务器错误，请稍后重试"}, status_code=500)


@router.get("/api/sessions/{session_id}")
async def get_session(session_id: str, request: Request):
    """获取特定会话详情（仅当前用户名下的会话）。"""
    user_id = get_or_create_user_id(request)
    try:
        session_data = await graph_service.fetch_session_detail_state(session_id, user_id)
    except Exception:
        logger.exception("获取会话详情时出错")
        return JSONResponse({"error": "服务器错误，请稍后重试"}, status_code=500)
    if session_data is None:
        return JSONResponse({"error": OWNERSHIP_ERROR}, status_code=403)
    return {"session": session_data}


@router.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, request: Request):
    """删除会话（仅当前用户名下的会话）。"""
    user_id = get_or_create_user_id(request)
    try:
        ok = await graph_service.delete_thread(session_id, user_id)
    except Exception:
        logger.exception("删除会话时出错")
        return JSONResponse({"error": "服务器内部错误，请稍后重试"}, status_code=500)
    if not ok:
        return JSONResponse({"error": OWNERSHIP_ERROR}, status_code=403)
    return {"message": "会话删除成功"}


@router.post("/api/sessions/{session_id}/clear")
async def clear_session(session_id: str, request: Request):
    """清空会话：删除旧线程并新建（仅当前用户名下的会话）。"""
    user_id = get_or_create_user_id(request)
    try:
        new_thread_id = await graph_service.clear_thread_and_create_new(session_id, user_id)
    except Exception:
        logger.exception("清空会话时出错")
        return JSONResponse({"error": "服务器内部错误，请稍后重试"}, status_code=500)
    if new_thread_id is None:
        return JSONResponse({"error": OWNERSHIP_ERROR}, status_code=403)
    return {"message": "会话清空成功", "new_thread_id": new_thread_id}
