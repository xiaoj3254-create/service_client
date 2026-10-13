"""FastAPI 应用入口：uvicorn app.main:app --host 0.0.0.0 --port 8000"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from app import graph_service
from app.config import (
    MAX_CONTENT_LENGTH_MB,
    RATE_LIMIT_CHAT_PER_MIN,
    RATE_LIMIT_IP_PER_MIN,
    SESSION_CLEANUP_BATCH,
    SESSION_CLEANUP_INTERVAL_SECONDS,
    SESSION_COOKIE_SECURE,
    SESSION_RETENTION_DAYS,
    SESSION_SAME_SITE,
    load_session_secret,
)
from app.db import close_threads_store, get_threads_store
from app.ratelimit import SlidingWindowLimiter
from app.routers import chat, sessions

logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")))
logger = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(_BASE_DIR, "templates"))

_SIZE_LIMIT_ERROR = {"error": "请求体过大，请减少图片数量或压缩图片后重试"}


class _BodyTooLarge(Exception):
    """接收字节数超出上限的内部信号（在中间件层捕获并转 413）。"""


class BodySizeLimitMiddleware:
    """
    请求体大小守卫（纯 ASGI）：按实际接收的字节数限制，覆盖无 Content-Length
    的 chunked 传输编码（仅检查请求头的旧守卫可被 chunked 绕过）。

    - Content-Length 快速路径：超限直接 413，不进入应用；
    - chunked 路径：接收过程中累计超限即中断请求并回 413；
    - 阈值动态读取 app.state.max_body_bytes（测试可覆盖）。
    """

    def __init__(self, app, max_body_bytes: int):
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        limit = getattr(getattr(scope.get("app"), "state", None), "max_body_bytes", None)
        if limit is None:
            limit = self.max_body_bytes

        content_length = next(
            (v for k, v in scope.get("headers", []) if k == b"content-length"), b""
        )
        if content_length.isdigit() and int(content_length) > limit:
            await self._send_413(scope, receive, send)
            return

        total = 0
        response_started = False

        async def sized_receive():
            nonlocal total
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > limit:
                    raise _BodyTooLarge
            return message

        async def tracking_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, sized_receive, tracking_send)
        except _BodyTooLarge:
            # 应用尚未开始响应才可回 413；理论仅在读取 body 阶段触发
            if not response_started:
                await self._send_413(scope, receive, send)

    async def _send_413(self, scope, receive, send):
        response = JSONResponse(_SIZE_LIMIT_ERROR, status_code=413)
        await response(scope, receive, send)


async def _retention_loop() -> None:
    """后台定期清理过期会话（保留策略见 SESSION_RETENTION_DAYS）。

    每 worker 独立运行；删除按 thread_id 幂等，多 worker 并发重复执行无害。
    单轮失败仅记日志，下个周期自动重试，任务本身永不退出。
    """
    while True:
        try:
            deleted = await graph_service.cleanup_expired_threads(
                SESSION_RETENTION_DAYS * 86400, SESSION_CLEANUP_BATCH
            )
            if deleted:
                logger.info("保留策略清理了 %d 个过期会话", deleted)
        except Exception:
            logger.exception("过期会话定期清理失败")
        await asyncio.sleep(SESSION_CLEANUP_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(application: FastAPI):
    # 启动：初始化线程存储（Postgres 模式下执行 DDL）
    store = get_threads_store()
    await store.initialize()
    # 保留策略：SESSION_RETENTION_DAYS=0 关闭
    cleanup_task = None
    if SESSION_RETENTION_DAYS > 0:
        cleanup_task = asyncio.create_task(_retention_loop())
        logger.info(
            "会话保留策略已启用：%d 天，每 %ds 清理一批（上限 %d/批）",
            SESSION_RETENTION_DAYS, SESSION_CLEANUP_INTERVAL_SECONDS, SESSION_CLEANUP_BATCH,
        )
    yield
    # 关停：先停后台任务，再释放连接池与 checkpointer
    if cleanup_task is not None:
        cleanup_task.cancel()
    await close_threads_store()
    await graph_service.close_graph()


def create_app() -> FastAPI:
    application = FastAPI(title="多智能体客服系统", lifespan=lifespan)

    # 会话 Cookie（匿名身份）；SessionMiddleware 会自动加 HttpOnly
    application.add_middleware(
        SessionMiddleware,
        secret_key=load_session_secret(),
        same_site=SESSION_SAME_SITE,
        https_only=SESSION_COOKIE_SECURE,
    )

    # 请求体大小守卫（nginx 侧另有 client_max_body_size 双重保险）；
    # 阈值挂 app.state，测试可动态调整。注册在 SessionMiddleware 之后 = 外层，
    # 超限请求不进入会话签名逻辑。
    application.state.max_body_bytes = MAX_CONTENT_LENGTH_MB * 1024 * 1024
    application.add_middleware(
        BodySizeLimitMiddleware, max_body_bytes=MAX_CONTENT_LENGTH_MB * 1024 * 1024
    )

    # 限流器（单机进程内；多 worker 每进程独立计数）
    application.state.limiter = SlidingWindowLimiter(RATE_LIMIT_CHAT_PER_MIN, 60.0)
    application.state.ip_limiter = SlidingWindowLimiter(RATE_LIMIT_IP_PER_MIN, 60.0)

    # 全局异常兜底：对外统一通用文案，详细堆栈只进日志
    @application.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception):
        logger.exception("未处理异常: %s %s", request.method, request.url.path)
        return JSONResponse({"error": "服务器内部错误，请稍后重试"}, status_code=500)

    # 路由
    application.include_router(chat.router)
    application.include_router(sessions.router)

    @application.get("/api/health")
    async def health():
        """健康检查（负载均衡探活；DB 连通性一并探测）。"""
        try:
            db_ok = await get_threads_store().ping()
        except Exception:
            logger.exception("健康检查探测数据库失败")
            db_ok = False
        return {
            "status": "healthy" if db_ok else "degraded",
            "timestamp": __import__("time").time(),
        }

    @application.get("/api/test")
    async def test_langgraph():
        """诊断端点（原连通性测试）；进程内架构下已无意义，仅 DEBUG 开放占位。"""
        if os.getenv("LOG_LEVEL", "").upper() != "DEBUG":
            return JSONResponse({"error": "未开放"}, status_code=404)
        return {"status": "ok", "mode": "in-process"}

    # 静态资源（含本地化的 marked/purify）与主页
    application.mount(
        "/static",
        StaticFiles(directory=os.path.join(_BASE_DIR, "static")),
        name="static",
    )

    @application.get("/")
    async def index(request: Request):
        return templates.TemplateResponse(request=request, name="index.html")

    return application


app = create_app()
