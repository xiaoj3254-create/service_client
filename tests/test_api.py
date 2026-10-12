"""FastAPI 应用层测试：健康检查、输入校验、归属 403、SSE 事件序列、限流。

图运行通过 monkeypatch 隔离（不依赖 LLM 网络 / Postgres）。
"""
import asyncio
import base64
import json

import pytest
from fastapi.testclient import TestClient

from app import config as app_config
from app import graph_service
from app.main import create_app


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


VALID_PNG = f"data:image/png;base64,{_b64(b'\\x89PNG\r\n\\x1a\\n')}"


@pytest.fixture()
def client(monkeypatch):
    """无 DB 的应用实例 + 隔离图调用。"""
    app = create_app()
    with TestClient(app) as c:
        yield c


def _fake_run_reply(monkeypatch, text="模拟回复"):
    seen = {}

    async def fake(user_message, thread_id, images, user_id):
        seen["args"] = (user_message, thread_id, images, user_id)
        return text

    monkeypatch.setattr(graph_service, "run_reply", fake)
    return seen


def _fake_stream_reply(monkeypatch, events=None):
    if events is None:
        events = [
            {"type": "status", "content": "正在接入客服…"},
            {"type": "token", "content": "你"},
            {"type": "token", "content": "好"},
            {"type": "done", "content": "你好", "session_id": "t", "thread_id": "t",
             "agent": "综合客服", "query_type": "general_inquiry"},
        ]

    async def fake(user_message, thread_id, images, user_id):
        for evt in events:
            yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(graph_service, "stream_reply", fake)
    return events


class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("healthy", "degraded")
        assert "timestamp" in body


class TestChatValidation:
    def test_empty_message_rejected(self, client):
        resp = client.post("/api/chat", json={"message": "  "})
        assert resp.status_code == 400

    def test_overlong_message_rejected(self, client):
        resp = client.post("/api/chat", json={"message": "长" * (app_config.MAX_QUERY_CHARS + 1)})
        assert resp.status_code == 400
        assert "字以内" in resp.json()["error"]

    def test_invalid_image_rejected(self, client, monkeypatch):
        _fake_run_reply(monkeypatch)
        resp = client.post("/api/chat", json={"message": "你好", "images": ["http://not-data-url"]})
        assert resp.status_code == 400

    def test_too_many_images_rejected(self, client, monkeypatch):
        _fake_run_reply(monkeypatch)
        resp = client.post(
            "/api/chat",
            json={"message": "你好", "images": [VALID_PNG] * (app_config.MAX_IMAGE_COUNT + 1)},
        )
        assert resp.status_code == 400
        assert "数量" in resp.json()["error"]

    def test_invalid_json_rejected(self, client):
        resp = client.post(
            "/api/chat",
            content=b"{invalid",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400

    def test_happy_path_returns_thread_id(self, client, monkeypatch):
        seen = _fake_run_reply(monkeypatch)
        resp = client.post("/api/chat", json={"message": "你好", "session_id": "default"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["response"] == "模拟回复"
        # thread_id 响应与下游收到的一致（会话隔离：ensure_thread 真实执行）
        assert body["thread_id"] == seen["args"][1]
        assert body["session_id"] == body["thread_id"]

    def test_foreign_session_not_reused(self, client, monkeypatch):
        """他人/不存在的 session_id 必须新建线程（隔离语义）。"""
        seen = _fake_run_reply(monkeypatch)
        resp = client.post("/api/chat", json={"message": "你好", "session_id": "not-a-real-thread"})
        assert resp.status_code == 200
        assert resp.json()["thread_id"] != "not-a-real-thread"


class TestChatStream:
    def test_sse_event_sequence(self, client, monkeypatch):
        events = _fake_stream_reply(monkeypatch)
        resp = client.post("/api/chat/stream", json={"message": "你好"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert resp.headers.get("x-accel-buffering") == "no"

        lines = [ln for ln in resp.text.split("\n") if ln.startswith("data: ")]
        assert lines[-1] == "data: [DONE]"
        parsed = [json.loads(ln[len("data: "):]) for ln in lines[:-1]]
        assert parsed[0]["type"] == "status"
        tokens = [p for p in parsed if p["type"] == "token"]
        assert "".join(t["content"] for t in tokens) == "你好"
        assert parsed[-1]["type"] == "done"

    def test_stream_error_event(self, client, monkeypatch):
        _fake_stream_reply(monkeypatch, events=[{"error": "客服处理超时或服务繁忙，请稍后重试"}])
        resp = client.post("/api/chat/stream", json={"message": "你好"})
        lines = [ln for ln in resp.text.split("\n") if ln.startswith("data: ")]
        assert lines[-1] == "data: [DONE]"
        parsed = [json.loads(ln[len("data: "):]) for ln in lines[:-1]]
        assert "error" in parsed[0]


class TestSessions:
    def test_detail_unknown_session_403(self, client):
        resp = client.get("/api/sessions/does-not-exist")
        assert resp.status_code == 403
        assert resp.json()["error"] == "会话不存在或无权访问"

    def test_delete_unknown_session_403(self, client):
        resp = client.delete("/api/sessions/does-not-exist")
        assert resp.status_code == 403

    def test_clear_unknown_session_403(self, client):
        resp = client.post("/api/sessions/does-not-exist/clear")
        assert resp.status_code == 403

    def test_empty_session_list(self, client):
        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        assert resp.json()["sessions"] == []


class TestRateLimit:
    def test_chat_rate_limited(self, client, monkeypatch):
        from app.ratelimit import SlidingWindowLimiter

        client.app.state.limiter = SlidingWindowLimiter(2, 60.0)
        _fake_run_reply(monkeypatch)
        assert client.post("/api/chat", json={"message": "1"}).status_code == 200
        assert client.post("/api/chat", json={"message": "2"}).status_code == 200
        third = client.post("/api/chat", json={"message": "3"})
        assert third.status_code == 429

    def test_ip_window_blocks_cookieless_flood(self, client, monkeypatch):
        """丢弃 cookie（每次请求都是新 user_id）的脚本必须被 IP 维度窗口拦下。"""
        from app.ratelimit import SlidingWindowLimiter

        client.app.state.limiter = SlidingWindowLimiter(1000, 60.0)  # 用户窗口实质放开
        client.app.state.ip_limiter = SlidingWindowLimiter(2, 60.0)
        _fake_run_reply(monkeypatch)
        # 每请求清掉 cookie，模拟无 Cookie 客户端
        assert client.post("/api/chat", json={"message": "1"}).status_code == 200
        client.cookies.clear()
        assert client.post("/api/chat", json={"message": "2"}).status_code == 200
        client.cookies.clear()
        assert client.post("/api/chat", json={"message": "3"}).status_code == 429


class TestRequestSize:
    def test_oversize_body_413(self, client):
        # 阈值动态挂在 app.state；置 0 后任何带 content-length 的请求都超限
        client.app.state.max_body_bytes = 0
        resp = client.post("/api/chat", json={"message": "x"})
        assert resp.status_code == 413
        assert "请求体过大" in resp.json()["error"]

    def test_oversize_chunked_body_413(self, client):
        """无 Content-Length 的 chunked 请求也必须被拦（按实际接收字节计数）。"""
        client.app.state.max_body_bytes = 0

        def gen():
            yield b"x"

        resp = client.post(
            "/api/chat", content=gen(), headers={"Content-Type": "application/json"}
        )
        assert resp.status_code == 413
        assert "请求体过大" in resp.json()["error"]
