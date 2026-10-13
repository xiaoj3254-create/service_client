"""线程注册表存储：记录 thread 归属（user_id）与统计，替代原 REST threads/search。

- 生产：Postgres（psycopg 异步连接池），与 LangGraph checkpointer 同库同实例
- 开发/测试：进程内字典（无 DATABASE_URL 时启用，仅限单 worker）

统一接口（异步）：
    get(thread_id, user_id) -> Optional[dict]
    create(thread_id, user_id)
    list_for_user(user_id, limit) -> List[dict]   # updated_at desc
    update_stats(thread_id, message_count=..., last_user_question=..., search_text=...)
    delete(thread_id, user_id) -> bool
    initialize() / close()
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from app.config import DATABASE_URL

logger = logging.getLogger(__name__)

# 归属校验失败统一文案（不区分「不存在」与「无权」，避免泄露线程存在性）
OWNERSHIP_ERROR: str = '会话不存在或无权访问'

_DDL_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS app_threads (
        thread_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        message_count INTEGER NOT NULL DEFAULT 0,
        last_user_question TEXT NOT NULL DEFAULT '',
        search_text TEXT NOT NULL DEFAULT ''
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_app_threads_user_updated ON app_threads (user_id, updated_at DESC)",
)

_COLS = "thread_id, created_at, updated_at, message_count, last_user_question, search_text"


def _row_to_dict(row: Any) -> Dict[str, Any]:
    """(thread_id, created_at, updated_at, count, last_q, search_text) → 前端会话项。"""
    thread_id, created_at, updated_at, count, last_q, search_text = row
    return {
        "session_id": thread_id,
        "created_at": _to_epoch(created_at),
        "updated_at": _to_epoch(updated_at),
        "message_count": count,
        "last_user_question": last_q or "",
        "search_text": search_text or "",
    }


def _to_epoch(value: Any) -> float:
    """timestamptz/epoch 兼容转换；无法解析时回退当前时间。"""
    if isinstance(value, (int, float)):
        return float(value)
    if value is not None:
        try:
            return value.timestamp()
        except Exception:
            pass
    return time.time()


class MemoryThreadsStore:
    """进程内线程注册表（开发/测试用；进程重启即丢，仅单 worker 语义成立）。"""

    def __init__(self) -> None:
        self._rows: Dict[str, Dict[str, Any]] = {}

    async def initialize(self) -> None:
        pass

    async def get(self, thread_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        row = self._rows.get(thread_id)
        if row and row["user_id"] == user_id:
            return dict(row)
        return None

    async def create(self, thread_id: str, user_id: str) -> None:
        now = time.time()
        self._rows[thread_id] = {
            "session_id": thread_id,
            "user_id": user_id,
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
            "last_user_question": "",
            "search_text": "",
        }

    async def list_for_user(self, user_id: str, limit: int) -> List[Dict[str, Any]]:
        rows = [dict(r) for r in self._rows.values() if r["user_id"] == user_id]
        rows.sort(key=lambda r: r["updated_at"], reverse=True)
        return rows[:limit]

    async def update_stats(self, thread_id: str, *, message_count: int,
                           last_user_question: str, search_text: str) -> None:
        row = self._rows.get(thread_id)
        if row:
            row["message_count"] = message_count
            row["last_user_question"] = last_user_question
            row["search_text"] = search_text
            row["updated_at"] = time.time()

    async def delete(self, thread_id: str, user_id: str) -> bool:
        row = self._rows.get(thread_id)
        if row and row["user_id"] == user_id:
            del self._rows[thread_id]
            return True
        return False

    async def close(self) -> None:
        pass


class PgThreadsStore:
    """Postgres 线程注册表（psycopg 异步连接池，懒初始化）。"""

    def __init__(self, conninfo: str) -> None:
        self._conninfo = conninfo
        self._pool: Any = None

    async def initialize(self) -> None:
        await self._get_pool()
        async with self._pool.connection() as conn:
            for stmt in _DDL_STATEMENTS:
                await conn.execute(stmt)
        logger.info("threads 注册表 DDL 就绪")

    async def _get_pool(self) -> Any:
        if self._pool is None:
            from psycopg_pool import AsyncConnectionPool
            self._pool = AsyncConnectionPool(self._conninfo, min_size=1, max_size=10, open=False)
            await self._pool.open(wait_ready=True)
        return self._pool

    async def get(self, thread_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(
                f"SELECT {_COLS} FROM app_threads WHERE thread_id = %s AND user_id = %s",
                (thread_id, user_id),
            )
            row = await cur.fetchone()
        return _row_to_dict(row) if row else None

    async def create(self, thread_id: str, user_id: str) -> None:
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                "INSERT INTO app_threads (thread_id, user_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (thread_id, user_id),
            )

    async def list_for_user(self, user_id: str, limit: int) -> List[Dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(
                f"SELECT {_COLS} FROM app_threads WHERE user_id = %s ORDER BY updated_at DESC LIMIT %s",
                (user_id, limit),
            )
            rows = await cur.fetchall()
        return [_row_to_dict(r) for r in rows]

    async def update_stats(self, thread_id: str, *, message_count: int,
                           last_user_question: str, search_text: str) -> None:
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                """
                UPDATE app_threads
                SET message_count = %s, last_user_question = %s, search_text = %s, updated_at = now()
                WHERE thread_id = %s
                """,
                (message_count, last_user_question, search_text, thread_id),
            )

    async def delete(self, thread_id: str, user_id: str) -> bool:
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(
                "DELETE FROM app_threads WHERE thread_id = %s AND user_id = %s",
                (thread_id, user_id),
            )
            return cur.rowcount > 0

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None


_store: Any = None


def get_threads_store() -> Any:
    """进程级单例：有 DATABASE_URL 用 Postgres，否则进程内字典。"""
    global _store
    if _store is None:
        _store = PgThreadsStore(DATABASE_URL) if DATABASE_URL else MemoryThreadsStore()
    return _store


async def close_threads_store() -> None:
    global _store
    if _store is not None:
        await _store.close()
        _store = None


def new_thread_id() -> str:
    return uuid.uuid4().hex
