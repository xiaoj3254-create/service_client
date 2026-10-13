"""会话保留策略：MemoryThreadsStore 过期批量删除 + cleanup_expired_threads 编排。"""
import asyncio
import time

from app import graph_service
from app.db import MemoryThreadsStore

_DAY = 86400


def _seed_store() -> MemoryThreadsStore:
    store = MemoryThreadsStore()
    asyncio.run(store.create("t_old", "u1"))
    asyncio.run(store.create("t_new", "u1"))
    asyncio.run(store.create("t_other_user", "u2"))
    # 拨旧 t_old 的 updated_at 到 100 天前（create 内部直接写内存行）
    store._rows["t_old"]["updated_at"] = time.time() - 100 * _DAY
    return store


def test_delete_older_than_removes_only_expired():
    store = _seed_store()
    deleted = asyncio.run(store.delete_older_than(90 * _DAY, 200))

    assert [d["thread_id"] for d in deleted] == ["t_old"]
    assert deleted[0]["user_id"] == "u1"
    # 过期行已删，未过期行与其他用户行不受影响
    assert asyncio.run(store.get("t_old", "u1")) is None
    assert asyncio.run(store.get("t_new", "u1")) is not None
    assert asyncio.run(store.get("t_other_user", "u2")) is not None


def test_delete_older_than_respects_limit_and_idempotent():
    store = _seed_store()
    store._rows["t_other_user"]["updated_at"] = time.time() - 200 * _DAY

    # limit=1：只删最旧的一个（t_other_user，200 天）
    first = asyncio.run(store.delete_older_than(90 * _DAY, 1))
    assert [d["thread_id"] for d in first] == ["t_other_user"]

    # 幂等：重复执行删除剩余过期行，已删行不会重复出现
    second = asyncio.run(store.delete_older_than(90 * _DAY, 200))
    assert [d["thread_id"] for d in second] == ["t_old"]
    assert asyncio.run(store.delete_older_than(90 * _DAY, 200)) == []


def test_cleanup_expired_threads_deletes_registry_and_checkpoint(monkeypatch):
    store = _seed_store()
    deleted_checkpoints = []

    class _FakeCheckpointer:
        async def adelete_thread(self, thread_id):
            deleted_checkpoints.append(thread_id)

    class _FakeGraph:
        checkpointer = _FakeCheckpointer()

    monkeypatch.setattr(graph_service, "get_threads_store", lambda: store)

    async def _fake_graph():
        return _FakeGraph()

    monkeypatch.setattr(graph_service, "get_graph", _fake_graph)

    count = asyncio.run(graph_service.cleanup_expired_threads(90 * _DAY, 200))

    assert count == 1
    assert deleted_checkpoints == ["t_old"]
    # 注册表行已删；活跃线程保留
    assert asyncio.run(store.get("t_old", "u1")) is None
    assert asyncio.run(store.get("t_new", "u1")) is not None


def test_cleanup_checkpoint_failure_does_not_block_batch(monkeypatch):
    store = _seed_store()
    store._rows["t_new"]["updated_at"] = time.time() - 95 * _DAY

    class _FlakyCheckpointer:
        async def adelete_thread(self, thread_id):
            if thread_id == "t_old":
                raise RuntimeError("checkpoint 清理失败")

    class _FakeGraph:
        checkpointer = _FlakyCheckpointer()

    monkeypatch.setattr(graph_service, "get_threads_store", lambda: store)

    async def _fake_graph():
        return _FakeGraph()

    monkeypatch.setattr(graph_service, "get_graph", _fake_graph)

    # t_old checkpoint 删除失败不阻断批次，两个注册表行仍都被删除
    count = asyncio.run(graph_service.cleanup_expired_threads(90 * _DAY, 200))
    assert count == 2
    assert asyncio.run(store.get("t_old", "u1")) is None
    assert asyncio.run(store.get("t_new", "u1")) is None
