"""进程内滑动窗口限流（单机部署；多 worker 时每进程独立计数，误差可接受）。

如需跨进程精确限流，后续可换 Redis 后端，接口保持 allow(key) 不变。
"""

import threading
import time
from collections import deque
from typing import Deque, Dict


class SlidingWindowLimiter:
    """按 key 的滑动窗口计数限流器：window_seconds 内最多 max_events 次。"""

    def __init__(self, max_events: int, window_seconds: float):
        self._max_events = max_events
        self._window = window_seconds
        self._events: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()
        self._last_evict: float = 0.0

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            # 淘汰节流：每半个窗口最多全表扫描一次，摊销 O(1)
            if now - self._last_evict > self._window / 2:
                self._last_evict = now
                self._evict_stale(now)
            q = self._events.get(key)
            if q is None:
                q = deque()
                self._events[key] = q
            # 淘汰窗口外记录
            while q and now - q[0] > self._window:
                q.popleft()
            if len(q) >= self._max_events:
                return False
            q.append(now)
            return True

    def _evict_stale(self, now: float) -> None:
        """删除窗口外已无记录的 key，防止丢 Cookie 脚本每次换 user_id 造成字典无限增长。"""
        stale = [k for k, q in self._events.items() if not q or now - q[-1] > self._window]
        for k in stale:
            del self._events[k]
