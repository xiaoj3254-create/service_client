#!/usr/bin/env python3
"""
一键清理 LangGraph 中阻塞执行队列的僵尸 / 积压运行（run）。

背景
----
`langgraph dev` 的 worker 并发度默认只有 1。若进程被强杀（taskkill /F、kill -9），
当时正在运行的 run 会被持久化，重启后恢复为 running 状态却永远不会结束 ——
它占住唯一的 worker，导致此后所有新对话只能排队，前端表现为
「一直在响应但没有任何输出」（连 HTTP 响应头都收不到）。

用法
----
    python scripts/clean_stale_runs.py                  # 只列出，不做任何改动（默认安全）
    python scripts/clean_stale_runs.py --cancel         # 列出并在二次确认后取消
    python scripts/clean_stale_runs.py --cancel --yes   # 跳过确认直接取消

环境变量
--------
    LANGGRAPH_API_URL   默认 http://127.0.0.1:2024

退出码
------
    0 成功 / 无需处理；1 有取消失败；2 无法连接 LangGraph
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

API = os.getenv("LANGGRAPH_API_URL", "http://127.0.0.1:2024").rstrip("/")

# 只有这两种状态会真正占住 / 排队等待 worker。
# 注意不要包含 interrupted：那是已停止执行的中断状态，不占用 worker，
# 误清理会破坏 human-in-the-loop 流程。
STALE_STATUSES = ("running", "pending")

# 「疑似僵尸」阈值（秒）：正常一轮对话不会跑这么久。超过即高亮提示。
SUSPECT_SECONDS = int(os.getenv("LANGGRAPH_STALE_SUSPECT_SECONDS", "300"))


def _age_seconds(created_at: Any) -> Optional[float]:
    """返回 created_at 距今的秒数；无法解析时返回 None。"""
    if not created_at:
        return None
    try:
        dt = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds()
    except Exception:
        return None


def _format_age(age: Optional[float], status: str) -> str:
    if age is None:
        return "时长未知"
    verb = "已等待" if status == "pending" else "已持续"
    if age < 60:
        text = f"{verb} {age:.0f} 秒"
    elif age < 3600:
        text = f"{verb} {age / 60:.1f} 分钟"
    else:
        text = f"{verb} {age / 3600:.1f} 小时"
    if status == "running" and age >= SUSPECT_SECONDS:
        text += "  ⚠️ 疑似僵尸（长时间未结束，很可能已占死 worker）"
    return text


def fetch_stale_runs() -> Tuple[int, List[Dict[str, Any]]]:
    """返回 (线程总数, 处于 running/pending 的 run 列表)。"""
    threads = requests.post(f"{API}/threads/search", json={}, timeout=15).json()
    stale: List[Dict[str, Any]] = []

    for t in threads or []:
        tid = t.get("thread_id")
        if not tid:
            continue
        try:
            runs = requests.get(f"{API}/threads/{tid}/runs", timeout=15).json()
        except Exception as e:
            print(f"  ! 读取线程 {tid} 的 run 列表失败：{type(e).__name__}: {e}")
            continue
        for r in runs or []:
            if r.get("status") in STALE_STATUSES:
                stale.append({
                    "thread_id": tid,
                    "run_id": r.get("run_id"),
                    "status": r.get("status"),
                    "created_at": r.get("created_at"),
                    "age": _age_seconds(r.get("created_at")),
                })
    return len(threads or []), stale


def main() -> int:
    parser = argparse.ArgumentParser(
        description="清理阻塞 LangGraph 执行队列的僵尸 / 积压 run",
    )
    parser.add_argument("--cancel", action="store_true",
                        help="真正执行取消（默认只列出，不做任何改动）")
    parser.add_argument("--yes", action="store_true", help="跳过二次确认")
    args = parser.parse_args()

    print(f"LangGraph API: {API}\n")

    try:
        total_threads, stale = fetch_stale_runs()
    except Exception as e:
        print(f"❌ 无法连接 LangGraph（{API}）：{type(e).__name__}: {e}")
        print("   请确认 langgraph dev 已启动，或设置 LANGGRAPH_API_URL。")
        return 2

    print(f"线程总数: {total_threads}")

    if not stale:
        print("未发现 running / pending 的 run，执行队列健康 ✅")
        return 0

    print(f"\n⚠️  发现 {len(stale)} 个可能阻塞队列的 run：")
    for item in stale:
        print(f"  {item['status']:8s} {_format_age(item.get('age'), item['status'])}")
        print(f"           thread={item['thread_id']}")
        print(f"           run   ={item['run_id']}")

    if not args.cancel:
        print("\n当前为「只列出」模式，未做任何改动。")
        print("确认要取消这些 run 时，请加 --cancel（会二次确认）；跳过确认用 --cancel --yes。")
        return 0

    if not args.yes:
        try:
            answer = input(f"\n确认取消以上 {len(stale)} 个 run？(y/N) ").strip().lower()
        except EOFError:
            print("当前环境无法交互确认，请改用 --cancel --yes。未做任何改动。")
            return 0
        if answer not in ("y", "yes"):
            print("已放弃，未做任何改动。")
            return 0

    print()
    ok = fail = 0
    for item in stale:
        url = f"{API}/threads/{item['thread_id']}/runs/{item['run_id']}/cancel"
        try:
            resp = requests.post(url, timeout=30)
            if 200 <= resp.status_code < 300:
                ok += 1
                print(f"  ✅ 已取消 {item['status']:9s} {item['run_id']} (HTTP {resp.status_code})")
            else:
                fail += 1
                print(f"  ❌ 取消失败 {item['run_id']} (HTTP {resp.status_code}) {resp.text[:100]}")
        except Exception as e:
            fail += 1
            print(f"  ❌ 取消失败 {item['run_id']}：{type(e).__name__}: {e}")

    print(f"\n完成：成功 {ok}，失败 {fail}")

    try:
        _, left = fetch_stale_runs()
        print("复查后剩余 running / pending：", len(left) if left else "0 ✅")
    except Exception:
        pass

    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
