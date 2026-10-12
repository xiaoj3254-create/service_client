"""FastAPI 应用配置（运行时/部署相关）；模型与 RAG 配置见根目录 config.py。"""

import os
import secrets
import time

from dotenv import load_dotenv

load_dotenv()

# --- 数据库（Postgres）---
# checkpointer 与线程注册表共用。生产（compose）必须设置；
# 未设置时退化为进程内存储（InMemorySaver + 内存线程表），仅供开发/测试单 worker 使用。
DATABASE_URL: str = os.getenv("DATABASE_URL", "")

# --- 会话 Cookie（匿名用户身份，user_id 隔离的根）---
_WEAK_SECRET_DEFAULTS = {"", "your-secret-key-here", "change-me", "secret"}


def _load_or_create_key_file(path: str) -> str:
    """
    读取密钥文件；不存在则以 O_CREAT|O_EXCL 原子创建。

    原子创建保证多 worker 并发首启时只有一个写入者，竞争失败的进程重读
    文件（短暂等待写入方完成），避免各 worker 持有不同签名密钥导致
    跨 worker 会话 cookie 拒签、用户身份漂移。
    """
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        for _ in range(20):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content:
                return content
            time.sleep(0.05)
        raise OSError(f"密钥文件已存在但始终为空: {path}")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(secrets.token_hex(32))
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_session_secret() -> str:
    """
    读取会话签名密钥；未设置或为已知弱默认值时生成随机密钥并持久化到本地文件
    （.gitignored），避免重启后所有匿名用户身份漂移、会话列表「全部消失」。
    生产环境必须显式设置 SESSION_SECRET_KEY。
    """
    _secret = os.getenv("SESSION_SECRET_KEY", "")
    if _secret not in _WEAK_SECRET_DEFAULTS:
        return _secret
    _key_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".flask_secret_key")
    try:
        _secret = _load_or_create_key_file(_key_file)
        import logging
        logging.getLogger(__name__).warning(
            "SESSION_SECRET_KEY 未设置，已使用本地持久化密钥（%s）；生产环境请设置强密钥环境变量", _key_file
        )
    except OSError:
        _secret = secrets.token_hex(32)
    return _secret


SESSION_COOKIE_SECURE: bool = os.getenv("SESSION_COOKIE_SECURE", "false").lower() == "true"
SESSION_SAME_SITE: str = os.getenv("SESSION_SAME_SITE", "lax")

# --- 请求限制 ---
# 单条用户消息最大长度（字符）
MAX_QUERY_CHARS: int = int(os.getenv("MAX_QUERY_CHARS", "2000"))
# 请求体总大小上限（MB）：需容纳最多 4 张 5MB 图片的 base64（约 27MB）+ JSON 余量
MAX_CONTENT_LENGTH_MB: int = int(os.getenv("MAX_CONTENT_LENGTH_MB", "32"))
# base64 解码后单张图片上限
MAX_IMAGE_BYTES: int = 5 * 1024 * 1024
MAX_IMAGE_COUNT: int = 4

# --- 限流（进程内滑动窗口；多 worker 时为每 worker 独立配额，单机部署误差可接受）---
RATE_LIMIT_CHAT_PER_MIN: int = int(os.getenv("RATE_LIMIT_CHAT_PER_MIN", "10"))
# IP 维度独立窗口：防无 Cookie 脚本每次请求换新 user_id 绕过用户窗口（NAT 多人共用 IP，放宽 3 倍）
RATE_LIMIT_IP_PER_MIN: int = int(os.getenv("RATE_LIMIT_IP_PER_MIN", "30"))

# 非流式单轮等待上限（秒）：图内 LLM 调用有自身超时，此为整体兜底
AINVOKE_TIMEOUT: int = int(os.getenv("AINVOKE_TIMEOUT", "120"))

# 会话列表一次返回的最大线程数
SESSION_LIST_LIMIT: int = int(os.getenv("SESSION_LIST_LIMIT", "30"))
