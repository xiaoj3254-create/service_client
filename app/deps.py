"""请求级依赖：匿名用户身份（Flask session cookie 语义平移到 Starlette Session）。"""

import uuid

from starlette.requests import Request


def get_or_create_user_id(request: Request) -> str:
    """
    获取当前浏览器访客的用户标识（匿名身份，持久化在签名 session cookie 中）。

    所有线程创建时都会写入 app_threads.user_id，会话列表 / 详情 / 删除
    均按该标识做归属校验，保证不同访客之间的会话相互隔离、互不可见。
    """
    user_id = request.session.get("user_id")
    if not user_id:
        user_id = uuid.uuid4().hex
        request.session["user_id"] = user_id
    return user_id
