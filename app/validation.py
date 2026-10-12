"""输入校验：消息长度、图片 data URL（原 web_app.py 校验逻辑迁移，测试引用本模块）。"""

import base64
import binascii
from typing import Any, List, Optional, Tuple

from app.config import MAX_IMAGE_BYTES, MAX_IMAGE_COUNT, MAX_QUERY_CHARS


def validate_query_length(user_message: str) -> Optional[str]:
    """校验用户消息长度；合法返回 None，超长返回错误信息。"""
    if len(user_message) > MAX_QUERY_CHARS:
        return f'消息过长，请控制在 {MAX_QUERY_CHARS} 字以内'
    return None


def _validate_single_image(image: str) -> Tuple[Optional[str], Optional[str]]:
    """校验单张图片 data URL；合法返回 (data_url, None)，非法返回 (None, 错误信息)。"""
    if not isinstance(image, str) or not image.startswith('data:image/'):
        return None, '图片格式无效，仅支持 data URL 图片'

    try:
        header, b64part = image.split(',', 1)
    except ValueError:
        return None, '图片数据格式错误'

    if 'base64' not in header:
        return None, '图片必须为 base64 编码'

    try:
        raw = base64.b64decode(b64part, validate=True)
    except (binascii.Error, ValueError):
        return None, '图片 base64 解码失败'

    if len(raw) == 0:
        return None, '图片内容为空'
    if len(raw) > MAX_IMAGE_BYTES:
        return None, f'图片过大，单张解码后不能超过 {MAX_IMAGE_BYTES // (1024 * 1024)}MB'

    return image, None


def validate_image_data(image: Any) -> Tuple[Optional[List[str]], Optional[str]]:
    """
    校验前端上传的图片 data URL 或图片列表。
    合法返回 (data_url_list, None)；无图片返回 (None, None)；非法返回 (None, 错误信息)。
    """
    if image is None or image == '':
        return None, None

    # 统一为列表处理
    if isinstance(image, str):
        images_raw = [image]
    elif isinstance(image, list):
        images_raw = image
    else:
        return None, '图片格式无效，仅支持 data URL 图片'

    if len(images_raw) == 0:
        return None, None
    if len(images_raw) > MAX_IMAGE_COUNT:
        return None, f'图片数量不能超过 {MAX_IMAGE_COUNT} 张'

    validated = []
    for img in images_raw:
        ok, err = _validate_single_image(img)
        if err:
            return None, err
        validated.append(ok)

    return validated, None
