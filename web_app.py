#!/usr/bin/env python3
"""
多智能体客服系统 - Web 入口
基于 LangGraph API 接口，路由与 Flask 会话；业务逻辑见 chat_web_service.py
"""

import os
import base64
import binascii
import logging
import secrets
import time
from typing import Dict, Any, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

from flask import Flask, render_template, request, jsonify, session, Response

from chat_web_service import (
    run_chat_sync,
    stream_chat_events,
    fetch_sessions_list,
    fetch_session_detail,
    delete_remote_thread,
    clear_thread_and_create_new,
    langgraph_connectivity_test,
)

# 导入配置（与历史行为保持一致）
from config import *  # noqa: E402,F401,F403

logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")))
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Flask 配置
# 生产环境必须设置强 FLASK_SECRET_KEY 环境变量；
# 未设置或使用已知弱默认值时，生成随机密钥（仅开发可用，重启后 session 失效）。
_WEAK_SECRET_DEFAULTS = {"", "your-secret-key-here", "change-me", "secret"}
_flask_secret = os.getenv("FLASK_SECRET_KEY", "")
if _flask_secret in _WEAK_SECRET_DEFAULTS:
    logger.warning("FLASK_SECRET_KEY 未设置或为弱默认值，已生成随机密钥（仅开发环境可用，重启后 session 失效）")
    _flask_secret = secrets.token_hex(32)
app.secret_key = _flask_secret
app.config['SESSION_TYPE'] = 'filesystem'

# 客户图片限制：base64 解码后单张最大 5MB，最多 4 张
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_COUNT = 4


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


# --- Flask session 内的本地对话占位（主页模板可能使用）---

def get_conversation_history(session_id: str) -> List[Dict[str, Any]]:
    if 'conversations' not in session:
        session['conversations'] = {}
    return session['conversations'].get(session_id, [])


def add_conversation_message(session_id: str, role: str, content: str) -> None:
    history = get_conversation_history(session_id)
    history.append({
        'role': role,
        'content': content,
    })
    session['conversations'][session_id] = history


@app.route('/')
def index():
    """主页"""
    current_session_id = session.get('current_session_id', 'default')
    conversation_history = get_conversation_history(current_session_id)
    return render_template('index.html', conversation_history=conversation_history)


@app.route('/api/chat', methods=['POST'])
def chat():
    """处理聊天请求"""
    try:
        data = request.get_json()
        user_message = (data.get('message') or '').strip()
        client_session_id = data.get('session_id', 'default')

        image_data, img_err = validate_image_data(data.get('images'))
        if img_err:
            return jsonify({'error': img_err}), 400

        ai_text, err_msg, http_code, thread_id = run_chat_sync(
            user_message, client_session_id, images=image_data
        )
        if err_msg:
            return jsonify({'error': err_msg}), http_code or 500

        return jsonify({
            'response': ai_text,
            'session_id': thread_id,
            'thread_id': thread_id,
        })
    except Exception as e:
        logger.exception("聊天处理错误")
        return jsonify({'error': '服务器内部错误，请稍后重试'}), 500


@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """处理流式聊天请求"""
    try:
        data = request.get_json()
        user_message = (data.get('message') or '').strip()
        client_session_id = data.get('session_id', 'default')

        image_data, img_err = validate_image_data(data.get('images'))
        if img_err:
            return jsonify({'error': img_err}), 400

        return Response(
            stream_chat_events(user_message, client_session_id, images=image_data),
            mimetype='text/event-stream'
        )

    except Exception as e:
        logger.exception("流式聊天处理错误")
        return jsonify({'error': '服务器内部错误，请稍后重试'}), 500


@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """获取会话列表"""
    sessions, err = fetch_sessions_list()
    if err:
        return jsonify({'error': err}), 500
    return jsonify({'sessions': sessions or []})


@app.route('/api/sessions/<session_id>', methods=['GET'])
def get_session(session_id):
    """获取特定会话详情"""
    session_data, err = fetch_session_detail(session_id)
    if err:
        return jsonify({'error': err}), 500
    return jsonify({'session': session_data})


@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """删除会话"""
    try:
        ok, status = delete_remote_thread(session_id)
        if ok:
            if 'conversations' in session and session_id in session['conversations']:
                del session['conversations'][session_id]
            return jsonify({'message': '会话删除成功'})
        return jsonify({'error': f'删除会话失败: {status}'}), 500
    except Exception as e:
        logger.exception("删除会话时出错")
        return jsonify({'error': '服务器内部错误，请稍后重试'}), 500


@app.route('/api/sessions/<session_id>/clear', methods=['POST'])
def clear_session(session_id):
    """清空会话"""
    try:
        new_thread_id, err = clear_thread_and_create_new(session_id)
        if err:
            return jsonify({'error': err}), 500

        if 'conversations' in session and session_id in session['conversations']:
            session['conversations'][session_id] = []

        return jsonify({
            'message': '会话清空成功',
            'new_thread_id': new_thread_id
        })
    except Exception as e:
        logger.exception("清空会话时出错")
        return jsonify({'error': '服务器内部错误，请稍后重试'}), 500


@app.route('/api/new_session', methods=['POST'])
def create_new_session():
    """创建新会话（Flask session 侧）"""
    try:
        import uuid
        new_session_id = str(uuid.uuid4())
        session['current_session_id'] = new_session_id
        if 'conversations' not in session:
            session['conversations'] = {}
        session['conversations'][new_session_id] = []
        return jsonify({
            'session_id': new_session_id,
            'message': '新会话创建成功'
        })
    except Exception as e:
        logger.exception("创建会话时出错")
        return jsonify({'error': '服务器内部错误，请稍后重试'}), 500


@app.route('/api/health')
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time()
    })


@app.route('/api/test')
def test_langgraph():
    """测试 LangGraph API 调用"""
    result, err = langgraph_connectivity_test()
    if err:
        return jsonify({'error': err}), 500
    return jsonify(result)


def main():
    """主函数"""
    logger.info("🚀 多智能体客服系统 Web 应用")
    logger.info("=" * 60)
    logger.info("🌐 启动 Web 服务...")
    logger.info("📱 访问地址: http://localhost:5000")
    logger.info("💡 按 Ctrl+C 停止服务")
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)


if __name__ == "__main__":
    main()
