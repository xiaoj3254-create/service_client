FROM python:3.13-slim

ENV PYTHONUTF8=1 \
    PYTHONUNBUFFERED=1 \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 先装依赖再拷代码，充分利用 Docker 层缓存
COPY requirements.lock.txt .
RUN pip install --no-cache-dir -r requirements.lock.txt

COPY . .

# 非 root 运行；chroma_db / .flask_secret_key 由挂载卷或运行时生成
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
# --proxy-headers: 信任 nginx 的 X-Forwarded-For，还原真实客户端 IP（限流依赖）。
# compose 中 8000 仅对 nginx 容器暴露（expose，不发布宿主机端口），因此信任所有来源是安全的。
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--proxy-headers", "--forwarded-allow-ips", "*"]
