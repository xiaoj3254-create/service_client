"""
RAG 知识检索
基于 Chroma（本地持久化）+ SiliconFlow OpenAI 兼容 embeddings 接口。

设计要点：
- chromadb 直接调用（已安装），不引入 langchain-chroma 额外依赖；
  embeddings 自行计算后传入，避免纠缠 chromadb 各版本的 EmbeddingFunction 接口差异。
- 全链路惰性初始化 + 失败静默降级：向量库缺失 / 未安装 / embeddings 调用失败
  时 retrieve() 返回空串，由 BaseAgent 回退到子类硬编码的 _match_data，
  保证未构建知识库时系统行为与改造前完全一致。
"""

import logging
import os
from typing import List, Optional

from config import OPENAI_API_KEY, OPENAI_BASE_URL

logger = logging.getLogger(__name__)

# Chroma 持久化目录与集合名
RAG_PERSIST_DIR: str = os.getenv("RAG_PERSIST_DIR", "./chroma_db")
RAG_COLLECTION_NAME: str = os.getenv("RAG_COLLECTION_NAME", "customer_service")

# embeddings 端点可独立于主 LLM 网关配置：
# 主网关（如 mimo）不一定提供 /embeddings，可单独指向 SiliconFlow 等。
EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL", OPENAI_BASE_URL)
EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY", OPENAI_API_KEY)

# 中文 embedding 模型（OpenAI 兼容 /embeddings 端点）
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# 检索返回的文档块数
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "3"))

_client = None
_collection = None
_init_error: Optional[str] = None


def _get_collection():
    """惰性获取 Chroma 集合；不可用时返回 None 并记住原因（避免每次检索都重试报错）。"""
    global _client, _collection, _init_error
    if _init_error:
        return None
    if _collection is not None:
        return _collection
    try:
        import chromadb
        _client = chromadb.PersistentClient(path=RAG_PERSIST_DIR)
        _collection = _client.get_or_create_collection(
            name=RAG_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("RAG 集合就绪: %s (%s)，当前文档数 %d", RAG_COLLECTION_NAME, RAG_PERSIST_DIR, _collection.count())
    except Exception as e:
        _init_error = str(e)
        logger.warning("RAG 初始化失败，检索将回退到硬编码知识: %s", e)
        return None
    return _collection


def _embed(texts: List[str]) -> List[List[float]]:
    """调用 SiliconFlow OpenAI 兼容 /embeddings 端点计算向量。"""
    import requests
    resp = requests.post(
        f"{EMBEDDING_BASE_URL}/embeddings",
        json={"model": EMBEDDING_MODEL, "input": texts},
        headers={"Authorization": f"Bearer {EMBEDDING_API_KEY}"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()["data"]
    # 按 index 排序，保证与输入顺序一致
    return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]


def retrieve(query: str, domain: str, k: int = RAG_TOP_K) -> str:
    """
    按领域检索知识片段。

    Args:
        query: 用户查询
        domain: 领域过滤（product / tech / billing / complaint / general）
        k: 返回文档块数

    Returns:
        拼接后的知识文本；不可用或无命中时返回空串（调用方回退硬编码匹配）。
    """
    if not query or not query.strip():
        return ""
    try:
        coll = _get_collection()
        if coll is None:
            return ""
        total = coll.count()
        if total == 0:
            logger.debug("RAG 集合为空（尚未构建知识库），回退硬编码匹配")
            return ""

        query_embedding = _embed([query.strip()])[0]
        result = coll.query(
            query_embeddings=[query_embedding],
            n_results=min(k, total),
            where={"domain": domain},
        )
        docs = (result.get("documents") or [[]])[0]
        if not docs:
            return ""
        logger.debug("RAG 命中 domain=%s 文档 %d 块", domain, len(docs))
        return "\n\n".join(docs)
    except Exception as e:
        # 检索异常不阻断对话：降级为硬编码匹配
        logger.warning("RAG 检索失败（domain=%s），回退硬编码匹配: %s", domain, e)
        return ""
