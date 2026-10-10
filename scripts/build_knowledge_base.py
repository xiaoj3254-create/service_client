#!/usr/bin/env python3
"""
构建 RAG 知识库：把 knowledge/<domain>/*.md 摄取到 Chroma 本地持久化库。

分块策略：按 markdown 二级标题（## ）切分；无二级标题的文件整篇作为一个文档块。
每块元数据：{"domain": <子目录名>, "source": <文件名>}，供检索时按智能体领域过滤。

幂等性：以 "文件路径#块序号" 作为文档 id 并使用 upsert，重跑自动覆盖旧内容；
加 --reset 先清空集合再摄取（域内容删除后建议用 --reset 同步清理）。

用法
----
    python scripts/build_knowledge_base.py            # 增量 upsert
    python scripts/build_knowledge_base.py --reset    # 清空后重建

环境变量
--------
    OPENAI_API_KEY / OPENAI_BASE_URL  embeddings 端点（必填 KEY）
    EMBEDDING_MODEL                   默认 BAAI/bge-m3
    RAG_PERSIST_DIR                   默认 ./chroma_db

退出码：0 成功；1 无可摄取内容；2 配置错误
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

# 保证脚本可从项目根直接运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("build_knowledge_base")

KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge")


def _chunk_markdown(text: str):
    """按 '## ' 标题切块；(chunk_id_part, content) 列表。无二级标题时整篇返回。"""
    sections = []
    current_title = None
    current_lines: list[str] = []

    def _flush():
        content = "\n".join(current_lines).strip()
        if content:
            sections.append((current_title or "", content))

    for line in text.splitlines():
        if line.startswith("## "):
            _flush()
            current_title = line[3:].strip()
            current_lines = [line]
        else:
            current_lines.append(line)
    _flush()

    if not sections:
        whole = text.strip()
        if whole:
            sections.append(("", whole))
    return sections


def collect_documents():
    """遍历 knowledge/<domain>/*.md，产出 (doc_id, content, metadata) 列表。"""
    docs = []
    if not os.path.isdir(KNOWLEDGE_DIR):
        logger.error("知识目录不存在: %s", KNOWLEDGE_DIR)
        return docs

    for domain in sorted(os.listdir(KNOWLEDGE_DIR)):
        domain_dir = os.path.join(KNOWLEDGE_DIR, domain)
        if not os.path.isdir(domain_dir):
            continue
        for filename in sorted(os.listdir(domain_dir)):
            if not filename.lower().endswith(".md"):
                continue
            path = os.path.join(domain_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            for i, (title, content) in enumerate(_chunk_markdown(text)):
                doc_id = f"{path}#{i}"
                title_part = f"【{title}】\n" if title else ""
                docs.append((
                    doc_id,
                    f"{title_part}{content}",
                    {"domain": domain, "source": filename, "section": title or ""},
                ))
    return docs


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 RAG 知识库（Chroma 本地持久化）")
    parser.add_argument("--reset", action="store_true", help="先清空集合再摄取")
    args = parser.parse_args()

    from config import OPENAI_API_KEY
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY 未设置，无法调用 embeddings 接口")
        return 2

    import chromadb
    from tools.rag_tools import RAG_PERSIST_DIR, RAG_COLLECTION_NAME, _embed

    docs = collect_documents()
    if not docs:
        logger.error("未在 knowledge/ 下找到任何 markdown 内容")
        return 1

    client = chromadb.PersistentClient(path=RAG_PERSIST_DIR)
    if args.reset:
        try:
            client.delete_collection(RAG_COLLECTION_NAME)
            logger.info("已清空集合 %s", RAG_COLLECTION_NAME)
        except Exception:
            pass  # 集合不存在时忽略

    collection = client.get_or_create_collection(
        name=RAG_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [d[0] for d in docs]
    contents = [d[1] for d in docs]
    metadatas = [d[2] for d in docs]

    logger.info("共 %d 个文档块，正在计算 embeddings（模型见 EMBEDDING_MODEL 环境变量）…", len(contents))
    try:
        embeddings = _embed(contents)
    except Exception as e:
        logger.error("embeddings 调用失败: %s", e)
        return 2

    # 分批 upsert（chroma 单批过大可能超限）
    batch = 32
    for i in range(0, len(ids), batch):
        collection.upsert(
            ids=ids[i:i + batch],
            documents=contents[i:i + batch],
            metadatas=metadatas[i:i + batch],
            embeddings=embeddings[i:i + batch],
        )

    logger.info(
        "摄取完成：%d 块 → %s（集合 %s，总数 %d）",
        len(ids), RAG_PERSIST_DIR, RAG_COLLECTION_NAME, collection.count(),
    )

    by_domain: dict[str, int] = {}
    for m in metadatas:
        by_domain[m["domain"]] = by_domain.get(m["domain"], 0) + 1
    for domain, count in sorted(by_domain.items()):
        logger.info("  %s: %d 块", domain, count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
