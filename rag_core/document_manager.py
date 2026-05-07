"""
多文档管理器 (Document Manager)

职责:
  - 文档上传/删除/列表管理
  - SQLite 记录文档元数据
  - ChromaDB 单 collection + metadata 过滤

设计思路:
  所有文档共享同一个 Chroma collection "knowledge_base"，
  每个 chunk 的 metadata 中包含 doc_id，方便按文档删除和过滤。
"""

import os
import uuid
import sqlite3
import shutil
from datetime import datetime
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

from config import (
    EMBEDDING_MODEL, EMBEDDING_DEVICE, EMBEDDING_NORMALIZE,
    CHROMA_PERSIST_DIR, PDFS_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
)
from rag_core.chunker import RecursiveChunker

# ================= SQLite 表结构 =================

DB_PATH = os.path.join(CHROMA_PERSIST_DIR, "documents.db")
COLLECTION_NAME = "knowledge_base"


def _get_db_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            upload_time TEXT NOT NULL,
            chunk_count INTEGER DEFAULT 0,
            status TEXT DEFAULT 'processing'
        )
    """)
    conn.commit()
    return conn


class DocumentManager:
    """管理多份 PDF 文档的生命周期和向量索引。"""

    def __init__(self):
        os.makedirs(PDFS_DIR, exist_ok=True)
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        self.embeddings = self._init_embeddings()
        self.chunker = RecursiveChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # ---------------------------------------------------------------
    # Embedding 模型（带缓存）
    # ---------------------------------------------------------------
    @staticmethod
    def _init_embeddings():
        return HuggingFaceBgeEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": EMBEDDING_NORMALIZE},
        )

    # ---------------------------------------------------------------
    # Chroma 操作
    # ---------------------------------------------------------------
    def _get_vectorstore(self) -> Chroma:
        """获取或创建共享的 Chroma collection。"""
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )

    def _collection_exists(self) -> bool:
        """检查 collection 是否已存在于磁盘。"""
        return os.path.exists(os.path.join(CHROMA_PERSIST_DIR, "chroma.sqlite3"))

    # ---------------------------------------------------------------
    # 文档上传
    # ---------------------------------------------------------------
    def upload_pdf(self, file_bytes: bytes, original_filename: str) -> str:
        """
        上传并索引一个 PDF 文档。

        返回: doc_id (UUID 字符串)
        流程: 保存文件 → 切片 → 写入 Chroma → 记录 SQLite
        """
        doc_id = str(uuid.uuid4())
        saved_name = f"{doc_id}_{original_filename}"
        save_path = os.path.join(PDFS_DIR, saved_name)

        # 1. 保存原始文件
        with open(save_path, "wb") as f:
            f.write(file_bytes)

        try:
            # 2. 加载并切片
            loader = PyPDFLoader(save_path)
            docs = loader.load()
            splits = self.chunker.chunk(docs)

            # 3. 往每个 chunk 的 metadata 注入 doc_id
            for split in splits:
                split.metadata["doc_id"] = doc_id
                split.metadata["filename"] = original_filename

            # 4. 写入 Chroma
            vectorstore = self._get_vectorstore()
            vectorstore.add_documents(documents=splits)
            vectorstore.persist()

            # 5. 记录元数据
            conn = _get_db_connection()
            conn.execute(
                """INSERT INTO documents (id, filename, original_filename, upload_time, chunk_count, status)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (doc_id, saved_name, original_filename, datetime.now().isoformat(), len(splits), "ready"),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            # 失败时清理文件
            if os.path.exists(save_path):
                os.remove(save_path)
            raise e

        return doc_id

    # ---------------------------------------------------------------
    # 文档删除
    # ---------------------------------------------------------------
    def delete_document(self, doc_id: str) -> None:
        """删除指定文档：清理 Chroma 中的 chunks 和 SQLite 记录和本地文件。"""
        conn = _get_db_connection()
        row = conn.execute("SELECT filename FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if row is None:
            conn.close()
            raise ValueError(f"文档 {doc_id} 不存在")

        saved_name = row["filename"]

        # 1. 从 Chroma 中删除该文档的所有 chunks
        if self._collection_exists():
            try:
                vectorstore = self._get_vectorstore()
                # 通过 metadata 过滤找到所有该文档的 chunk IDs
                results = vectorstore.get(where={"doc_id": doc_id})
                if results and results.get("ids"):
                    vectorstore.delete(ids=results["ids"])
                    vectorstore.persist()
            except Exception as e:
                print(f"警告: Chroma 删除失败: {e}")

        # 2. 删除本地文件
        file_path = os.path.join(PDFS_DIR, saved_name)
        if os.path.exists(file_path):
            os.remove(file_path)

        # 3. 删除 SQLite 记录
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
        conn.close()

    # ---------------------------------------------------------------
    # 文档列表与查询
    # ---------------------------------------------------------------
    def list_documents(self) -> List[dict]:
        """返回所有文档的元数据列表。"""
        conn = _get_db_connection()
        rows = conn.execute("SELECT * FROM documents ORDER BY upload_time DESC").fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_document(self, doc_id: str) -> Optional[dict]:
        """获取单个文档的元数据。"""
        conn = _get_db_connection()
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        conn.close()
        return dict(row) if row else None

    # ---------------------------------------------------------------
    # 获取所有文档的 Chunks（供检索器使用）
    # ---------------------------------------------------------------
    def get_all_chunks(self) -> List[dict]:
        """
        从 Chroma 中获取所有文档的 text + metadata。
        供 BM25 检索器构建关键词索引。
        """
        if not self._collection_exists():
            return []
        vectorstore = self._get_vectorstore()
        results = vectorstore.get()
        chunks = []
        if results and results.get("documents"):
            for i, doc in enumerate(results["documents"]):
                chunks.append({
                    "text": doc,
                    "metadata": results["metadatas"][i] if results.get("metadatas") else {},
                    "id": results["ids"][i] if results.get("ids") else f"chunk_{i}",
                })
        return chunks
