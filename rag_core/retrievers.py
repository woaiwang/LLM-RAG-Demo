"""
多策略检索器

支持三种检索策略:
  1. VectorRetriever    - 向量语义检索 (ChromaDB)
  2. BM25Retriever      - 关键词稀疏检索
  3. HybridRetriever    - RRF 融合 (向量 + BM25)

设计思路:
  通过统一的 Retriever 抽象类，新增策略只需继承并实现 retrieve()。
"""

import os
from abc import ABC, abstractmethod
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi
import jieba

from config import VECTOR_RETRIEVE_K, FINAL_RETRIEVE_K, RRF_K, RERANK_TOP_K
from rag_core.document_manager import DocumentManager, COLLECTION_NAME, CHROMA_PERSIST_DIR
from rag_core.reranker import Reranker


class Retriever(ABC):
    """检索器基类"""

    @abstractmethod
    def retrieve(self, query: str, k: int = FINAL_RETRIEVE_K) -> List[Document]:
        ...


# ======================== 1. 向量检索器 ========================

class VectorRetriever(Retriever):
    """基于 ChromaDB 的向量语义检索"""

    def __init__(self, doc_manager: DocumentManager):
        self.embeddings = doc_manager.embeddings
        self.vectorstore = self._init_vectorstore()

    def _init_vectorstore(self) -> Optional[Chroma]:
        if not os.path.exists(os.path.join(CHROMA_PERSIST_DIR, "chroma.sqlite3")):
            return None
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )

    def retrieve(self, query: str, k: int = FINAL_RETRIEVE_K) -> List[Document]:
        if self.vectorstore is None:
            return []
        return self.vectorstore.similarity_search(query, k=k)


# ======================== 2. BM25 检索器 ========================

class BM25Retriever(Retriever):
    """
    基于 BM25 的关键词稀疏检索

    对政策文档中的专有名词（如"推荐优秀应届本科毕业生免试攻读"）效果好，
    与向量检索互补。
    """

    def __init__(self, doc_manager: DocumentManager):
        self._build_index(doc_manager)

    def _build_index(self, doc_manager: DocumentManager):
        """从 DocumentManager 获取 chunks 并构建 BM25 索引"""
        chunks = doc_manager.get_all_chunks()
        self.chunks = chunks
        if not chunks:
            self.bm25 = None
            return
        # 中文分词后构建 BM25
        tokenized_corpus = [list(jieba.cut(c["text"])) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, k: int = FINAL_RETRIEVE_K) -> List[Document]:
        if self.bm25 is None or not self.chunks:
            return []

        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)
        # 返回 Top-K 的索引
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            doc = Document(
                page_content=chunk["text"],
                metadata=chunk.get("metadata", {}),
            )
            results.append(doc)
        return results


# ======================== 3. 混合检索器 (RRF) ========================

class HybridRetriever(Retriever):
    """
    混合检索: Vector + BM25 → Reciprocal Rank Fusion (RRF)

    RRF 公式: score(d) = Σ 1/(k + rank_i(d))
    - 不需要调权重
    - 对分数尺度不敏感
    - 能利用两种检索的优势互补
    """

    def __init__(self, doc_manager: DocumentManager):
        self.vector_retriever = VectorRetriever(doc_manager)
        self.bm25_retriever = BM25Retriever(doc_manager)

    def retrieve(self, query: str, k: int = FINAL_RETRIEVE_K) -> List[Document]:
        # 1. 从两个检索器中各取 Top-N
        vector_k = max(VECTOR_RETRIEVE_K, k)
        vector_results = self.vector_retriever.retrieve(query, k=vector_k)
        bm25_results = self.bm25_retriever.retrieve(query, k=vector_k)

        # 2. RRF 融合
        # 用 dict 去重: key 是 page_content 的前 50 个字符（近似去重）
        content_to_doc = {}
        fused_scores = {}

        for rank, doc in enumerate(vector_results):
            key = doc.page_content[:50]
            content_to_doc[key] = doc
            fused_scores[key] = fused_scores.get(key, 0) + 1.0 / (RRF_K + rank + 1)

        for rank, doc in enumerate(bm25_results):
            key = doc.page_content[:50]
            content_to_doc[key] = doc
            fused_scores[key] = fused_scores.get(key, 0) + 1.0 / (RRF_K + rank + 1)

        # 3. 按融合分数排序，取 Top-K
        sorted_keys = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)[:k]
        return [content_to_doc[key] for key in sorted_keys]


# ======================== 工厂函数 ========================

def create_retriever(strategy: str, doc_manager: DocumentManager) -> Retriever:
    """策略工厂: 根据名称创建检索器"""
    strategy_map = {
        "vector": VectorRetriever,
        "bm25": BM25Retriever,
        "hybrid": HybridRetriever,
    }
    cls = strategy_map.get(strategy)
    if cls is None:
        raise ValueError(f"未知检索策略: {strategy}，可选: {list(strategy_map.keys())}")
    return cls(doc_manager)


def retrieve_and_rerank(
    retriever: Retriever,
    reranker: Reranker,
    query: str,
    retrieve_k: int = 10,
    rerank_k: int = RERANK_TOP_K,
) -> List[Document]:
    """
    检索 → Reranker 重排序 流水线。

    先取 Top-N 候选文档，再用 Cross-Encoder 精排取 Top-K。

    Args:
        retriever: 检索器实例
        reranker: Reranker 实例
        query: 用户查询
        retrieve_k: 粗排阶段返回的文档数
        rerank_k: 精排后最终保留的文档数

    Returns:
        重排序后的 Top-K 文档
    """
    docs = retriever.retrieve(query, k=retrieve_k)
    if not docs:
        return []
    return reranker.rerank(query, docs, k=rerank_k)
