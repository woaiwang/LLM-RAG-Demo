"""
RAG 系统评估指标

检索指标:
  - Hit Rate@K:      相关文档是否出现在前 K 个结果中
  - MRR (Mean Reciprocal Rank):  第一个相关文档排名的倒数均值
  - Precision@K:     前 K 个结果中相关文档的比例

生成质量指标由 evaluator.py 中的 LLM-as-Judge 完成。
"""

from typing import List, Set


def hit_rate_at_k(
    retrieved_docs: List[str],
    relevant_docs: Set[str],
    k: int,
) -> float:
    """
    Hit Rate@K: 前 K 个检索结果中是否包含相关文档。

    返回: 1.0 (命中) 或 0.0 (未命中)
    """
    top_k = retrieved_docs[:k]
    return 1.0 if any(doc in relevant_docs for doc in top_k) else 0.0


def average_hit_rate_at_k(
    all_retrieved: List[List[str]],
    all_relevant: List[Set[str]],
    k: int,
) -> float:
    """所有测试样本的 Hit Rate@K 均值"""
    if not all_retrieved:
        return 0.0
    scores = [
        hit_rate_at_k(retrieved, relevant, k)
        for retrieved, relevant in zip(all_retrieved, all_relevant)
    ]
    return sum(scores) / len(scores)


def reciprocal_rank(retrieved_docs: List[str], relevant_docs: Set[str]) -> float:
    """单个查询的 Reciprocal Rank"""
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc in relevant_docs:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    all_retrieved: List[List[str]],
    all_relevant: List[Set[str]],
) -> float:
    """MRR: 首个相关文档排名的倒数，对所有查询取均值"""
    if not all_retrieved:
        return 0.0
    scores = [
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(all_retrieved, all_relevant)
    ]
    return sum(scores) / len(scores)


def precision_at_k(
    retrieved_docs: List[str],
    relevant_docs: Set[str],
    k: int,
) -> float:
    """Precision@K: 前 K 个结果中相关文档的比例"""
    if k == 0:
        return 0.0
    top_k = retrieved_docs[:k]
    hits = sum(1 for doc in top_k if doc in relevant_docs)
    return hits / k


def average_precision_at_k(
    all_retrieved: List[List[str]],
    all_relevant: List[Set[str]],
    k: int,
) -> float:
    """所有测试样本的 Precision@K 均值"""
    if not all_retrieved:
        return 0.0
    scores = [
        precision_at_k(retrieved, relevant, k)
        for retrieved, relevant in zip(all_retrieved, all_relevant)
    ]
    return sum(scores) / len(scores)
