"""
查询处理器 (Query Processor)

职责:
  - Query Rewriting:  将用户口语查询改写为官方术语表述
  - Multi-Query:      将复杂问题拆解为多个子问题分别检索
  - RRF Merge:        将多路检索结果融合去重

设计思路:
  通过 Generator 调用 LLM 完成改写/扩展，输出解析为查询列表后，
  分别检索并经过 RRF 融合得到最终结果。

  改写与扩展都可以独立开关，互不依赖。
"""

import re
from typing import List, Dict

from langchain_core.documents import Document

from config import FINAL_RETRIEVE_K, RRF_K, MULTI_QUERY_SEARCH_K
from rag_core.generator import Generator
from rag_core.retrievers import Retriever


def rrf_merge(
    query_results: Dict[str, List[Document]],
    k: int = FINAL_RETRIEVE_K,
) -> List[Document]:
    """
    多路检索结果的 RRF 融合。

    每路检索结果都有自己的排序，RRF 通过排名而非分数来融合，
    避免不同检索器分数尺度不一致的问题。

    Args:
        query_results: {查询文本: 该查询检索到的文档列表}
        k: 最终返回的文档数

    Returns:
        融合后的 Top-K 文档列表
    """
    content_to_doc = {}
    fused_scores: Dict[str, float] = {}

    for query, docs in query_results.items():
        for rank, doc in enumerate(docs):
            key = doc.page_content[:80]
            content_to_doc[key] = doc
            fused_scores[key] = fused_scores.get(key, 0) + 1.0 / (RRF_K + rank + 1)

    sorted_keys = sorted(fused_scores, key=fused_scores.__getitem__, reverse=True)[:k]
    return [content_to_doc[key] for key in sorted_keys]


# ========================= QueryProcessor =========================


class QueryProcessor:
    """
    查询处理器: 改写 + 多查询扩展。

    使用方式:
        qp = QueryProcessor(generator)
        queries = qp.rewrite("保研要啥条件")   # → [原始, 改写1, 改写2]
        docs = qp.retrieve_with_expansion(retriever, "奖学金怎么申请")
    """

    def __init__(self, generator: Generator):
        self.generator = generator

    # ---------------------------------------------------------------
    # Step A: 查询改写 (Rewriting)
    # ---------------------------------------------------------------

    def rewrite(self, query: str) -> List[str]:
        """
        将口语查询改写为官方术语表述。

        Returns:
            查询列表，第一条为原始查询，后续为改写版本。
            如果改写失败（LLM 返回异常），回退为仅含原始查询的列表。
        """
        try:
            system_prompt, user_prompt = self.generator.build_rewrite_prompt(query)
            result = self.generator.generate(system_prompt, user_prompt)
            rewritten = self._parse_query_list(result, query)
        except Exception as e:
            rewritten = []

        # 始终包含原始查询作为兜底
        all_queries = [query]
        for q in rewritten:
            cleaned = q.strip().strip('"').strip("'")
            if cleaned and cleaned not in all_queries:
                all_queries.append(cleaned)

        # 限制改写数量，避免延迟过高
        return all_queries[:4]

    # ---------------------------------------------------------------
    # Step B: 多查询扩展 (Expansion)
    # ---------------------------------------------------------------

    def expand(self, query: str) -> List[str]:
        """
        将复杂查询拆解为多个独立子查询。

        Returns:
            子查询列表。如果扩展失败，回退为仅含原始查询。
        """
        try:
            system_prompt, user_prompt = self.generator.build_expand_prompt(query)
            result = self.generator.generate(system_prompt, user_prompt)
            expanded = self._parse_query_list(result, query)
        except Exception as e:
            expanded = []

        all_queries = [query]
        for q in expanded:
            cleaned = q.strip().strip('"').strip("'")
            if cleaned and cleaned not in all_queries:
                all_queries.append(cleaned)

        return all_queries[:6]

    # ---------------------------------------------------------------
    # Step C: 增强检索 (Rewrite + Expand 二选一)
    # ---------------------------------------------------------------

    def retrieve_enhanced(
        self,
        retriever: Retriever,
        query: str,
        mode: str = "none",
        search_k: int = MULTI_QUERY_SEARCH_K,
        final_k: int = FINAL_RETRIEVE_K,
    ) -> List[Document]:
        """
        增强检索入口，支持三种模式。

        Args:
            retriever: 检索器实例
            query: 用户原始查询
            mode: "none" | "rewrite" | "expand"
            search_k: 每次检索返回的文档数
            final_k: 最终融合后的文档数

        Returns:
            检索到的文档列表
        """
        if mode == "none":
            return retriever.retrieve(query, k=final_k)

        # 生成多查询
        if mode == "rewrite":
            queries = self.rewrite(query)
        else:
            queries = self.expand(query)

        # 所有查询分别检索
        all_results = {}
        for q in queries:
            docs = retriever.retrieve(q, k=search_k)
            if docs:
                all_results[q] = docs

        if not all_results:
            return retriever.retrieve(query, k=final_k)

        return rrf_merge(all_results, k=final_k)

    # ---------------------------------------------------------------
    # 工具方法
    # ---------------------------------------------------------------

    @staticmethod
    def _parse_query_list(text: str, original: str) -> List[str]:
        """
        解析 LLM 输出的编号列表。
        支持格式:
          1. 查询文本
          2. 查询文本
        """
        queries = []
        # 匹配 "数字. 内容" 或 "数字、内容" 格式
        for line in text.split("\n"):
            line = line.strip()
            # 跳过空行和可能的说明文字
            if not line:
                continue
            match = re.match(r"^\d+[\.\、\s]\s*(.+)", line)
            if match:
                q = match.group(1).strip()
                if q and q != original:
                    queries.append(q)

        # 如果正则没匹配到，尝试直接按换行分割取非空行
        if not queries:
            for line in text.split("\n"):
                line = line.strip()
                if line and line != original:
                    queries.append(line)

        return queries
