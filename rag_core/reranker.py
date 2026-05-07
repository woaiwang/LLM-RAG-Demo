"""
Cross-Encoder 重排序器 (Reranker)

作用:
  - 对检索结果的二次精排
  - 使用 Cross-Encoder 对每对 (query, doc) 进行深度交互计算
  - 比双编码器（Bi-Encoder）更精确，但速度较慢

为什么用 CrossEncoder 而非 FlagEmbedding:
  - FlagEmbedding 与 transformers >= 5.0 不兼容
  - sentence_transformers 内置 CrossEncoder，接口更稳定
  - 底层使用同样的 BGE-Reranker 模型，效果一致

流程:
  Retriever → Top-10 → Reranker (Cross-Encoder) → Top-3 → LLM
"""

from typing import List, Optional

from langchain_core.documents import Document

from config import RERANKER_MODEL, RERANK_TOP_K


# 全局缓存，避免重复加载模型（兼容 Streamlit 和非 Streamlit 环境）
_reranker_model = None


def _load_reranker_model(model_name: str):
    """
    懒加载 Reranker 模型。

    使用模块级全局变量缓存模型，这样即使多次调用 load_reranker_model()，
    模型也只会被加载一次。
    """
    global _reranker_model
    if _reranker_model is None:
        from sentence_transformers import CrossEncoder
        _reranker_model = CrossEncoder(model_name)
    return _reranker_model


class Reranker:
    """
    Cross-Encoder 重排序器。

    对检索结果进行二次精排，将 query 与每个 doc 配对输入 Cross-Encoder，
    输出相关性分数，按分数降序排列后返回 Top-K。
    """

    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        """获取模型（懒加载 + 缓存）"""
        if self._model is None:
            try:
                self._model = _load_reranker_model(self.model_name)
            except Exception as e:
                self._model = None
        return self._model

    def rerank(
        self,
        query: str,
        documents: List[Document],
        k: int = RERANK_TOP_K,
    ) -> List[Document]:
        """
        对检索结果做 Cross-Encoder 重排序。

        Args:
            query: 用户查询
            documents: 检索到的文档列表
            k: 最终保留的文档数

        Returns:
            重排序后的 Top-K 文档

        异常处理:
          - 模型加载失败 → 返回原始文档（截取 Top-K）
          - 文档数不足 → 直接返回
          - 空列表 → 返回空
        """
        if not documents:
            return []

        if len(documents) <= k:
            return documents

        model = self._get_model()
        if model is None:
            return documents[:k]

        try:
            pairs = [[query, doc.page_content] for doc in documents]
            scores = model.predict(pairs, show_progress_bar=False)

            scored = list(zip(documents, scores))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in scored[:k]]
        except Exception as e:
            # 任何错误都降级为原始排序
            return documents[:k]
