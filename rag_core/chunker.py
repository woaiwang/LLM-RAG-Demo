"""
文本切片策略封装

设计思路:
  抽象 BaseChunker 接口，当前实现 RecursiveChunker（默认策略），
  后续可扩展 SemanticChunker（基于 Embedding 的语义分割）等。
"""

from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class BaseChunker(ABC):
    """切片策略基类"""

    @abstractmethod
    def chunk(self, documents: List[Document]) -> List[Document]:
        ...


class RecursiveChunker(BaseChunker):
    """
    递归字符分割器（默认）
    - chunk_size: 每个块的目标字符数
    - chunk_overlap: 块之间的重叠字符数（防止切断关键上下文）
    """

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)
