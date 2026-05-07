"""
RAG 系统评估器

使用 LLM-as-Judge 范式评估生成质量:
  - Faithfulness:   回答是否忠实于检索片段，不编造
  - Relevance:     回答是否切题
  - Completeness:  回答是否完整

同时支持检索指标（Hit Rate, MRR 等）。
"""

import json
import re
from typing import List, Dict, Optional

from evaluation.test_dataset import TestDataset
from evaluation.metrics import (
    average_hit_rate_at_k,
    mean_reciprocal_rank,
    average_precision_at_k,
)
from rag_core.generator import Generator
from rag_core.retrievers import Retriever

from config import FINAL_RETRIEVE_K


class Evaluator:
    """RAG 系统评估器，支持检索和生成质量评估。"""

    def __init__(self, generator: Optional[Generator] = None):
        self.generator = generator

    # ---------------------------------------------------------------
    # 检索评估
    # ---------------------------------------------------------------
    def evaluate_retrieval(
        self,
        retriever: Retriever,
        dataset: TestDataset,
        k: int = FINAL_RETRIEVE_K,
    ) -> Dict:
        """
        评估检索器的效果。

        对每个测试问题，运行检索并检查 source_doc 是否出现在检索结果中。
        """
        all_retrieved_texts = []
        all_relevant_sources = []

        for item in dataset:
            query = item["question"]
            source_doc = item.get("source_doc", "")

            # 检索
            docs = retriever.retrieve(query, k=k)

            # 提取检索到的文档内容（用于匹配）
            retrieved_texts = [d.page_content[:50] for d in docs]

            # 标记相关文档（这里简单地用 source_doc 文件名校验）
            relevant = set()
            if source_doc:
                for doc in docs:
                    meta_source = doc.metadata.get("filename", "")
                    if source_doc in meta_source:
                        relevant.add(doc.page_content[:50])

            all_retrieved_texts.append(retrieved_texts)
            all_relevant_sources.append(relevant)

        # 计算指标
        results = {
            "hit_rate": average_hit_rate_at_k(all_retrieved_texts, all_relevant_sources, k),
            "mrr": mean_reciprocal_rank(all_retrieved_texts, all_relevant_sources),
            "precision": average_precision_at_k(all_retrieved_texts, all_relevant_sources, k),
            "sample_count": len(dataset),
        }
        return results

    # ---------------------------------------------------------------
    # 生成质量评估 (LLM-as-Judge)
    # ---------------------------------------------------------------
    def evaluate_generation(
        self,
        retriever: Retriever,
        dataset: TestDataset,
        k: int = FINAL_RETRIEVE_K,
    ) -> Dict:
        """
        评估生成质量。对每个问题:
          1. 检索上下文
          2. 生成回答
          3. LLM-as-Judge 评分
        """
        if self.generator is None:
            raise ValueError("需要提供 Generator 实例才能评估生成质量")

        all_scores = {"faithfulness": [], "relevance": [], "completeness": []}
        details = []

        for item in dataset:
            query = item["question"]

            # 1. 检索
            docs = retriever.retrieve(query, k=k)
            context = "\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)])

            # 2. 生成回答
            system_prompt, user_prompt = self.generator.build_qa_prompt(context, query)
            answer = self.generator.generate(system_prompt, user_prompt)

            # 3. LLM-as-Judge 评分
            judge_system, judge_user = self.generator.build_judge_prompt(query, context, answer)
            judge_result = self.generator.generate(judge_system, judge_user)

            # 解析 JSON 评分
            scores = self._parse_judge_result(judge_result)
            for key in all_scores:
                all_scores[key].append(scores.get(key, 0))

            details.append({
                "question": query,
                "answer": answer,
                **scores,
            })

        # 计算均值
        results = {
            "faithfulness": self._safe_mean(all_scores["faithfulness"]),
            "relevance": self._safe_mean(all_scores["relevance"]),
            "completeness": self._safe_mean(all_scores["completeness"]),
            "overall": self._safe_mean(
                all_scores["faithfulness"]
                + all_scores["relevance"]
                + all_scores["completeness"]
            ),
            "details": details,
            "sample_count": len(dataset),
        }
        return results

    # ---------------------------------------------------------------
    # 对比评估（多个策略对比）
    # ---------------------------------------------------------------
    def compare_retrieval_strategies(
        self,
        retrievers: Dict[str, Retriever],
        dataset: TestDataset,
        k: int = FINAL_RETRIEVE_K,
    ) -> Dict:
        """对比多个检索策略的检索指标"""
        results = {}
        for name, retriever in retrievers.items():
            results[name] = self.evaluate_retrieval(retriever, dataset, k)
        return results

    # ---------------------------------------------------------------
    # 工具方法
    # ---------------------------------------------------------------
    @staticmethod
    def _parse_judge_result(text: str) -> Dict:
        """从 LLM 输出中解析 JSON 评分"""
        # 尝试提取 JSON 块
        json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return {"faithfulness": 0, "relevance": 0, "completeness": 0}

    @staticmethod
    def _safe_mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0
