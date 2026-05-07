"""
评估测试集管理

测试集格式 (JSONL):
  {"question": "...", "ground_truth": "...", "source_doc": "文件名"}
  - question:     用户问题
  - ground_truth: 标准答案（人工标注）
  - source_doc:   答案来源的文件名（用于检查检索命中）
"""

import json
import os
from typing import List, Dict, Optional

from config import EVAL_DATASET_PATH


class TestDataset:
    """管理 RAG 评估测试集。"""

    def __init__(self, path: str = EVAL_DATASET_PATH):
        self.path = path
        self.data: List[Dict] = []
        if os.path.exists(path):
            self._load()

    # ---------------------------------------------------------------
    # 数据加载与保存
    # ---------------------------------------------------------------
    def _load(self):
        self.data = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

    def save(self, path: Optional[str] = None):
        save_path = path or self.path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            for item in self.data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # ---------------------------------------------------------------
    # 数据操作
    # ---------------------------------------------------------------
    def add_item(self, question: str, ground_truth: str, source_doc: str = ""):
        self.data.append({
            "question": question,
            "ground_truth": ground_truth,
            "source_doc": source_doc,
        })

    def remove_item(self, index: int):
        if 0 <= index < len(self.data):
            self.data.pop(index)

    def get_item(self, index: int) -> Optional[Dict]:
        if 0 <= index < len(self.data):
            return self.data[index]
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
