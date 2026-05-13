"""
RAG 消融实验综合评估脚本

运行 7 组配置，产出检索指标 + 按类别分析。
可选：对关键配置运行 LLM-as-Judge 生成质量评估。
"""
import json
import os
import sys
import time
from statistics import mean

sys.path.insert(0, 'D:/LLM-RAG-Internship')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from rag_core.document_manager import DocumentManager
from rag_core.retrievers import create_retriever
from rag_core.query_processor import QueryProcessor
from rag_core.reranker import Reranker
from rag_core.generator import Generator
from config import FINAL_RETRIEVE_K, MULTI_QUERY_SEARCH_K, RERANK_TOP_K

# ======================== 1. 加载测试集 ========================
eval_path = 'D:/LLM-RAG-Internship/data/eval_dataset.jsonl'
dataset = []
with open(eval_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            dataset.append(json.loads(line))

print(f"[DATA] 加载测试集: {len(dataset)} 题")
cats = {}
for item in dataset:
    cats[item['category']] = cats.get(item['category'], 0) + 1
for c in sorted(cats):
    print(f"       类别 {c}: {cats[c]} 题")

# ======================== 2. 初始化 ========================
print("\n[INIT] 初始化 DocumentManager...")
doc_manager = DocumentManager()
docs = doc_manager.list_documents()
print(f"       {len(docs)} 份文档已索引:")
for d in docs:
    print(f"         - {d['original_filename']} ({d['chunk_count']} chunks)")

# 初始化各检索器
vector_retriever = create_retriever("vector", doc_manager)
bm25_retriever = create_retriever("bm25", doc_manager)
hybrid_retriever = create_retriever("hybrid", doc_manager)

# 初始化 Generator (需要 API Key)
generator = Generator()
query_processor = QueryProcessor(generator)

# 初始化 Reranker
reranker = Reranker()

# ======================== 3. 定义配置和检索函数 ========================

CONFIGS = [
    {"name": "1-Vector",     "strategy": "vector",  "enhance": "none",    "rerank": False},
    {"name": "2-BM25",       "strategy": "bm25",    "enhance": "none",    "rerank": False},
    {"name": "3-Hybrid",     "strategy": "hybrid",  "enhance": "none",    "rerank": False},
    {"name": "4-Hybrid+Rewrite", "strategy": "hybrid", "enhance": "rewrite", "rerank": False},
    {"name": "5-Hybrid+Expand",  "strategy": "hybrid", "enhance": "expand",  "rerank": False},
    {"name": "6-Hybrid+Reranker", "strategy": "hybrid", "enhance": "none",   "rerank": True},
    {"name": "7-Full",       "strategy": "hybrid",  "enhance": "rewrite", "rerank": True},
]

def get_retriever(strategy):
    if strategy == "vector":
        return vector_retriever
    elif strategy == "bm25":
        return bm25_retriever
    else:
        return hybrid_retriever

def retrieve_for_question(retriever, qp, rrk, query, config):
    """
    按配置检索，返回 (docs_list, timing_seconds)
    """
    t0 = time.time()
    if config["enhance"] != "none":
        docs = qp.retrieve_enhanced(
            retriever, query, mode=config["enhance"],
            search_k=10, final_k=10
        )
    else:
        docs = retriever.retrieve(query, k=10)

    if config["rerank"] and docs:
        docs = rrk.rerank(query, docs, k=5)

    elapsed = time.time() - t0
    return docs[:5], round(elapsed, 2)

def is_relevant(doc, source_doc_key):
    """检查文档是否来自目标来源文件"""
    filename = doc.metadata.get("filename", "")
    return source_doc_key in filename

# ======================== 4. 运行所有配置 ========================

all_results = {}

for cfg in CONFIGS:
    name = cfg["name"]
    print(f"\n{'='*60}")
    print(f"[RUN] {name}")
    print(f"{'='*60}")

    retriever = get_retriever(cfg["strategy"])
    per_query = []  # [{query, category, retrieved_texts, relevant_set, timing}]

    for item in dataset:
        query = item["question"]
        cat = item["category"]
        source_key = item["source_doc"]

        docs, timing = retrieve_for_question(retriever, query_processor, reranker, query, cfg)

        retrieved_texts = [d.page_content[:80] for d in docs]
        relevant_set = set()
        for d in docs:
            if is_relevant(d, source_key):
                relevant_set.add(d.page_content[:80])

        per_query.append({
            "category": cat,
            "query": query,
            "retrieved": retrieved_texts,
            "relevant": relevant_set,
            "n_retrieved": len(docs),
            "timing": timing,
        })

    # ---- 计算整体指标 ----
    hit_1, hit_3, hit_5 = [], [], []
    mrr_vals = []
    prec_3 = []

    for pq in per_query:
        rel = pq["relevant"]
        ret = pq["retrieved"]

        hit_1.append(1 if len(ret) > 0 and any(r in rel for r in ret[:1]) else 0)
        hit_3.append(1 if any(r in rel for r in ret[:3]) else 0)
        hit_5.append(1 if any(r in rel for r in ret[:5]) else 0)

        first = next((i+1 for i, r in enumerate(ret) if r in rel), 0)
        mrr_vals.append(1/first if first > 0 else 0)

        n_rel_k3 = sum(1 for r in ret[:3] if r in rel)
        prec_3.append(n_rel_k3 / min(3, len(ret)) if len(ret) > 0 else 0)

    overall = {
        "hit_rate_1": round(mean(hit_1), 3),
        "hit_rate_3": round(mean(hit_3), 3),
        "hit_rate_5": round(mean(hit_5), 3),
        "mrr": round(mean(mrr_vals), 3),
        "precision_3": round(mean(prec_3), 3),
        "avg_timing": round(mean([pq["timing"] for pq in per_query]), 2),
    }
    print(f"  整体: HR@1={overall['hit_rate_1']:.3f} HR@3={overall['hit_rate_3']:.3f} "
          f"MRR={overall['mrr']:.3f} P@3={overall['precision_3']:.3f} "
          f"t={overall['avg_timing']}s")

    # ---- 按类别计算 ----
    cat_results = {}
    for cat_label in sorted(cats.keys()):
        cat_items = [pq for pq in per_query if pq["category"] == cat_label]
        if not cat_items:
            continue
        c_hit1 = mean([1 if len(pq["retrieved"]) > 0 and any(r in pq["relevant"] for r in pq["retrieved"][:1]) else 0 for pq in cat_items])
        c_hit3 = mean([1 if any(r in pq["relevant"] for r in pq["retrieved"][:3]) else 0 for pq in cat_items])
        c_mrr = mean([1/next((i+1 for i,r in enumerate(pq["retrieved"]) if r in pq["relevant"]), 0) if next((i+1 for i,r in enumerate(pq["retrieved"]) if r in pq["relevant"]), 0) > 0 else 0 for pq in cat_items])
        cat_results[cat_label] = {
            "hit_rate_3": round(c_hit3, 3),
            "mrr": round(c_mrr, 3),
        }
        print(f"  类别 {cat_label}: HR@3={c_hit3:.3f} MRR={c_mrr:.3f}")

    all_results[name] = {
        "overall": overall,
        "by_category": cat_results,
        "per_query": per_query,
    }

# ======================== 5. 汇总输出 ========================

print("\n\n")
print("=" * 80)
print("                         评 估 结 果 汇 总")
print("=" * 80)

print(f"\n{'配置':<22} {'HR@1':<8} {'HR@3':<8} {'HR@5':<8} {'MRR':<8} {'P@3':<8} {'耗时':<8}")
print("-" * 80)
for cfg in CONFIGS:
    name = cfg["name"]
    res = all_results[name]["overall"]
    print(f"{name:<22} {res['hit_rate_1']:<8.3f} {res['hit_rate_3']:<8.3f} "
          f"{res['hit_rate_5']:<8.3f} {res['mrr']:<8.3f} {res['precision_3']:<8.3f} "
          f"{res['avg_timing']:<8.2f}s")

# ---- 按类别对比表 ----
print(f"\n--- 按类别 HR@3 对比 ---")
cat_names = {"A": "精确事实", "B": "口语化术语", "C": "条件排除", "D": "跨文档关联", "E": "复杂多维度"}
header = f"{'配置':<22}"
for cl in sorted(cats.keys()):
    header += f" {cat_names.get(cl, cl):<12}"
print(header)
print("-" * 80)
for cfg in CONFIGS:
    name = cfg["name"]
    line = f"{name:<22}"
    for cl in sorted(cats.keys()):
        val = all_results[name]["by_category"].get(cl, {}).get("hit_rate_3", 0)
        line += f" {val:<12.3f}"
    print(line)

# ======================== 6. 保存结果 ========================
output_path = 'D:/LLM-RAG-Internship/data/eval_results.json'
# 只保存摘要（去掉 per_query 避免文件太大）
summary = {}
for cfg in CONFIGS:
    name = cfg["name"]
    summary[name] = {
        "overall": all_results[name]["overall"],
        "by_category": all_results[name]["by_category"],
    }
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"\n[SAVE] 结果保存至: {output_path}")

# ======================== 7. 生成质量评估（LLM-as-Judge）=======================
print(f"\n{'='*60}")
print("[SKIP] LLM-as-Judge 生成质量评估暂未运行")
print("       需要时取消下方注释块即可运行")
print(f"{'='*60}")

"""
# 对 ① ③ ⑦ 三种关键配置做生成质量评估
GEN_CONFIGS = ["1-Vector", "3-Hybrid", "7-Full"]
for cfg_name in GEN_CONFIGS:
    print(f"\n[GEN] {cfg_name} - 正在评估生成质量...")
    ...
"""
