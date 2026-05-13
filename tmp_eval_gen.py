"""
LLM-as-Judge 生成质量评估

对 ① Vector ③ Hybrid ④ Hybrid+Rewrite 三种配置，
逐题生成回答并用 DeepSeek 自动评分。
"""
import json, os, sys, re, time
from statistics import mean

sys.path.insert(0, 'D:/LLM-RAG-Internship')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from rag_core.document_manager import DocumentManager
from rag_core.retrievers import create_retriever
from rag_core.query_processor import QueryProcessor
from rag_core.generator import Generator

# ======================== 加载测试集 ========================
dataset = []
with open('D:/LLM-RAG-Internship/data/eval_dataset.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            dataset.append(json.loads(line))
print(f"[DATA] 加载 {len(dataset)} 道测试题")

# ======================== 初始化 ========================
doc_manager = DocumentManager()
vector_ret = create_retriever("vector", doc_manager)
hybrid_ret = create_retriever("hybrid", doc_manager)
generator = Generator()
query_processor = QueryProcessor(generator)

EVAL_CONFIGS = {
    "1-Vector": {"retriever": vector_ret, "enhance": False},
    "3-Hybrid": {"retriever": hybrid_ret, "enhance": False},
    "4-Hybrid+Rewrite": {"retriever": hybrid_ret, "enhance": True},
}

def build_context(docs):
    parts = []
    for i, d in enumerate(docs):
        parts.append(f"[{i+1}] {d.page_content}")
    return "\n".join(parts)

def parse_judge(text):
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except:
            pass
    return {"faithfulness": 0, "relevance": 0, "completeness": 0}

results = {}

for cfg_name, cfg in EVAL_CONFIGS.items():
    print(f"\n{'='*60}")
    print(f"[GEN] {cfg_name}")
    print(f"{'='*60}")

    retriever = cfg["retriever"]
    all_scores = {"faithfulness": [], "relevance": [], "completeness": []}
    details = []

    for idx, item in enumerate(dataset):
        query = item["question"]
        cat = item["category"]

        # 1. 检索
        if cfg["enhance"]:
            docs = query_processor.retrieve_enhanced(retriever, query, mode="rewrite", search_k=10, final_k=3)
        else:
            docs = retriever.retrieve(query, k=3)

        context = build_context(docs)

        # 2. 生成回答
        sys_p, user_p = generator.build_qa_prompt(context, query)
        try:
            answer = generator.generate(sys_p, user_p)
        except Exception as e:
            print(f"  [{idx+1}] 生成失败: {e}")
            answer = ""

        # 3. LLM-as-Judge
        judge_sys, judge_user = generator.build_judge_prompt(query, context, answer)
        try:
            judge_text = generator.generate(judge_sys, judge_user)
            scores = parse_judge(judge_text)
        except Exception as e:
            print(f"  [{idx+1}] 评分失败: {e}")
            scores = {"faithfulness": 0, "relevance": 0, "completeness": 0}

        for k in all_scores:
            all_scores[k].append(scores.get(k, 0))

        details.append({
            "question": query,
            "category": cat,
            "answer": answer[:100],
            "scores": scores,
        })

        status = f"[{idx+1}/{len(dataset)}] ({cat}) {query[:25]}... "
        status += f"F={scores.get('faithfulness',0):.1f} R={scores.get('relevance',0):.1f} C={scores.get('completeness',0):.1f}"
        print(status)

    results[cfg_name] = {
        "faithfulness": round(mean(all_scores["faithfulness"]), 2),
        "relevance": round(mean(all_scores["relevance"]), 2),
        "completeness": round(mean(all_scores["completeness"]), 2),
        "overall": round(mean(all_scores["faithfulness"] + all_scores["relevance"] + all_scores["completeness"]), 2),
        "details": details,
    }
    print(f"  >> {cfg_name}: F={results[cfg_name]['faithfulness']} R={results[cfg_name]['relevance']} C={results[cfg_name]['completeness']} O={results[cfg_name]['overall']}")

# ======================== 汇总输出 ========================
print("\n\n" + "=" * 70)
print("           LLM-as-Judge 生成质量评估结果")
print("=" * 70)
print(f"\n{'配置':<22} {'Faithfulness':<16} {'Relevance':<16} {'Completeness':<16} {'Overall':<16}")
print("-" * 70)
for cfg_name in EVAL_CONFIGS:
    r = results[cfg_name]
    print(f"{cfg_name:<22} {r['faithfulness']:<16.2f} {r['relevance']:<16.2f} {r['completeness']:<16.2f} {r['overall']:<16.2f}")

# 保存结果
output = {
    name: {k: v for k, v in res.items() if k != "details"}
    for name, res in results.items()
}
output_path = 'D:/LLM-RAG-Internship/data/eval_gen_results.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"\n[SAVE] 结果已保存至: {output_path}")
