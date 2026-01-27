import numpy as np
from sentence_transformers import SentenceTransformer
from llm_api import get_completion # 引用你刚才写的LLM函数

# 1. 初始化模型 (加载一次即可，避免重复加载耗时)
print("正在加载 Embedding 模型...")
emb_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

# 2. 模拟一个简单的“向量数据库”
# 假设这是某公司的内部规定，大模型训练时没见过这些
knowledge_base = [
    "公司的午休时间是 12:00 到 14:00。",
    "申请年假需要提前 3 天在钉钉上提交审批。",
    "报销打车费需要提供纸质发票，且金额不能超过 200 元。",
    "公司的无线密码是: hello_world_2024。",
]

# 提前把知识库变成向量 (Indexing)
# 在实际生产中，这步是存进 ChromaDB/Milvus 里的，这里我们用内存存
print("正在构建知识库索引...")
kb_embeddings = emb_model.encode(knowledge_base)

def retrieve(query):
    """
    检索函数：找最相似的那条知识
    """
    # 1. 把用户的问题也变成向量
    query_vec = emb_model.encode(query)
    
    # 2. 计算相似度 (用你之前的逻辑)
    scores = []
    for doc_vec in kb_embeddings:
        # 手写余弦相似度
        dot = np.dot(query_vec, doc_vec)
        norm_q = np.linalg.norm(query_vec)
        norm_d = np.linalg.norm(doc_vec)
        score = dot / (norm_q * norm_d)
        scores.append(score)
    
    # 3. 找到得分最高的索引
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    
    print(f"🔍 检索结果: 匹配到第 {best_idx} 条，相似度: {best_score:.4f}")
    return knowledge_base[best_idx]

def rag_chat(user_query):
    # R (Retrieve): 检索
    best_context = retrieve(user_query)
    
    # A (Augment): 增强 Prompt
    # 这是 RAG 的灵魂：不仅给问题，还把刚才查到的“答案”贴在脸上传给它
    prompt = f"""
    你是一个智能助手。请根据下面的【参考资料】回答用户的问题。
    如果你不知道，就说不知道，不要瞎编。

    【参考资料】：
    {best_context}

    【用户问题】：
    {user_query}
    """
    
    # G (Generate): 生成
    answer = get_completion(prompt)
    return answer

if __name__ == "__main__":
    # 测试案例
    questions = [
        "公司的wifi密码是多少？",
        "我想要报销300元的打车费可以吗？",
        "怎么请年假？"
    ]
    
    for q in questions:
        print(f"\n🙋‍♂️ 用户提问: {q}")
        response = rag_chat(q)
        print(f"🤖 AI 回答: {response}")
        print("-" * 50)