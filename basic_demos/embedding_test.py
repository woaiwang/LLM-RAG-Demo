import numpy as np
from sentence_transformers import SentenceTransformer

def cosine_similarity(vec1, vec2):
    """
    手写余弦相似度公式: (A . B) / (||A|| * ||B||)
    """
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b)

def main():
    print("📥 正在加载 Embedding 模型 (第一次运行需要下载约 100MB，请耐心等待)...")
    # 使用智源 BGE 小模型，中文效果好且速度快
    model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
    
    sentences = [
        "我喜欢编程",         # 句子 A
        "写代码很有趣",       # 句子 B (语义相似)
        "今天午饭吃什么"      # 句子 C (语义无关)
    ]
    
    print("🧠 正在将文本转化为向量...")
    embeddings = model.encode(sentences)
    
    vec_a = embeddings[0]
    vec_b = embeddings[1]
    vec_c = embeddings[2]
    
    # 打印向量维度，通常是 512
    print(f"向量维度: {vec_a.shape}") 
    
    # 计算相似度
    score_ab = cosine_similarity(vec_a, vec_b)
    score_ac = cosine_similarity(vec_a, vec_c)
    
    print("-" * 30)
    print(f"A ('我喜欢编程') vs B ('写代码很有趣') 相似度: {score_ab:.4f}")
    print(f"A ('我喜欢编程') vs C ('今天午饭吃什么') 相似度: {score_ac:.4f}")
    print("-" * 30)
    
    if score_ab > score_ac:
        print("✅ 测试成功！意思相近的句子，向量距离更近。")
    else:
        print("❌ 测试失败，请检查代码。")

if __name__ == "__main__":
    main()