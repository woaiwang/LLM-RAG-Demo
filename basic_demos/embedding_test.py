import numpy as np
from sentence_transformers import SentenceTransformer

def cosine_similarity(vec1, vec2):
    """
    手写余弦相似度公式: (A . B) / (||A|| * ||B||)
    范围: [-1, 1], 越高越相似
    """
    dot_val = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_val / (norm_a * norm_b)

def euclidean_distance(vec1, vec2):
    """
    手写欧氏距离公式: ||A - B||
    范围: [0, +∞), 越低越相似
    """
    return np.linalg.norm(vec1 - vec2)

def dot_product(vec1, vec2):
    """
    手写内积公式: A . B
    如果向量已归一化，则完全等于余弦相似度
    """
    return np.dot(vec1, vec2)

def main():
    print("📥 正在加载 Embedding 模型 (第一次运行需要下载约 100MB，请耐心等待)...")
    # 使用智源 BGE 小模型，中文效果好且速度快
    # normalize_embeddings=True 表示让模型直接输出归一化向量 (模长为1)
    # 这样 Dot Product 就直接等于 Cosine Similarity，计算更快
    model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
    
    sentences = [
        "我喜欢编程",         # 句子 A
        "写代码很有趣",       # 句子 B (语义相似)
        "今天午饭吃什么"      # 句子 C (语义无关)
    ]
    
    print("🧠 正在将文本转化为向量...")
    # 这里我们故意先不开启 normalize_embeddings，为了演示各种距离的区别
    embeddings = model.encode(sentences, normalize_embeddings=False)
    
    vec_a = embeddings[0]
    vec_b = embeddings[1]
    vec_c = embeddings[2]
    
    # 打印向量维度，通常是 512
    print(f"向量维度: {vec_a.shape}") 
    
    # === 1. 余弦相似度 (Cosine) ===
    cos_ab = cosine_similarity(vec_a, vec_b)
    cos_ac = cosine_similarity(vec_a, vec_c)
    
    # === 2. 欧氏距离 (Euclidean) ===
    dist_ab = euclidean_distance(vec_a, vec_b)
    dist_ac = euclidean_distance(vec_a, vec_c)

    # === 3. 内积 (Dot Product) ===
    dot_ab = dot_product(vec_a, vec_b)
    dot_ac = dot_product(vec_a, vec_c)
    
    print("-" * 50)
    print(f"对比 A ('我喜欢编程') vs B ('写代码很有趣'):")
    print(f"  余弦相似度: {cos_ab:.4f} (越高越好)")
    print(f"  欧氏距离  : {dist_ab:.4f} (越低越好)")
    print(f"  内积      : {dot_ab:.4f}")
    
    print("-" * 50)
    print(f"对比 A ('我喜欢编程') vs C ('今天午饭吃什么'):")
    print(f"  余弦相似度: {cos_ac:.4f}")
    print(f"  欧氏距离  : {dist_ac:.4f}")
    print(f"  内积      : {dot_ac:.4f}")
    print("-" * 50)
    
    # 验证测试结论
    success = (cos_ab > cos_ac) and (dist_ab < dist_ac)
    if success:
        print("✅ 测试成功！语义相近的句子，余弦值更大，欧氏距离更小。")
    else:
        print("❌ 测试失败，请检查代码。")

if __name__ == "__main__":
    main()