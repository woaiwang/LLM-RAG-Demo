import os
from dotenv import load_dotenv

load_dotenv()

# ========== LLM 配置 ==========
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# ========== Embedding 配置 ==========
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DEVICE = "cpu"
EMBEDDING_NORMALIZE = True

# ========== Chunk 配置 ==========
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# ========== 检索配置 ==========
VECTOR_RETRIEVE_K = 10
HYBRID_RETRIEVE_K = 10  # 粗排阶段
FINAL_RETRIEVE_K = 3     # 最终返回给 LLM 的文档数
RRF_K = 60               # RRF 融合常数

# ========== ChromaDB 配置 ==========
CHROMA_PERSIST_DIR = "./chroma_db"

# ========== 路径配置 ==========
DATA_DIR = "./data"
PDFS_DIR = os.path.join(DATA_DIR, "pdfs")
EVAL_DATASET_PATH = os.path.join(DATA_DIR, "eval_dataset.jsonl")

# ========== Reranker 配置 ==========
RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANK_TOP_K = 3

# ========== Multi-Query 配置 ==========
MULTI_QUERY_SEARCH_K = 10

# ========== 生成配置 ==========
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024
