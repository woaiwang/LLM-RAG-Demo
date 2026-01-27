from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from FlagEmbedding import FlagReranker 
from openai import OpenAI
import os
import shutil 

# ================= 配置区域 =================
os.environ["OPENAI_API_KEY"] = "sk-02315205540a43ec9f0c87241add5d2c" 
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com"

# 1. Embedding 模型 (保持不变，本来就很小)
print("1. 正在加载 Embedding 模型...")
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 2. Reranker 模型 (🔴 改为 Lite 版配置)
print("2. 正在加载 Reranker 模型 (Lite版)...")
# 'use_fp16=False' 是关键，CPU 必须关掉混合精度，否则会报错或乱码
reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=False) 

# ================= 核心逻辑 (不变) =================

def create_vector_db(pdf_path):
    if os.path.exists("./chroma_db_lite"):
        shutil.rmtree("./chroma_db_lite")

    if not os.path.exists(pdf_path):
        print(f"❌ 错误：找不到文件 {pdf_path}")
        return None

    print(f"3. 正在加载 PDF: {pdf_path} ...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    splits = text_splitter.split_documents(documents)
    print(f"   -> 文档已切分为 {len(splits)} 个片段")

    print("4. 正在建立向量索引...")
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        persist_directory="./chroma_db_lite" 
    )
    print("✅ 向量库构建完成！")
    return vectorstore

def advanced_chat(vectorstore, query):
    print(f"\n🔍 [1/3] 粗排检索 (Top 10)...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    init_docs = retriever.invoke(query)
    
    print(f"⚖️  [2/3] 重排序 (Rerank)...")
    pairs = [[query, doc.page_content] for doc in init_docs]
    scores = reranker.compute_score(pairs)
    
    if isinstance(scores, float): scores = [scores]
    results = list(zip(init_docs, scores))
    results.sort(key=lambda x: x[1], reverse=True)
    
    top_3 = results[:3]
    
    print("\n🏆 精排后的 Top 3 片段:")
    final_context = ""
    for i, (doc, score) in enumerate(top_3):
        print(f"   [{i+1}] Score: {score:.4f} | {doc.page_content[:30].replace(chr(10), ' ')}...")
        final_context += doc.page_content + "\n"

    print(f"🤖 [3/3] 生成回答...")
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_API_BASE"]
    )

    prompt = f"""
    你是一个专业助手。请基于以下【参考片段】回答用户问题。
    如果参考片段里没有答案，请直接说“文档中未提及”。
    
    【参考片段】：
    {final_context}
    
    【用户问题】：
    {query}
    """

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    db = create_vector_db("data.pdf")
    if db:
        while True:
            user_input = input("\n🙋‍♂️ 请输入问题 (输入 'q' 退出): ")
            if user_input == 'q': break
            print(f"\n💡 AI 回答:\n{advanced_chat(db, user_input)}")
            print("-" * 50)