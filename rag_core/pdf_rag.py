from langchain_community.document_loaders import PyPDFLoader
# ✅ 改用这个新的导入路径
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from openai import OpenAI
import os

# ================= 配置区域 =================
# 1. 设置 API (还是用你之前的 DeepSeek key)
os.environ["OPENAI_API_KEY"] = "sk-02315205540a43ec9f0c87241add5d2c" 
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com"

# 2. 初始化 Embedding 模型 (LangChain 封装版)
# 这会自动下载 BGE 模型，和昨天那个是一样的
print("1. 正在加载 Embedding 模型...")
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True} #只有当向量归一化后，内积才等于余弦相似度，计算更快
)

# ================= 核心逻辑 =================

def create_vector_db(pdf_path):
    """
    读取PDF -> 切片 -> 存入向量数据库
    """
    if not os.path.exists(pdf_path):
        print(f"❌ 错误：找不到文件 {pdf_path}")
        return None

    print(f"2. 正在加载 PDF: {pdf_path} ...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 关键步骤：Text Splitter (切片)
    # 为什么要切？因为文章太长，大模型吃不下，且切小了检索更准。
    # chunk_size=300: 每个块300个字
    # chunk_overlap=50: 每个块和上一块重叠50字（防止把句子切断）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, 
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(documents)
    print(f"   -> 文档已切分为 {len(splits)} 个片段")

    # 存入 Chroma 向量数据库
    # 这一步会自动把文本变成向量，并存到本地文件夹 './chroma_db'
    print("3. 正在建立向量索引 (第一次运行会慢)...")
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        persist_directory="./chroma_db" 
    )
    print("✅ 向量库构建完成！")
    return vectorstore

def chat(vectorstore, query):
    """
    标准的 RAG 检索流程
    """
    # 1. 检索 (Retrieve): 找最相关的3个片段
    # k=3 表示找前3名
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    docs = retriever.invoke(query)
    
    # 打印出来看看检索到了什么（调试用）
    print(f"\n🔍 检索到的相关片段 (Top 3):")
    context_text = ""
    for i, doc in enumerate(docs):
        print(f"   [{i+1}] {doc.page_content[:50]}...") # 只打印前50字
        context_text += doc.page_content + "\n"

    # 2. 增强 (Augment) & 生成 (Generate)
    # 这里我们不用 LangChain 复杂的 Chain，直接用最简单的 API 调用，让你看清楚原理
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_API_BASE"]
    )

    prompt = f"""
    你是一个专业助手。请基于以下【参考片段】回答用户问题。
    如果参考片段里没有答案，请直接说“文档中未提及”。
    
    【参考片段】：
    {context_text}
    
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
    # 第一次运行：构建数据库
    # 以后运行：如果 './chroma_db' 存在，可以直接 load，不用每次都 create
    # 这里为了演示，每次都重新创建
    
    # 自动寻找 data.pdf，无论是在当前目录还是上一级 data 目录
    pdf_path = "data.pdf"
    if not os.path.exists(pdf_path):
        pdf_path = "../data/data.pdf" # 尝试去上级目录找
    
    db = create_vector_db(pdf_path) 
    
    if db:
        while True:
            user_input = input("\n🙋‍♂️ 请输入问题 (输入 'q' 退出): ")
            if user_input == 'q':
                break
            
            answer = chat(db, user_input)
            print(f"\n🤖 回答:\n{answer}")
            print("-" * 50)