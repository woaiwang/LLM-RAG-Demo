import streamlit as st
import os
import shutil
# 复用你之前写的 RAG 逻辑
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ================= 页面配置 =================
st.set_page_config(page_title="RAG 智能知识库", layout="wide")
st.title("🤖 垂直领域智能知识库 (RAG)")

# ================= 初始化 Session State =================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ================= 初始化模型 (缓存以提高速度) =================
@st.cache_resource
def get_embeddings():
    return HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

embeddings = get_embeddings()

# ================= 初始化侧边栏 =================
with st.sidebar:
    st.header("⚙️ 配置面板")
    # API Key 输入框 (优先从环境变量读取，或者是用户输入)
    default_key = os.getenv("DEEPSEEK_API_KEY", "")
    api_key = st.text_input("DeepSeek API Key", type="password", value=default_key)
    
    # 文件上传
    uploaded_file = st.file_uploader("上传 PDF 知识库", type=["pdf"])
    
    # 按钮
    if st.button("🔄 重建知识库"):
        if uploaded_file and api_key:
            with st.spinner("正在处理文档... (请勿刷新)"):
                try:
                    # 1. 保存上传的文件
                    with open("temp.pdf", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # 2. 清理旧库 (防止残留)
                    if os.path.exists("./chroma_db_web"):
                        shutil.rmtree("./chroma_db_web")
                    
                    # 3. 加载与切片
                    loader = PyPDFLoader("temp.pdf")
                    docs = loader.load()
                    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
                    splits = splitter.split_documents(docs)
                    
                    # 4. 建库
                    vectorstore = Chroma.from_documents(
                        documents=splits, 
                        embedding=embeddings,
                        persist_directory="./chroma_db_web"
                    )
                    st.session_state.vectorstore = vectorstore
                    st.success(f"✅ 知识库构建成功！共 {len(splits)} 个片段。")
                except Exception as e:
                    st.error(f"❌ 构建失败: {str(e)}")
        else:
            st.error("请先上传文件并填写 API Key！")

# ================= 聊天主界面 =================

# 1. 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 2. 处理用户输入
if prompt := st.chat_input("请输入你的问题..."):
    # 显示用户问题
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 生成回答
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # 检查向量库状态
        if not st.session_state.vectorstore:
            # 尝试自动加载持久化的数据库
            if os.path.exists("./chroma_db_web"):
                try:
                    st.session_state.vectorstore = Chroma(
                        persist_directory="./chroma_db_web", 
                        embedding_function=embeddings
                    )
                except Exception as e:
                    st.error("⚠️ 数据库加载失败，请重新上传文件重建！")
                    st.stop()
            else:
                 full_response = "⚠️ 请先在左侧上传文档并构建知识库！"
                 message_placeholder.warning(full_response)
                 st.session_state.messages.append({"role": "assistant", "content": full_response})
                 st.stop()
        
        try:
            # === RAG 检索 ===
            vectorstore = st.session_state.vectorstore
            # K=3 召回
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(prompt)
            
            # 拼接上下文
            context = "\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)])
            
            # === 调用LLM ===
            client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            
            sys_prompt = f"""你是一个专业助手。请严格基于参考资料回答问题。
            如果资料不足，请说明"文档中未提及"。
            
            参考资料：
            {context}
            """
            
            stream = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                stream=True 
            )
            
            # 流式输出
            full_response = ""
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response) # 最后一次更新，去掉光标
            
            # 存入历史
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"❌ 出错了: {str(e)}")
