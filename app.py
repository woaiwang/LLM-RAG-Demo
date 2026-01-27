import streamlit as st
import os
import shutil
# 复用你之前写的 RAG 逻辑
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from openai import OpenAI

# ================= 页面配置 =================
st.set_page_config(page_title="王同学的 AI 助手", layout="wide")
st.title("🤖 垂直领域智能知识库 (RAG + Rerank)")

# ================= 初始化侧边栏 =================
with st.sidebar:
    st.header("⚙️ 配置面板")
    # API Key 输入框 (安全起见)
    api_key = st.text_input("DeepSeek API Key", type="password", value="sk-02315205540a43ec9f0c87241add5d2c")
    
    # 文件上传
    uploaded_file = st.file_uploader("上传 PDF 知识库", type=["pdf"])
    
    # 按钮
    if st.button("🔄 重建知识库"):
        if uploaded_file and api_key:
            with st.spinner("正在处理文档... (请勿刷新)"):
                # 1. 保存上传的文件
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # 2. 调用之前的建库逻辑 (为了简单，直接写在这里)
                # 清理旧库
                if os.path.exists("./chroma_db_web"):
                    shutil.rmtree("./chroma_db_web")
                
                # 加载 Embedding
                embeddings = HuggingFaceBgeEmbeddings(
                    model_name="BAAI/bge-small-zh-v1.5",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                
                # 切片
                loader = PyPDFLoader("temp.pdf")
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
                splits = splitter.split_documents(docs)
                
                # 建库
                vectorstore = Chroma.from_documents(
                    documents=splits, 
                    embedding=embeddings,
                    persist_directory="./chroma_db_web"
                )
                st.session_state['vectorstore'] = vectorstore
                st.success(f"✅ 知识库构建成功！共 {len(splits)} 个片段。")
        else:
            st.error("请先上传文件并填写 API Key！")

# ================= 聊天主界面 =================
# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 处理用户输入
if prompt := st.chat_input("请输入你的问题..."):
    # 1. 显示用户问题
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. 生成回答
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        if 'vectorstore' not in st.session_state:
            st.warning("⚠️ 请先在左侧上传文档并构建知识库！")
            full_response = "请先配置知识库。"
        else:
            try:
                # === RAG 逻辑 ===
                vectorstore = st.session_state['vectorstore']
                
                # 粗排
                retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                docs = retriever.invoke(prompt)
                
                # (这里为了响应速度，Web版我们先暂时去掉 Rerank)
                # 直接拼接
                context = "\n".join([d.page_content for d in docs])
                
                # 调用 API
                client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
                
                sys_prompt = f"""你是一个专业助手。基于参考资料回答问题。
                参考资料：
                {context}
                """
                
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    stream=True 
                )
                
                # 流式输出
                full_response = ""
                for chunk in response:
                    # ✅ 修正了这里的语法错误
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
            except Exception as e:
                full_response = f"❌ 出错了: {str(e)}"
                message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})