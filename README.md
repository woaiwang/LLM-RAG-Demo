# 🤖 垂直领域智能知识库助手 (Vertical Domain RAG Assistant)

> 基于 LangChain + RAG + LoRA 微调的智能问答系统，解决通用大模型在特定领域（如企业文档、校园教务）存在的知识幻觉与时效性问题。

## ✨ 项目亮点 (Key Features)

*   **⚡️ 端到端 RAG 架构**: 实现了从 PDF 文档解析、分块 (Chunking)、向量化 (Embedding) 到检索 (Retrieval) 的完整链路。
*   **🎯 双阶段检索优化**: 引入 **BGE-Reranker** 模型，采用“Recall (粗排) + Rerank (精排)”策略，显著解决了向量检索语义匹配不准的问题。
*   **🧠 身份认知微调**: 基于 **XTuner** 框架对 **Qwen-1.8B** 模型进行 LoRA 微调，成功注入特定的身份认知（Identity），消除模型幻觉。
*   **🖥 可视化交互**: 使用 **Streamlit** 搭建了友好的 Web 聊天界面，支持 PDF 实时上传与知识库热更新。

## 🛠️ 技术栈 (Tech Stack)

*   **大语言模型 (LLM)**: DeepSeek V3 (API) / Qwen-1.8B (Local)
*   **开发框架**: LangChain, PyTorch
*   **向量数据库**: ChromaDB
*   **检索与重排**: BAAI/bge-small-zh, BAAI/bge-reranker
*   **微调工具**: XTuner, PEFT, LoRA
*   **前端界面**: Streamlit

## 📂 目录结构 (Directory Structure)

```text
LLM-RAG-Internship/
├── app.py                  # Streamlit Web 应用入口 (主程序)
├── rag_core/               # RAG 核心算法模块
│   ├── pdf_rag.py          # 基础 RAG 实现
│   └── advanced_rag.py     # 进阶 RAG 实现 (加入 Rerank)
├── basic_demos/            # 基础原理验证代码 (API调用, 余弦相似度手写实现)
├── data/                   # 测试用 PDF 数据
├── requirements.txt        # 项目依赖库
└── README.md               # 项目说明文档
```

## 🚀 快速开始 (Quick Start)

### 1. 环境准备
建议使用 Conda 创建虚拟环境：
```bash
conda create -n rag_env python=3.10
conda activate rag_env
pip install -r requirements.txt
```

### 2. 配置 API Key
请在 `app.py` 中填写你的 API Key，或在 Web 界面侧边栏输入。

### 3. 启动应用
```bash
streamlit run app.py
```
启动后，浏览器会自动打开 `http://localhost:8501`。

## 📝 核心原理解析

### 1. 为什么需要 RAG？
通用大模型虽然知识渊博，但无法获知“私有数据”（如个人文档、公司财报）。通过 RAG 技术，我们将私有数据转化为向量存储，当用户提问时，系统先检索相关片段，再将其作为“参考资料”喂给大模型，从而实现精准问答。

### 2. Rerank 重排序的意义
传统的向量检索（Bi-Encoder）基于语义相似度，虽然速度快，但容易检索到“似是而非”的内容。本项目引入了 Cross-Encoder (Reranker) 进行二次精排，通过对【问题-文档】对进行深度交互计算，大幅提升了 Top-3 的准确率。

---

**Author**:查志渊