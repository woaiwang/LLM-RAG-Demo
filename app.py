"""
垂直领域智能知识库 (RAG) — 多文档版

改进功能:
  1. 多文档上传与独立管理
  2. 多策略检索 (Vector / BM25 / Hybrid)
  3. 检索来源引用显示
  4. RAG 评估 Dashboard
"""

import streamlit as st
import os
import time
import json

from dotenv import load_dotenv

from config import DEEPSEEK_API_KEY, FINAL_RETRIEVE_K, RERANK_TOP_K, MULTI_QUERY_SEARCH_K
from rag_core.document_manager import DocumentManager
from rag_core.retrievers import create_retriever
from rag_core.generator import Generator
from rag_core.query_processor import QueryProcessor
from rag_core.reranker import Reranker

load_dotenv()

# ================= 页面配置 =================
st.set_page_config(page_title="RAG 智能知识库", layout="wide")
st.title("垂直领域智能知识库 (RAG)")

# ================= 初始化 Session State =================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_manager" not in st.session_state:
    st.session_state.doc_manager = DocumentManager()
if "generator" not in st.session_state:
    st.session_state.generator = None
if "retrieval_strategy" not in st.session_state:
    st.session_state.retrieval_strategy = "hybrid"

# ================= 模型初始化 =================

@st.cache_resource
def load_local_qwen():
    """加载本地 LoRA 微调的 Qwen 模型（保持原有功能）"""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if not hasattr(transformers, "BeamSearchScorer"):
        transformers.BeamSearchScorer = type("BeamSearchScorer", (object,), {})
    if not hasattr(transformers, "DisjunctiveConstraint"):
        transformers.DisjunctiveConstraint = type("DisjunctiveConstraint", (object,), {})
    if not hasattr(transformers, "PhrasalConstraint"):
        transformers.PhrasalConstraint = type("PhrasalConstraint", (object,), {})
    if not hasattr(transformers, "Constraint"):
        transformers.Constraint = type("Constraint", (object,), {})
    if not hasattr(transformers, "ConstrainedBeamSearchScorer"):
        transformers.ConstrainedBeamSearchScorer = type("ConstrainedBeamSearchScorer", (object,), {})

    if not hasattr(transformers.generation.utils, "SampleOutput"):
        transformers.generation.utils.SampleOutput = type("SampleOutput", (object,), {})
    if not hasattr(transformers.generation.utils, "GreedySearchDecoderOnlyOutput"):
        transformers.generation.utils.GreedySearchDecoderOnlyOutput = type("GreedySearchDecoderOnlyOutput", (object,), {})
    if not hasattr(transformers.generation.utils, "BeamSearchDecoderOnlyOutput"):
        transformers.generation.utils.BeamSearchDecoderOnlyOutput = type("BeamSearchDecoderOnlyOutput", (object,), {})
    if not hasattr(transformers.generation.utils, "BeamSampleDecoderOnlyOutput"):
        transformers.generation.utils.BeamSampleDecoderOnlyOutput = type("BeamSampleDecoderOnlyOutput", (object,), {})

    model_id = "Qwen/Qwen-1_8B-Chat"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
        trust_remote_code=True,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
    )
    lora_path = "./models/qwen_lora_weights/"
    if os.path.exists(lora_path):
        model = PeftModel.from_pretrained(base_model, lora_path)
    else:
        model = base_model
    model.eval()
    return tokenizer, model, device


# ================= 侧边栏 =================

with st.sidebar:
    st.header("配置面板")

    # --- API Key ---
    default_key = os.getenv("DEEPSEEK_API_KEY", "")
    api_key = st.text_input("DeepSeek API Key", type="password", value=default_key)

    # --- LLM 选择 ---
    model_choice = st.radio("LLM:", ["DeepSeek API (云端)", "本地微调 Qwen (LoRA)"])

    st.divider()

    # ==================== 文档管理 ====================
    st.header("文档管理")

    uploaded_files = st.file_uploader(
        "上传 PDF 知识库（可多选）",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if not api_key:
            st.error("请先填写 API Key！")
        else:
            doc_manager = st.session_state.doc_manager
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in [d["original_filename"] for d in doc_manager.list_documents()]:
                    with st.status(f"正在处理: {uploaded_file.name}...") as status:
                        try:
                            doc_id = doc_manager.upload_pdf(
                                uploaded_file.getbuffer(),
                                uploaded_file.name,
                            )
                            status.update(label=f"完成: {uploaded_file.name}", state="complete")
                        except Exception as e:
                            status.update(label=f"失败: {uploaded_file.name}: {str(e)}", state="error")

    # 文档列表
    doc_manager = st.session_state.doc_manager
    docs = doc_manager.list_documents()
    if docs:
        st.markdown("**已上传文档:**")
        for d in docs:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(f"{d['original_filename']} ({d['chunk_count']} chunks)")
            with col2:
                if st.button("删除", key=f"del_{d['id']}"):
                    doc_manager.delete_document(d["id"])
                    st.rerun()
    else:
        st.caption("暂无文档，请上传 PDF")

    st.divider()

    # ==================== 检索策略 ====================
    st.header("检索策略")
    strategy = st.selectbox(
        "选择检索方式:",
        options=["hybrid", "vector", "bm25"],
        format_func=lambda x: {
            "hybrid": "Hybrid (向量 + BM25，RRF 融合)",
            "vector": "向量检索 (Vector Only)",
            "bm25": "关键词检索 (BM25 Only)",
        }[x],
        index=0,
    )
    st.session_state.retrieval_strategy = strategy

    st.divider()

    # ==================== 增强检索 (Phase 2) ====================
    st.header("增强检索")
    enhance_mode = st.selectbox(
        "查询预处理:",
        options=["none", "rewrite", "expand"],
        format_func=lambda x: {
            "none": "关闭",
            "rewrite": "查询改写 (Rewrite)",
            "expand": "多查询扩展 (Expand)",
        }[x],
        index=0,
        key="enhance_mode",
    )
    st.caption(
        "Rewrite: 口语→官方术语  |  Expand: 复杂问题拆解"
    )
    enable_reranker = st.checkbox("启用重排序 (Reranker)", value=False, key="enable_reranker")

    st.divider()

    # ==================== 评估看板 ====================
    st.header("评估看板")
    if st.button("运行评估 (对比检索策略)"):
        if not api_key:
            st.error("请先填写 API Key！")
        else:
            docs_count = len(doc_manager.list_documents())
            if docs_count == 0:
                st.warning("请先上传文档！")
            else:
                with st.spinner("正在评估，请稍候..."):
                    from evaluation.test_dataset import TestDataset
                    from evaluation.evaluator import Evaluator

                    generator = Generator(api_key=api_key)
                    evaluator = Evaluator(generator=generator)

                    # 使用内置测试集或自动生成
                    dataset = TestDataset()
                    if len(dataset) == 0:
                        st.info("未找到测试集，请先在 data/eval_dataset.jsonl 中准备测试数据")
                    else:
                        # 对比各策略
                        retrievers = {
                            "vector": create_retriever("vector", doc_manager),
                            "bm25": create_retriever("bm25", doc_manager),
                            "hybrid": create_retriever("hybrid", doc_manager),
                        }

                        ret_results = evaluator.compare_retrieval_strategies(retrievers, dataset)
                        st.session_state["eval_results"] = ret_results

                        try:
                            gen_results = {}
                            for name, ret in retrievers.items():
                                gen_results[name] = evaluator.evaluate_generation(ret, dataset)
                            st.session_state["gen_results"] = gen_results
                        except Exception as e:
                            st.warning(f"生成质量评估未完成（仅展示检索指标）: {e}")

    # 显示评估结果
    if "eval_results" in st.session_state:
        st.markdown("**检索指标对比**")
        results = st.session_state["eval_results"]
        for name, metrics in results.items():
            st.caption(
                f"{name}:  Hit Rate={metrics['hit_rate']:.2%},  "
                f"MRR={metrics['mrr']:.3f},  "
                f"Precision={metrics['precision']:.2%}"
            )

    if "gen_results" in st.session_state:
        st.markdown("**生成质量对比**")
        gen_results = st.session_state["gen_results"]
        for name, metrics in gen_results.items():
            st.caption(
                f"{name}:  Faithfulness={metrics['faithfulness']:.2f},  "
                f"Relevance={metrics['relevance']:.2f},  "
                f"Completeness={metrics['completeness']:.2f}"
            )


# ================= 主聊天区 =================

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

        doc_manager = st.session_state.doc_manager
        docs_list = doc_manager.list_documents()
        if not docs_list:
            full_response = "请先在左侧上传文档并构建知识库！"
            message_placeholder.warning(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.stop()

        try:
            # ============ RAG 检索 ============
            strategy = st.session_state.retrieval_strategy
            retriever = create_retriever(strategy, doc_manager)

            # Phase 2: 增强检索（改写/扩展）
            enhance_mode = st.session_state.enhance_mode
            if enhance_mode != "none":
                generator = Generator(api_key=api_key)
                query_processor = QueryProcessor(generator)
                retrieved_docs = query_processor.retrieve_enhanced(
                    retriever, prompt, mode=enhance_mode,
                    search_k=MULTI_QUERY_SEARCH_K, final_k=FINAL_RETRIEVE_K,
                )
            else:
                retrieved_docs = retriever.retrieve(prompt, k=FINAL_RETRIEVE_K)

            # Phase 2: 重排序
            if st.session_state.enable_reranker and retrieved_docs:
                try:
                    reranker = Reranker()
                    retrieved_docs = reranker.rerank(prompt, retrieved_docs, k=RERANK_TOP_K)
                except Exception as e:
                    st.warning(f"Reranker 加载失败，跳过重排序: {e}")

            # 构建检索上下文（保留来源文件名）
            context_parts = []
            source_refs = []
            for i, doc in enumerate(retrieved_docs):
                filename = doc.metadata.get("filename", "未知来源")
                context_parts.append(f"[{i+1}] {doc.page_content}")
                source_refs.append(f"  [{i+1}] {filename}")

            context = "\n".join(context_parts)
            sources_text = "\n".join(source_refs) if source_refs else ""

            # ============ 调用 LLM ============
            if model_choice == "DeepSeek API (云端)":
                generator = Generator(api_key=api_key)
                sys_prompt, user_prompt = generator.build_qa_prompt(context, prompt)

                full_response = ""
                for chunk in generator.generate_stream(sys_prompt, user_prompt):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "")

                # 末尾追加来源引用
                if sources_text:
                    full_response += f"\n\n---\n**来源:**\n{sources_text}"
                message_placeholder.markdown(full_response)

            else:
                # === 本地 LoRA 模型模式 ===
                with st.spinner("正在加载并调用本地大模型（初次加载可能较慢）..."):
                    tokenizer, model, device = load_local_qwen()
                    import torch

                    sys_prompt_text = f"""你是一个专业助手。请严格基于参考资料回答问题。
                    如果资料不足，请说明"文档中未提及"。

                    参考资料：
                    {context}
                    """

                    messages = [
                        {"role": "system", "content": sys_prompt_text},
                        {"role": "user", "content": prompt},
                    ]

                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    model_inputs = tokenizer([text], return_tensors="pt").to(device)

                    with torch.no_grad():
                        generated_ids = model.generate(
                            **model_inputs,
                            max_new_tokens=512,
                            temperature=0.1,
                        )
                        generated_ids = [
                            output_ids[len(input_ids):]
                            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                        ]

                    final_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

                # 打字机效果
                full_response = ""
                for char in final_response:
                    full_response += char
                    message_placeholder.markdown(full_response + "")
                    time.sleep(0.01)

                if sources_text:
                    full_response += f"\n\n---\n**来源:**\n{sources_text}"
                message_placeholder.markdown(full_response)

            # 存入历史
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"出错: {str(e)}")
