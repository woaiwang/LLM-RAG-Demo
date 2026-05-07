# 垂直领域智能知识库 (RAG) 助手

> 基于 LangChain + 多策略检索 + Query Rewriting + Multi-Query + Cross-Encoder Reranker 的智能问答系统，解决通用大模型在特定领域（如企业文档、校园教务、政策法规）存在的幻觉与时效性问题。

---

## 目录

- [项目背景](#项目背景)
- [核心架构](#核心架构)
- [RAG 管道详解](#rag-管道详解)
- [查询预处理 (Query Processing)](#查询预处理-query-processing)
- [检索策略原理](#检索策略原理)
- [重排序 (Reranker)](#重排序-reranker)
- [评估系统](#评估系统)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [使用指南](#使用指南)
- [技术栈](#技术栈)
- [后续规划](#后续规划)

---

## 项目背景

### 要解决什么问题？

通用大模型（如 GPT、DeepSeek）虽然知识渊博，但存在两个致命缺陷：

1. **私有数据盲区** — 模型训练时没见过你的内部文档、校规校纪、公司制度
2. **时效性滞后** — 政策每年都在变，模型知识停留在训练截止日期

**RAG（检索增强生成）** 的思路是：不指望模型背下所有知识，而是每次回答问题前去知识库中检索相关片段，让模型基于检索结果生成答案。

### 与微调的区别

| 方案 | 优点 | 缺点 |
|---|---|---|
| **RAG（本项目）** | 无需训练，知识实时更新，可追溯来源 | 依赖检索质量 |
| **LoRA 微调** | 让模型"记住"特定知识 | 需训练数据，更新成本高，有灾难性遗忘风险 |

本项目以 **RAG 为主**，同时保留 **LoRA 微调接口**，支持双模式切换。

---

## 核心架构

系统采用**三层递进式**架构设计：

```
┌──────────────────────────────────────────────────────────────────┐
│                        Streamlit Web UI                          │
│    (多文档管理 / 聊天界面 / 策略选择 / 增强检索开关 / 评估看板)    │
└──────────────────────┬───────────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────────┐
│                       RAG Core 引擎                               │
│                                                                   │
│  ┌──────────┐  ┌──────────┐  ┌────────────────┐  ┌────────────┐ │
│  │ 文档管理   │  │  切片器   │  │  查询预处理      │  │   检索器    │ │
│  │ SQLite    │→│ Chunker  │  │  QueryProcessor │→│  Retriever  │ │
│  │ + Chroma  │  │          │  │  ┌─ Rewrite    │  │  3种策略    │ │
│  └──────────┘  └──────────┘  │  ├─ Expand      │  └─────┬──────┘ │
│                               │  └─ RRF Merge   │        │        │
│                               └────────────────┘        ▼        │
│                                                    ┌────────────┐ │
│                                                    │  重排序     │ │
│                                                    │  Reranker   │ │
│                                                    │ Cross-Enc. │ │
│                                                    └──────┬─────┘ │
│                                                           │        │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │                     生成器 Generator                      │      │
│  │              DeepSeek API / 本地 Qwen + LoRA             │      │
│  └─────────────────────────────────────────────────────────┘      │
└──────────────────────┬───────────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────────┐
│                     Evaluation 评估引擎                            │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐             │
│  │ 测试集管理  │  │ 检索指标   │  │ LLM-as-Judge       │             │
│  │ JSONL    │→│ Hit Rate │→│ Faithfulness       │             │
│  │          │  │ MRR      │  │ Relevance          │             │
│  │          │  │ P@K      │  │ Completeness       │             │
│  └──────────┘  └──────────┘  └────────────────────┘             │
└──────────────────────────────────────────────────────────────────┘
```

### 数据流：一次完整的 RAG 问答

```
用户提问："保研要啥条件？"
       │
       ▼
┌─────────────────────────────────────────────────────┐
│ ① 查询预处理 (可选，侧边栏开关控制)                  │
│                                                      │
│  ├─ Rewrite 模式：                                    │
│  │   "保研要啥条件" → ["保研要啥条件",                │
│  │                     "推荐优秀应届本科毕业生...",   │
│  │                     "推免资格申请要求与选拔标准"]  │
│  │                                                   │
│  ├─ Expand 模式：                                    │
│  │   "奖学金怎么申请" → ["奖学金怎么申请",           │
│  │                       "GPA最低要求是多少？",      │
│  │                       "申请流程和截止日期",       │
│  │                       "需要提交哪些材料"]         │
│  │                                                   │
│  └─ 多个查询分别检索 → RRF 融合去重                  │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ ② 检索阶段 (按所选策略)                      │
│  ├─ Vector: 语义相似度检索 (Top-10)          │
│  ├─ BM25:   关键词命中检索 (Top-10)          │
│  └─ Hybrid: RRF 融合两种结果 (Top-10)        │
└──────────────────────┬───────────────────────┘
                       │ Top-10 候选文档
                       ▼
┌──────────────────────────────────────────────┐
│ ③ 重排序 (可选，侧边栏开关控制)              │
│  ├─ Cross-Encoder 深度交互计算               │
│  ├─ 对每对 (query, doc) 重新打分            │
│  └─ 按相关性降序取 Top-3                     │
└──────────────────────┬───────────────────────┘
                       │ Top-3 精排结果
                       ▼
┌──────────────────────────────────────────────┐
│ ④ 增强阶段 (Augment)                         │
│ 将检索结果 + 用户问题                         │
│ 组装成 System Prompt                         │
└──────────────────────┬───────────────────────┘
                       ▼
┌──────────────────────────────────────────────┐
│ ⑤ 生成阶段 (Generate)                        │
│ DeepSeek API / 本地 Qwen                     │
│ 基于检索片段生成回答 + 标注来源               │
└──────────────────────────────────────────────┘
```

### 关键设计决策

- **单 Chroma Collection + Metadata 过滤**：所有文档共享一个向量集合，每个 Chunk 的 metadata 中标记 `doc_id`。检索时全集合搜索，删除时按 `doc_id` 过滤删除。避免了多 Collection 管理的复杂度。
- **SQLite 文档元数据**：记录文档 ID、文件名、上传时间、Chunk 数量、状态。轻量零配置，无需额外数据库服务。
- **策略工厂模式**：新增检索策略只需继承 `Retriever` 基类并实现 `retrieve()`，即可无缝接入系统。
- **全局模型缓存**：Reranker 模型通过模块级全局变量缓存，避免 Streamlit 重渲染时反复加载模型。

---

## RAG 管道详解

### 1. 文档切片 (Chunking)

文档不能整篇喂给大模型（上下文窗口有限，且长文本检索精度差），需要切成小块。

```python
# 当前默认策略：递归字符分割
RecursiveCharacterTextSplitter(
    chunk_size=300,    # 每块约300个字符
    chunk_overlap=50   # 块间重叠50字符，防止切断关键句
)
```

**为什么需要重叠？** 如果一段话刚好在200字处被切断，下半段会丢失前文语境。重叠保证了切分边界附近的语义完整性。

### 2. 向量化 (Embedding)

将文本片段转化为固定维度的向量（本项目使用 512 维），语义相近的文本在向量空间中距离更近。

**BAAI/bge-small-zh-v1.5**：智源研究院开发的中文 Embedding 小模型，兼顾效果与速度。输出经过 L2 归一化，使得**内积等价于余弦相似度**，大幅提升计算效率。

### 3. 查询预处理 + 检索 + 重排序

详见后续章节。

### 4. 生成 (Generation)

将检索到的片段拼接成 System Prompt 的一部分：

```
你是一个专业的政策问答助手。请严格基于以下【参考资料】回答问题。
如果参考资料不足，请说"文档中未提及"。
在回答末尾标注来源。

【参考资料】：
[1] 根据《XX大学推荐优秀应届本科毕业生免试攻读...
[2] 申请推免的学生应同时满足以下基本条件：...

【用户问题】：
保研要啥条件？
```

调用 LLM（DeepSeek API 或本地 Qwen），流式输出回答。回答末尾自动追加检索来源标注。

---

## 查询预处理 (Query Processing)

### 解决什么问题？

用户不会用"官方术语"提问。比如用户问"保研要啥条件"，但文档写的是"推荐优秀应届本科毕业生免试攻读硕士学位研究生基本条件"。两者用词完全不重叠，向量检索和 BM25 都难以命中。

查询预处理模块通过调用 LLM 将用户问题改写/扩展为更易检索的形式。

### 模式一：Query Rewriting (查询改写)

**原理**：让 LLM 将口语化查询改写为官方术语表述。

```
输入: "保研要啥条件"
输出:
  1. 保研要啥条件？
  2. 推荐优秀应届本科毕业生免试攻读硕士学位研究生基本条件
  3. 推免资格申请要求与选拔标准
```

**Prompt 设计**：

```
你是一个政策文档检索优化专家。将用户的自然语言问题改写为2-3个
更适合文档检索的官方表述。要求：
1. 使用政策文件中可能出现的正式术语
2. 保留原问题的核心意图
3. 每行一条，以数字开头
```

**实现**：`QueryProcessor.rewrite()` → 调用 LLM → 解析编号列表 → 返回 `[原始查询, 改写1, 改写2]`

### 模式二：Multi-Query Expansion (多查询扩展)

**原理**：将复杂问题拆解为多个独立的子问题，覆盖不同维度。

```
输入: "保研需要满足什么条件？"
输出:
  1. GPA最低要求是多少？
  2. 科研论文或竞赛获奖的加分政策是什么？
  3. 英语六级或四级有硬性要求吗？
  4. 德育考核和综合素质评价标准是什么？
```

**Prompt 设计**：

```
你是一个查询分解专家。将用户的问题拆解为3-5个独立的子问题，
每个子问题聚焦一个具体维度。要求：
1. 子问题应覆盖原问题的不同方面，互不重叠
2. 每个子问题独立可检索
3. 每行一条，以数字开头
```

**实现**：`QueryProcessor.expand()` → 调用 LLM → 解析编号列表 → 返回 `[原始查询, 子问题1, 子问题2, ...]`

### RRF 融合 (多路检索结果合并)

多个改写/子问题分别检索后，通过 RRF（Reciprocal Rank Fusion）将多路结果融合去重：

```python
def rrf_merge(query_results, k=FINAL_RETRIEVE_K):
    # 每路检索结果都有自己的排序
    # RRF 通过排名而非分数来融合
    # 避免不同检索器分数尺度不一致的问题
    for query, docs in query_results.items():
        for rank, doc in enumerate(docs):
            fused_score += 1 / (RRF_K + rank + 1)
    # 按融合分数排序取 Top-K
```

### 限制

- Rewrite 和 Expand 互斥，不能同时启用（避免延迟过高）
- 每次改写/扩展会额外调用 1-2 次 LLM API
- 改写最多保留 4 条查询，扩展最多 6 条

---

## 检索策略原理

这是本系统的核心亮点，实现了三种可切换的检索策略：

### 策略一：向量检索 (Vector Only)

**原理**：将用户问题和文档片段都映射到同一个向量空间，通过余弦相似度找到最相似的片段。

```
"保研需要什么条件？"  →  Embedding →  [0.23, -0.56, 0.89, ...]
                                        ↓ 点积比较
"推免基本条件如下"    →  Embedding →  [0.21, -0.51, 0.92, ...]
```

**优点**：理解语义，同义词匹配（"保研"≈"推免"）
**缺点**：对专有名词和精确数字的匹配不够敏感

### 策略二：BM25 关键词检索 (BM25 Only)

**原理**：基于词频（TF）和逆文档频率（IDF）的经典信息检索算法，专为精确关键词匹配设计。

```
Score(文档, 查询) = Σ IDF(t) × TF(t, 文档) × (k₁ + 1) / (TF + k₁ × (1 - b + b × |文档|/avg|文档|))
```

针对政策文档做中文分词（jieba），确保"推荐优秀应届本科毕业生免试攻读硕士学位研究生"这种长专有名词能被正确索引。

**优点**：精确命中专有术语和数字（"GPA 3.5"、"截止日期 2025年9月"）
**缺点**：不理解语义近义词

### 策略三：混合检索 Hybrid (RRF 融合) ⭐ **推荐**

**核心算法：Reciprocal Rank Fusion (RRF)**

同时运行向量检索和 BM25 检索，将两个排序结果融合：

```
score(d) = 1 / (60 + rank_vector(d)) + 1 / (60 + rank_bm25(d))
```

**为什么用 RRF 而不是加权平均？**
- 向量的分数和 BM25 的分数不在同一个量纲上，直接加权平均没有意义
- RRF 只依赖**排名**而非原始分数，天然对分数尺度不敏感
- 常数 k=60 是业界经验值（RRF 论文推荐值），可以有效平滑排名噪声

**举例说明**：

| 文档 | 向量排名 | BM25排名 | RRF 得分 | 最终排名 |
|------|---------|---------|---------|---------|
| 文档A | 1 | 3 | 1/61 + 1/63 = 0.0325 | 第1 |
| 文档B | 5 | 1 | 1/65 + 1/61 = 0.0318 | 第2 |
| 文档C | 2 | 20 | 1/62 + 1/80 = 0.0286 | 第3 |

两个排序都不错的文档会获得更高的融合分数，而只在单一策略中排名高的文档会被适当压低。

---

## 重排序 (Reranker)

### 解决什么问题？

检索出的 Top-10 中经常混入"语义相似但实际不相关"的内容。这是因为双编码器（Bi-Encoder）将 query 和 doc 分别编码后在向量空间比较，两者的交互非常浅层。

### Cross-Encoder 原理

Cross-Encoder 将 query 和 doc **拼接在一起**输入同一个 Transformer，做深度交互计算：

```
Bi-Encoder (向量检索):
  query → encoder → 向量q
  doc   → encoder → 向量d
  score = cos(向量q, 向量d)

Cross-Encoder (重排序):
  [CLS] query [SEP] doc [SEP] → encoder → score
```

**关键区别**：Cross-Encoder 的 attention 可以在 query 和 doc 的每个 token 之间自由交互，
因此能捕捉到"这个文档片段确实回答了用户问题的某个方面"这类深层关系，
而 Bi-Encoder 只能做粗粒度的语义匹配。

### 模型选择

使用 **BAAI/bge-reranker-base**（智源研究院），通过 `sentence_transformers.CrossEncoder` 加载。

**为什么不使用 FlagEmbedding？**
- `FlagEmbedding` 与较新版本的 `transformers`（>= 5.0）存在兼容性问题
- `sentence_transformers` 内置 `CrossEncoder`，接口更稳定，底层使用同样的 BGE-Reranker 模型

### 实现细节

```python
# 延迟加载 + 全局缓存
_reranker_model = None

def _load_reranker_model(model_name):
    global _reranker_model
    if _reranker_model is None:
        from sentence_transformers import CrossEncoder
        _reranker_model = CrossEncoder(model_name)
    return _reranker_model

# 重排序核心逻辑
pairs = [[query, doc.page_content] for doc in documents]
scores = model.predict(pairs, show_progress_bar=False)
# 按分数降序排列，取 Top-K
```

### 容错设计

- 模型加载失败 → 返回原始文档顺序（截取 Top-K）
- 文档数不足 k 时 → 直接返回
- 任意运行时异常 → 降级为原始排序

### 与查询预处理的关系

两者可以独立开关、组合使用：

```
Rewrite/Expand 负责"多找"（提高召回率）
     ↓
Reranker 负责"精选"（提高精确率）
```

| 组合 | 效果 |
|------|------|
| 仅 Rewrite | 关键词命中率提升，但仍有噪声 |
| 仅 Reranker | 排序更准，但可能漏掉相关文档 |
| Rewrite + Reranker | 召回率和精确率同时提升 |

---

## 评估系统

### 为什么要做评估？

没有量化指标，任何优化都是"我觉得变好了"。评估体系让你用数据说话：

### 检索指标

| 指标 | 公式 | 含义 |
|---|---|---|
| **Hit Rate@K** | 相关文档是否出现在 Top-K 中 | 召回率，越高越好 |
| **MRR** | 1 / 首个相关文档的排名，对所有查询取平均 | 首个命中排名的质量 |
| **Precision@K** | Top-K 中相关文档的比例 | 精确率，越高越准 |

### 生成质量指标 (LLM-as-Judge)

用大模型来评判大模型的输出质量，按 1-5 分打分：

| 维度 | 含义 | 评分标准 |
|---|---|---|
| **Faithfulness (忠实度)** | 回答是否完全基于检索片段，不编造 | 5=完全忠实，1=完全在编 |
| **Relevance (相关性)** | 回答是否回答了用户的问题 | 5=完全切题，1=答非所问 |
| **Completeness (完整性)** | 回答是否完整覆盖问题所需信息 | 5=完整无遗漏，1=严重遗漏 |

### 使用流程

```
准备测试集 (data/eval_dataset.jsonl) →
点击"运行评估" →
系统自动执行:
  ├─ 对每个问题，用三种策略分别检索
  ├─ 计算 Hit Rate / MRR / Precision
  ├─ 生成回答，LLM-as-Judge 打分
  └─ 展示对比结果
```

---

## 项目结构

```
LLM-RAG-Internship/
│
├── app.py                      # Streamlit Web 入口 (主程序)
│                               #   多文档管理 + 聊天 + 评估看板
│                               #   Phase 2: 增强检索开关 (侧边栏)
│
├── config.py                   # 统一配置管理
│                               #   API Key、模型路径、检索参数
│                               #   Phase 2: Reranker/Multi-Query 参数
│
├── rag_core/                   # RAG 核心引擎
│   ├── __init__.py
│   ├── document_manager.py     # 多文档管理器
│   │                           #   - SQLite 文档元数据
│   │                           #   - ChromaDB 向量索引 (单Collection)
│   │                           #   - 上传/删除/列表
│   ├── chunker.py              # 文本切片策略
│   │                           #   - BaseChunker 抽象基类
│   │                           #   - RecursiveChunker (默认)
│   ├── retrievers.py           # 多策略检索器
│   │                           #   - VectorRetriever (ChromaDB)
│   │                           #   - BM25Retriever (jieba + rank_bm25)
│   │                           #   - HybridRetriever (RRF 融合)
│   │                           #   - create_retriever 工厂函数
│   │                           #   - Phase 2: retrieve_and_rerank()
│   ├── query_processor.py      # Phase 2: 查询预处理
│   │                           #   - QueryProcessor.rewrite()
│   │                           #   - QueryProcessor.expand()
│   │                           #   - QueryProcessor.retrieve_enhanced()
│   │                           #   - rrf_merge() 多路融合
│   ├── reranker.py             # Phase 2: Cross-Encoder 重排序
│   │                           #   - Reranker 类
│   │                           #   - 懒加载 + 全局缓存
│   │                           #   - 失败降级
│   ├── generator.py            # LLM 生成器
│   │                           #   - 流式/非流式输出
│   │                           #   - Tenacity 自动重试 (3次)
│   │                           #   - Prompt 模板管理
│   │                           #   - Phase 2: rewrite/expand/judge prompt
│   ├── pdf_rag.py              # (保留) 基础 RAG 实现
│   └── advanced_rag.py         # (保留) 进阶 RAG + Rerank
│
├── evaluation/                 # 评估引擎
│   ├── __init__.py
│   ├── test_dataset.py         # 测试集管理 (JSONL)
│   ├── metrics.py              # 检索指标
│   │                           #   - Hit Rate@K
│   │                           #   - MRR
│   │                           #   - Precision@K
│   └── evaluator.py            # 评估器
│                               #   - LLM-as-Judge 打分
│                               #   - 多策略对比
│
├── basic_demos/                # (保留) 基础原理演示
│   ├── embedding_test.py       #   Embedding + 余弦相似度手写实现
│   ├── llm_api.py              #   API 调用 + 流式输出 + 重试
│   └── simple_rag.py           #   最简 RAG 原型 (< 80行)
│
├── data/
│   ├── data.pdf                # 测试用 PDF
│   ├── pdfs/                   # 上传的 PDF 文件 (持久化存储)
│   ├── eval_dataset.jsonl      # 评估测试集
│   └── finetune_data.jsonl     # LoRA 微调训练数据
│
├── chroma_db/                  # 向量数据库持久化目录
│   └── documents.db            # 文档元数据 SQLite
│
├── models/                     # 本地模型权重
│   └── qwen_lora_weights/      # Qwen LoRA 微调权重
│
├── .env                        # 环境变量 (API Key 等)
├── .env.example                # 环境变量模板
├── requirements.txt            # Python 依赖
└── README.md                   # 本文件
```

---

## 快速开始

### 1. 环境准备

```bash
# 使用 conda 创建环境 (推荐)
conda create -n ultrarag python=3.11
conda activate ultrarag

# 安装依赖
pip install -r requirements.txt

# (可选) 安装额外依赖
pip install rank_bm25 jieba scikit-learn plotly
```

### 2. 配置 API Key

```bash
# 复制模板
cp .env.example .env

# 编辑 .env 文件，填入你的 DeepSeek API Key
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 3. 启动应用

```bash
# 标准启动
streamlit run app.py

# 如果遇到文件监视器报错，加 --server.fileWatcherType none
streamlit run app.py --server.fileWatcherType none
```

启动成功后浏览器自动打开 `http://localhost:8501`。

### 4. (可选) 首次启动自动下载模型

首次使用 Reranker 时，会自动下载 BGE-Reranker 模型（约 1.1GB）：

```
Downloading BAAI/bge-reranker-base...
```

---

## 使用指南

### 第一步：上传文档

1. 在左侧侧边栏，点击"上传 PDF 知识库"（支持多选）
2. 等待系统自动完成：PDF 解析 → 切片 → 向量化 → 索引入库
3. 上传成功后，文档会显示在"已上传文档"列表中
4. 可随时删除文档，系统会自动清理向量索引

### 第二步：选择检索策略

在侧边栏"检索策略"下拉框中选择：

| 策略 | 适用场景 |
|---|---|
| **Hybrid (推荐)** | 通用场景，兼顾语义和关键词匹配 |
| **Vector Only** | 需要理解语义同义词时 |
| **BM25 Only** | 精确关键词查询（如政策编号） |

### 第三步：配置增强检索 (Phase 2)

侧边栏新增"增强检索"区块：

| 选项 | 说明 |
|---|---|
| **查询预处理** | 关闭 / 查询改写 (Rewrite) / 多查询扩展 (Expand) |
| **启用重排序 (Reranker)** | 勾选后对检索结果做 Cross-Encoder 精排 |

**推荐组合**：

- **日常使用**：Hybrid + Rewrite + Reranker（质量最高，延迟略高）
- **快速问答**：Hybrid + 关闭增强检索（最低延迟）
- **复杂问题**：Expand + Reranker（覆盖面最广）

### 第四步：开始问答

在聊天输入框输入问题，系统会：
1. 按配置执行查询预处理（Rewrite/Expand）
2. 按所选策略检索所有已上传文档
3. 可选：Reranker 对检索结果重排序
4. 将精排后的 Top-3 片段作为参考上下文
5. 调用 LLM 生成回答
6. 在回答末尾标注来源文件名

### 第五步：运行评估

1. 准备测试集 `data/eval_dataset.jsonl`，格式如下：
   ```json
   {"question": "保研GPA要求是多少？", "ground_truth": "3.5", "source_doc": "保研政策2025.pdf"}
   {"question": "奖学金申请条件", "ground_truth": "GPA≥3.0", "source_doc": "奖学金管理办法.pdf"}
   ```
2. 点击侧边栏"运行评估"按钮
3. 系统自动对比三种策略的检索指标和生成质量

---

## 技术栈

| 层面 | 技术 | 用途 |
|---|---|---|
| **大语言模型** | DeepSeek V3 (API) | 云端 LLM 推理 |
| | Qwen-1.8B + LoRA | 本地模型生成 |
| **向量检索** | ChromaDB | 向量数据库 |
| | BAAI/bge-small-zh-v1.5 | 中文 Embedding 模型 (512维) |
| **关键词检索** | BM25 (rank_bm25) + jieba | 稀疏检索 |
| **排序融合** | Reciprocal Rank Fusion (RRF) | 混合检索融合 |
| **重排序** | BAAI/bge-reranker-base (Cross-Encoder) | 二次精排 |
| **查询预处理** | DeepSeek LLM (Rewrite/Expand Prompt) | 口语→官方术语 / 问题拆解 |
| **开发框架** | LangChain, PyTorch | 切片、向量化、模型加载 |
| **Web 界面** | Streamlit | 交互式 UI |
| **评估** | LLM-as-Judge | 自动质量评分 (Faithfulness/Relevance/Completeness) |
| **数据处理** | PyPDF, jieba, scikit-learn | PDF 解析、分词、指标计算 |

---

## 后续规划

### Phase 3 — 知识图谱 (GraphRAG)

- [ ] 从文档中抽取实体和关系，构建轻量知识图谱
- [ ] 向量检索 + 图遍历双通道融合
- [ ] 支持"政策变化对比"类关系推理问题

### Phase 4 — 生产化增强

- [ ] 对话记忆/多轮上下文管理
- [ ] 流式 Reranker（减少首 token 延迟）
- [ ] Self-RAG / Corrective RAG：让 LLM 判断检索结果质量，决定是否需要重新检索
- [ ] 异步并行检索（Rewrite/Expand 多路同时检索）

---

**作者**: 查志渊
