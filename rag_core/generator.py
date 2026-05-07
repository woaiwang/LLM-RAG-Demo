"""
LLM 生成器封装

职责:
  - 封装 DeepSeek API 调用
  - 支持流式/非流式输出
  - 自动重试（复用 tenacity）
  - 统一的 Prompt 模板管理
"""

from typing import Optional, Generator
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_TEMPERATURE, LLM_MAX_TOKENS


class Generator:
    """LLM 生成器，封装 DeepSeek API。"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or DEEPSEEK_API_KEY
        if not self.api_key:
            raise ValueError("⚠️ 未配置 DeepSeek API Key")
        self.client = OpenAI(api_key=self.api_key, base_url=DEEPSEEK_BASE_URL)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_stream(self, system_prompt: str, user_prompt: str) -> Generator[str, None, None]:
        """流式生成，逐 chunk 产出文本片段。"""
        stream = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """非流式生成，直接返回完整文本。"""
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            stream=False,
        )
        return response.choices[0].message.content

    # ========== Prompt 模板 ==========

    @staticmethod
    def build_qa_prompt(context: str, query: str) -> tuple:
        """
        构建 RAG 问答用的 system/user prompt。

        返回: (system_prompt, user_prompt)
        """
        system = f"""你是一个专业的政策问答助手。请严格基于以下【参考资料】回答问题。

        规则：
        1. 如果参考资料足够，给出准确、简洁的回答
        2. 如果参考资料不足，明确说明"文档中未提及"
        3. 不要编造信息
        4. 在回答末尾用 [来源: 文件名] 的形式标注引用

        【参考资料】：
        {context}
        """
        return system, query

    # ========== Query Rewriting Prompt ==========

    @staticmethod
    def build_rewrite_prompt(query: str) -> tuple:
        """
        构建 Query Rewriting prompt。
        将用户口语查询改写为更适合文档检索的官方表述。

        返回: (system_prompt, user_prompt)
        """
        system = """你是一个政策文档检索优化专家。
    你的任务是将用户的自然语言问题改写为更适合文档检索的官方表述。

    要求：
    1. 使用政策文件中可能出现的正式术语（如"保研"→"推荐优秀应届本科毕业生免试攻读"）
    2. 保留原问题的核心意图
    3. 输出 2-3 条改写版本（含原始问题）
    4. 每行一条，以数字开头

    示例：
    用户：保研要什么条件？
    输出：
    1. 保研要什么条件？
    2. 推荐优秀应届本科毕业生免试攻读硕士学位研究生基本条件
    3. 推免资格申请要求与选拔标准"""

        return system, query

    # ========== Multi-Query Expansion Prompt ==========

    @staticmethod
    def build_expand_prompt(query: str) -> tuple:
        """
        构建 Multi-Query Expansion prompt。
        将复杂问题拆解为多个独立子问题，覆盖不同维度。

        返回: (system_prompt, user_prompt)
        """
        system = """你是一个查询分解专家。
    将用户的问题拆解为 3-5 个独立的子问题，每个子问题聚焦一个具体维度。

    要求：
    1. 子问题应覆盖原问题的不同方面，互不重叠
    2. 每个子问题独立可检索，即可以单独用于搜索
    3. 输出每行一条，以数字开头

    示例：
    用户：保研需要满足什么条件？
    输出：
    1. GPA最低要求是多少？
    2. 科研论文或竞赛获奖的加分政策是什么？
    3. 英语六级或四级有硬性要求吗？
    4. 德育考核和综合素质评价标准是什么？"""

        return system, query

    @staticmethod
    def build_judge_prompt(question: str, context: str, answer: str) -> tuple:
        """
        构建 LLM-as-Judge 评估 prompt。

        评估维度：
          - Faithfulness: 回答是否忠实于检索片段
          - Relevance: 回答是否相关
          - Completeness: 回答是否完整

        返回: (system_prompt, user_prompt)
        """
        system = """你是一个 RAG 系统输出质量评审员。请从以下三个维度评分 (1-5分)：
        - Faithfulness (忠实度): 回答是否完全基于参考资料，没有编造
        - Relevance (相关性): 回答是否切题
        - Completeness (完整性): 回答是否完整覆盖了问题

        请以 JSON 格式输出：
        {"faithfulness": N, "relevance": N, "completeness": N, "reason": "简短理由"}
        """

        user = f"""【问题】
        {question}

        【参考资料】
        {context}

        【模型回答】
        {answer}

        请给出评分 (JSON 格式):"""
        return system, user
