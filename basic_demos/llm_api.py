from openai import OpenAI
import os
from dotenv import load_dotenv # 需要 pip install python-dotenv

# 1. 加载本地的 .env 文件
load_dotenv()

# 2. 从环境变量里读取，而不是写死在代码里
api_key = os.getenv("deepseek_api_key")

if not api_key:
    raise ValueError("⚠️ 没找到 API Key！请检查环境变量配置。")

print("✅ 安全获取 Key 成功！(代码里看不到具体字符)")
client = OpenAI(
    api_key=api_key, 
    base_url="https://api.deepseek.com"
)

# 引入 tenacity 库进行重试 (Mock Interview Question 3: 鲁棒性优化)
from tenacity import retry, stop_after_attempt, wait_exponential

# 改进版：增加重试机制 (Retry)
# 遇到错误时自动重试，最多 3 次，每次等待 1s, 2s, 4s...
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_completion(prompt):
    print(f"🤖 模型正在思考: {prompt} ...")
    
    # === 原来的写法 (同步阻塞，无重试) ===
    # try:
    #     response = client.chat.completions.create(
    #         model="deepseek-chat",
    #         messages=[
    #             {"role": "user", "content": prompt}
    #         ],
    #         temperature=0.1 
    #     )
    #     return response.choices[0].message.content
    # except Exception as e:
    #     return f"❌ 出错了: {e}"

    # === 改进后的写法 (Mock Interview Question 2: 流式输出 + 鲁棒性) ===
    # 启用 stream=True，实现打字机效果，大幅降低用户感知的延迟
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        stream=True  # 👈 关键改动：开启流式输出
    )
    
    # 收集流式输出的结果
    full_content = ""
    print("AI 回复: ", end="") # 并不换行，准备接龙
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True) # 实时打印到终端
            full_content += content
            
    print() # 换行
    return full_content 

if __name__ == "__main__":
    try:
        # 测试调用
        result = get_completion("请用一句话解释什么是 '特种兵执行力'？")
        # print(f"\n✅ 最终完整回答:\n{result}") # 不需要再打印一次了，上面已经流式打印了
    except Exception as e:
        print(f"\n❌ 最终失败 (即使重试了3次): {e}")