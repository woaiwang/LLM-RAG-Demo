from openai import OpenAI

# 🔴 请务必替换为你自己的 DeepSeek API Key
# 格式通常是 "sk-" 开头的一长串字符
api_key = "sk-02315205540a43ec9f0c87241add5d2c" 

client = OpenAI(
    api_key=api_key, 
    base_url="https://api.deepseek.com"  # 这是一个兼容 OpenAI 的接口地址
)

def get_completion(prompt):
    print(f"🤖 模型正在思考: {prompt} ...")
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",  # 这是一个性价比极高的模型
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1 # RAG 场景通常设为 0 或 0.1，让回答更稳定
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ 出错了: {e}"

if __name__ == "__main__":
    # 测试调用
    result = get_completion("请用一句话解释什么是 '特种兵执行力'？")
    print(f"\n✅ 回答:\n{result}")