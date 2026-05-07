import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def test_my_lora():
    print("⏳ 1. 正在加载基础模型 Qwen-1.8B... (轻薄本可能需要两三分钟，请耐心等待)")
    model_id = "Qwen/Qwen-1_8B-Chat"
    
    # 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # 加载底座模型（优先使用 CPU，如果有 GPU 自动用）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map=device, 
        trust_remote_code=True,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16 # CPU 用 float32
    )

    print("🔌 2. 正在插上你的专属记忆 U盘 (LoRA Adapter)...")
    lora_path = "./models/qwen_lora_weights/"
    
    # 这一步是魔法：把底层模型和你的小查性格拼装在一起
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()

    print("\n✅ 融合成功！现在开始面试你的专属小助手（输入 'quit' 退出）：\n")
    
    # 开始对话测试
    history = []
    while True:
        user_input = input("你问: ")
        if user_input.strip().lower() == 'quit':
            break
            
        print("小查思考中...")
        # 组装 openai 格式的 message
        messages = [
            {"role": "system", "content": "你是一个垂直领域知识库助手。"},
            {"role": "user", "content": user_input}
        ]
        
        # 使用 Qwen 的内置模板机制生成输入
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs, 
                max_new_tokens=100
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"小查答: {response}\n")

if __name__ == "__main__":
    test_my_lora()
