import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ==========================================
# 0. 基础路径配置
# ==========================================
model_path = "/root/autodl-tmp/qwen/Qwen2.5-7B-Instruct"
data_path = "dataset.jsonl"

print(">>> 1. 正在准备分词器...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

print(">>> 2. 正在启动 4-bit 核武器量化引擎...")
# [你的任务 1]：根据你的手册，完善下面的 BitsAndBytesConfig
compute_dtype = getattr(torch, "float16")
# 量化和LoRA
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
    # 填入计算精度（使用 torch.float16）
    # 填入是否使用双重量化（bnb_4bit_use_double_quant）
)

print(">>> 3. 加载宿主模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quant_config,
    device_map="auto"
)

# 开启梯度检查点（显存救星，绝不能漏）
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

print(">>> 4. 注入 LoRA 寄生虫...")
# [你的任务 2]：填入上一战的 LoRA 配置 (r=8, lora_alpha=16, 目标层 q_proj和v_proj)
peft_config = LoraConfig(
    r=8,                 # 低秩矩阵的维度
    lora_alpha=32,       # 缩放因子 = alpha/r
    target_modules=["q_proj", "v_proj"],  # 需要适配的模块
    lora_dropout=0.05,   # 防止过拟合
    bias="none",         # 偏置项处理方式
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

print(">>> 5. 加载弹药库...")
dataset = load_dataset("json", data_files=data_path, split="train")

def formatting_prompts_func(example):
    # 将 JSONL 中的数据拼装成 Qwen 认识的格式
    messages = [
        {"role": "system", "content": "你是一个冷酷的AI，说话不留情面。"},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return text

print(">>> 6. 配置训练引擎 (TrainingArguments)...")
# [你的任务 3]：严格遵守我的火力限制！
# 学习率设为 2e-4
# per_device_train_batch_size 设为 4
# gradient_accumulation_steps 设为 4
# max_steps 设为 50 (我们只做快速验证)
# logging_steps 设为 10
# output_dir 设为 "./qwen_output"
training_args = TrainingArguments(
    # 填入上述严格限制的参数...
    output_dir = "./qwen_output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    max_steps=50,
    learning_rate=2e-4,
    logging_steps=10,
    optim="paged_adamw_8bit",
    save_strategy="no", # 测试阶段不保存庞大的 checkpoint
    report_to="none"    # 关掉第三方监控面板的烦人提示
)

print(">>> 7. 终极点火...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    # peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    # max_seq_length=512, # 限制最大长度，防止显存爆炸
)

# 开始训练！
trainer.train()