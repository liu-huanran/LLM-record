import torch
import time
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments
)
from peft import PeftModel
device = "cuda"

compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
    # 填入计算精度（使用 torch.float16）
    # 填入是否使用双重量化（bnb_4bit_use_double_quant）
)

model_path = "/root/autodl-tmp/qwen/Qwen2.5-7B-Instruct"
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quant_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

ft_model = PeftModel.from_pretrained(
    base_model,
    "/root/autodl-tmp/qwen_lora_test",
)

prompt = "你是谁？"
messages = [
    {"role": "system", "content": "你是一个冷酷的AI，说话不留情面。"},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)


generated_ids = ft_model.generate(
    **model_inputs,
    max_new_tokens=512
)


generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


print("\n========== 模型输出 ==========\n")
print(response)