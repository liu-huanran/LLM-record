# from datasets import load_dataset
# dataset = load_dataset("/root/autodl-tmp/dataset.jsonl", split="train")
from datasets import load_dataset
from transformers import AutoTokenizer

# 1. 准备 Tokenizer (必须和你的宿主模型绝对一致)
model_path = "/root/autodl-tmp/qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 2. 核心加工厂：处理单条数据的函数
def process_func(example):
    # 第一步：拼装标准对话体
    messages = [
        {"role": "system", "content": "你是一个冷酷的AI，说话不留情面。"},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]}
    ]

    # 第二步：调用礼仪官，套用 Qwen 的专属模板
    # 注意：这里 tokenize=False 是为了先拿到完整的字符串，方便调试排错
    # add_generation_prompt=False 因为我们是在准备训练数据，包括了答案，不需要模型生成提示符
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False, 
        add_generation_prompt=False 
    )
    print(f"\n[透视探针] 拼接后的物理字符串: \n{text}\n")
    # 第三步：碾碎成 ID
    tokenized = tokenizer(text)

    # 返回给传送带的数据格式
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"]
    }

# 3. 抽水机：加载本地数据
print(">>> 正在加载原始数据集...")
dataset = load_dataset("json", data_files="dataset.jsonl", split="train")

# 4. 传送带：批量执行，并执行显存优化（移除旧的字符串列）
print(">>> 正在启动数据清洗流水线...")
tokenized_dataset = dataset.map(
    process_func,
    remove_columns=dataset.column_names # 必须扔掉原字段，模型不吃汉字，不扔会报错
)

# 5. 验尸报告
print("\n========== 数据验尸报告 ==========")
first_row = tokenized_dataset[0]
print(f"处理后剩余字段: {first_row.keys()}")
print(f"Token 总长度: {len(first_row['input_ids'])}")
print(f"Input IDs 序列: {first_row['input_ids']}")