# Qwen2.5-7B QLoRA Micro-Tuning Pipeline

## 1. 战役目标 (Objective)
在极端算力限制（单卡 RTX 3090 24GB）下，打通 Qwen2.5-7B 模型从 4-bit 量化加载、LoRA 权重注入、SFT 监督微调，到最终持久化与挂载唤醒的完整工业级闭环。

## 2. 物理火力配置 (Hardware & Environment)
* **GPU**: 1x NVIDIA RTX 3090 (24GB VRAM)
* **Base Model**: Qwen2.5-7B-Instruct
* **Core Frameworks**: `transformers`, `peft`, `trl`, `bitsandbytes`

## 3. 核心技术栈 (Core Technologies)
* **4-bit 量化底盘**: 使用 `BitsAndBytesConfig` (nf4 精度 + 双重量化)，将模型静态显存压缩至 ~5GB。
* **旁路矩阵劫持**: 配置 `LoraConfig` ($r=8$, $\alpha=16$)，精准打入 `q_proj` 和 `v_proj`，可训练参数占比仅 0.03%。
* **显存防爆墙**: 强制开启 `gradient_checkpointing_enable()`，以计算时间换取空间。

## 4. 极速复现指南 (Quick Start)

### 依赖安装
```bash
pip install -r requirements.txt
```

## Step 1: 引擎点火 (SFT Training)

执行极速冒烟测试（Smoke Test）：

```Bash
python scripts/train_qwen.py
```

*(注：当前配置采用 Batch Size=4, Gradient Accumulation=4 进行梯度累加模拟。)*

## Step 2: 唤醒与推理 (Inference)

将生成的 LoRA 寄生权重挂载回 4-bit 宿主模型：

```bash
python scripts/inference.py
```

## 5. 战损与排雷账本 (Bug Ledger & Pitfalls)

- **地雷 A**: HuggingFace 框架报 `AttributeError: 'list' object has no attribute 'endswith'`。
  - **根源**: 最新版 `SFTTrainer` 数据预处理期望返回纯字符串以验证 EOS Token。
  - **修复**: 将 `formatting_prompts_func` 的返回值从 `[text]` 剥离为纯 `text`。
- **地雷 B**: Qwen 唤醒时报 `ValueError: add_bos_token = True but bos_token = None`。
  - **根源**: 盲目套用 LLaMA 系的慢速分词器配置，Qwen 底层使用 `<|im_start|>` 协议栈，无传统 BOS token。
  - **修复**: 彻底剥离 `use_fast=False` 和 `add_bos_token=True`，使用 Qwen 原生分词架构。



**环境与基座确认**：执行 `model_download.py` 确保权重在本地。

**数据格式化清洗**：运行 `prepare_data.py`（后续我们将引入真实的开源数据集）。

**配置量化与劫持参数**：在 `train.py` 中写死 `load_in_4bit=True`  与 `LoraConfig`。

**训练与持久化**：运行 `train.py`，必须看到 Loss 下降，且必须在代码末尾执行 `save_model` 落盘。

**隔离唤醒测试**：彻底杀掉训练进程释放显存，运行 `inference.py` 验证战利品的人格是否符合预期。