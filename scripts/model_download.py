# # 架构师版：规避冗余权重的极限 I/O 下载脚本
import os
from modelscope import snapshot_download

# 强行把全局缓存环境变量指过去
os.environ['MODELSCOPE_CACHE'] = '/root/autodl-tmp/modelscope_cache'

print("准备拉取模型，已开启黑名单过滤，拒绝下载冗余的 pth 文件...")

# 架构师视角：坚决拒绝下载冗余的 .pth 文件，只保留加载最快、最安全的 safetensors
model_dir = snapshot_download(
    'LLM-Research/Meta-Llama-3-8B-Instruct', 
    cache_dir='/root/autodl-tmp', 
    revision='master',
    ignore_file_pattern=['*.pth', '*.pt', '*.bin', 'original/*'] # 核心杀招：屏蔽 Meta 原始权重文件
)

print(f"✅ 下载完成！精简版模型安全落盘于: {model_dir}")