# ============================================================================
# 使用 transformers 库加载离线的 Qwen2.5-VL-7B-VRAG 模型并进行推理：
# 传入一段提示词（prompt），让模型生成回复并打印出来。
#
# 整体流程：
#   1. 导入依赖
#   2. 加载模型权重与处理器（processor）
#   3. 按对话模板构造输入文本，并转换为模型可接受的张量
#   4. 调用 generate() 自回归生成新 token
#   5. 解码生成的 token，打印模型回复
# ============================================================================

# 导入 PyTorch，本文件中主要用到 torch.no_grad（推理时关闭梯度计算以节省显存）
import torch

# 从 transformers 库导入：
#   Qwen2_5_VLForConditionalGeneration —— Qwen2.5-VL 的“条件生成”模型类，
#        “ConditionalGeneration”表示输入（文本/图像等）作为条件，输出由模型生成的文本；
#   AutoProcessor —— 自动处理器，负责把文本/图像等原始输入转换成模型需要的张量，
#        同时也负责把生成的 token id 解码回字符串。
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# 模型在本地磁盘中的目录路径（离线加载，不从 HuggingFace 下载）
model_dir = "/root/.cache/huggingface/hub/Qwen2.5-VL-7B-VRAG/"

# 从本地目录加载预训练模型权重与结构
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir,                    # 模型文件所在目录（含 config.json、权重文件等）
    torch_dtype=torch.float16,    # 以 FP16（半精度）加载权重，显存占用约减半，推理更快
    device_map="auto",            # 由 accelerate 自动把模型各部分分配到可用设备（GPU/CPU）
    trust_remote_code=True,       # 允许执行模型仓库自带的自定义代码（该模型需要）
).eval()                          # 切换到“评估模式”：关闭 dropout 等训练期行为，
                                 #   保证推理结果确定、可复现

# 加载预训练模型配套的处理器（tokenizer + 图像/视频预处理器的组合）
processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

# 构造对话消息：采用 OpenAI 风格的 messages 列表，便于复用官方的对话模板
messages = [
    # role 表示说话角色（"user" / "system" / "assistant"）；
    # content 是一个列表，可同时包含文本、图片等多种模态，这里只放一段纯文本
    {"role": "user", "content": [{"type": "text", "text": "请介绍下你自己"}]}
]

# 套用模型自带的对话模板，把 messages 拼接成模型熟悉的带特殊标记的字符串
#   tokenize=False       —— 只返回字符串，暂不转成 token id（后续交给 processor 统一处理）
#   add_generation_prompt=True —— 在结尾补上“助手开始回答”的提示标记，告诉模型该轮换了
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 用 processor 把文本编码成模型输入张量
#   text=[text]          —— 包成列表，表示一次处理 1 条样本（batch size = 1）
#   return_tensors="pt"  —— 返回 PyTorch 张量（也可选 "tf" / "np"）
#   .to(model.device)    —— 把输入张量搬到模型所在设备，避免出现设备不一致的报错
inputs = processor(text=[text], return_tensors="pt").to(model.device)

# 打印模型生成配置中的 max_length（输入+输出的总长度上限），仅用于观察默认值
print(f"model.generation_config.max_length: {model.generation_config.max_length}")
# 打印 max_new_tokens（只限制“新生成”部分的长度上限），同样仅用于观察默认值
print(f"model.generation_config.max_new_tokens: {model.generation_config.max_new_tokens}")

# 推理阶段不需要反向传播，用 no_grad 关闭梯度计算，可显著降低显存占用并提速
with torch.no_grad():
    # 调用 generate() 进行自回归生成：逐个预测下一个 token，直到满足停止条件
    generated = model.generate(
        **inputs,             # 把 inputs 里的 input_ids、attention_mask 等作为关键字参数传入
        max_new_tokens=512,   # 最多新生成 512 个 token（防止无限生成）
        do_sample=True,       # 启用随机采样（而非贪心搜索），输出更多样
        top_p=0.95,           # 核采样：只在累计概率前 95% 的候选词中采样，兼顾多样性与合理性
        temperature=0.8,      # 温度：<1 使分布更平滑集中，>1 则更随机
    )

# 生成的序列里包含了“输入提示”部分，这里做切片只保留新生成的内容：
#   inputs["input_ids"].shape[1] 即输入序列的长度
#   generated[:, 该长度:]        取第 0 维所有样本、第 1 维从输入长度之后到结尾的部分
out = processor.batch_decode(
    generated[:, inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,  # 解码时跳过 <|endoftext|> 等特殊标记，只保留可读文本
)

# batch 里只有一条样本，取第 0 个即为模型回复，打印到控制台
print(out[0])
