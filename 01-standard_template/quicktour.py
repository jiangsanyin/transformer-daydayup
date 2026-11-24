# -*- coding: utf-8 -*-
"""
Hugging Face Transformers 快速入门示例。

本文件根据 Hugging Face Quicktour 整理而成，演示以下内容：
1. 登录 Hugging Face Hub；
2. 加载预训练的因果语言模型并生成文本；
3. 使用 Pipeline 完成文本生成、图像分割和语音识别；
4. 使用 Trainer 微调文本分类模型并上传到 Hub。
"""

# 如未安装所需依赖，可在 Notebook 中取消下一行的注释后执行。
# !pip install -U transformers datasets evaluate accelerate timm torch


# ==================== 1. 登录 Hugging Face Hub ====================

# notebook_login() 会在 Notebook 中弹出输入框，用于粘贴 Hugging Face Access Token。
# 公开模型可不登录；需要访问受限模型、下载私有模型或上传模型到 Hub 时需要登录。
from huggingface_hub import notebook_login

notebook_login()


# ==================== 2. 加载预训练模型并生成文本 ====================

# AutoModelForCausalLM：根据模型仓库中的 config.json 自动选择“因果语言模型”类，
# 其适合根据已有文本继续预测下一个词元，例如文本续写或对话生成。
# AutoTokenizer：自动选择与模型匹配的分词器，负责在文本与词元 ID 之间转换。
from transformers import AutoModelForCausalLM, AutoTokenizer

# Qwen2.5-0.5B-Instruct 是公开模型，无需申请 Llama 2 那样的受限仓库访问权限。
# dtype="auto"：按仓库中权重保存的数据类型加载，可减少不必要的类型转换和内存占用。
# device_map="auto"：优先将模型放到可用的加速设备上；显存不足时会自动安排部分权重。
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    dtype="auto",
    device_map="auto",
)

# 分词器会一并下载词表、分词规则和该指令模型使用的对话模板等文件。
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# tokenizer() 将文本转换为 input_ids、attention_mask 等 PyTorch 张量。
# .to(model.device) 将这些输入移到模型所在设备，避免 CPU 与 GPU/MPS 设备不一致。
# 此处保留 Quicktour 的“文本续写”演示；Qwen 是指令模型，实际对话通常应使用 chat template。
model_inputs = tokenizer(
    ["The secret to baking a good cake is "],
    return_tensors="pt",
).to(model.device)

# generate() 根据输入逐步生成词元。max_length 包含输入和生成结果的总词元数。
generated_ids = model.generate(**model_inputs, max_length=30)

# batch_decode() 将生成的词元 ID 还原为文本；索引 [0] 取批次中的第一条结果。
tokenizer.batch_decode(generated_ids)[0]


# ==================== 3. 使用 Pipeline 进行推理 ====================

# Pipeline 将预处理、模型推理和后处理封装在一起，适合快速体验指定任务。
from transformers import pipeline
from accelerate import Accelerator

# Accelerator 自动探测可用设备，例如 CUDA GPU、Apple Silicon 的 MPS 或 CPU。
device = Accelerator().device

# 创建文本生成 Pipeline。此处也使用公开的 Qwen 模型，避免受限 Llama 2 模型的权限问题。
# Pipeline 会从本地缓存读取已经下载的模型文件；若缓存不存在，则会自动下载。
pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    device=device,
)

# 向文本生成 Pipeline 传入提示词，并允许输出总长度最多为 50 个词元的续写结果。
pipe("The secret to baking a good cake is ", max_length=50)


# ==================== 4. 图像分割 Pipeline ====================

# 下面再次导入是为了对应 Quicktour 中每个独立 Notebook 单元格都可单独运行的写法。
from transformers import pipeline
from accelerate import Accelerator

device = Accelerator().device

# image-segmentation 任务会为图像中的不同区域预测类别和分割掩码。
# facebook/detr-resnet-50-panoptic 是用于全景分割的公开预训练模型。
pipeline = pipeline(
    "image-segmentation",
    model="facebook/detr-resnet-50-panoptic",
    device=device,
)

# Pipeline 可以直接接收公开图片的 URL，也可以接收本地图片路径。
segments = pipeline(
    "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"
)

# segments 是分割结果列表；每项包含标签、分数、掩码等信息。
segments[0]["label"]
segments[1]["label"]


# ==================== 5. 自动语音识别 Pipeline ====================

from transformers import pipeline
from accelerate import Accelerator

device = Accelerator().device

# automatic-speech-recognition 将语音转换为文本。
# whisper-large-v3 是公开的 Whisper 语音识别模型，模型体积较大，首次下载需要较长时间。
pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    device=device,
)

# 传入音频 URL 或本地音频路径，返回结果字典中的 "text" 字段即识别出的文本。
pipeline(
    "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac"
)


# ==================== 6. 使用 Trainer 微调文本分类模型 ====================

# AutoModelForSequenceClassification 会加载带有分类头的预训练模型。
# 这里的 DistilBERT 用于二分类情感分析，与前面的 Qwen 文本生成模型是不同任务。
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

# 加载模型、匹配的分词器，以及 Rotten Tomatoes 影评情感分类数据集。
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased"
)
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")


# map() 会将该函数批量应用到数据集的每个批次。
# 函数读出 "text" 列，并将每条文本转换为模型所需的词元 ID 与注意力掩码。
def tokenize_dataset(dataset):
    return tokenizer(dataset["text"])


# batched=True 表示一次传入多条样本，通常比逐条分词更高效。
dataset = dataset.map(tokenize_dataset, batched=True)

# DataCollatorWithPadding 会在每个训练批次内动态补齐序列，
# 避免把整个数据集都填充到全局最长长度而浪费计算资源。
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# TrainingArguments 集中保存训练超参数和输出相关配置。
from transformers import TrainingArguments

training_args = TrainingArguments(
    # 训练产生的 checkpoint、日志等文件保存到该目录。
    output_dir="distilbert-rotten-tomatoes",
    # 优化器每次更新参数时使用的学习率。
    learning_rate=2e-5,
    # 单个设备上每次训练/评估处理的样本数；实际可根据显存大小调整。
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    # 整个训练集被完整遍历的次数。
    num_train_epochs=2,
    # 训练完成后允许将模型上传至当前登录账号的 Hugging Face Hub 仓库。
    push_to_hub=True,
)

# Trainer 将模型、参数、训练数据、评估数据、分词器和组批规则组合为完整训练流程。
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    # 训练集用于更新模型参数。
    train_dataset=dataset["train"],
    # 测试集用于观察模型在未参与训练的数据上的表现。
    eval_dataset=dataset["test"],
    # processing_class 指定文本预处理器；新版 Transformers 推荐使用该参数。
    processing_class=tokenizer,
    # data_collator 负责将不同长度的样本整理为一个可输入模型的批次。
    data_collator=data_collator,
)

# 开始训练。训练时间取决于当前硬件、网络与数据集下载情况。
trainer.train()

# 将训练好的模型权重、配置与分词器文件上传到 Hugging Face Hub。
# 执行此操作需要已登录账号，且会在外部 Hub 上创建或更新模型仓库。
trainer.push_to_hub()
