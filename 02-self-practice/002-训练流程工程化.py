"""
训练流程工程化示例（在 001 基础上补齐验证、checkpoint 与日志）：

0. 导入必要的库和模块
1. 集中定义超参数（一个字典管全部，方便记录和复现）
2. 固定随机种子，保证实验可复现
3. 配置日志：同时输出到控制台和文件，且落盘超参数
4. 数据准备：从训练集中切分出一份验证集（train / val / test 三分离）
5. 定义神经网络模型
6. 把「训练一个 epoch」和「评估」抽成独立函数，训练/验证/测试复用同一套逻辑
7. 训练循环：每个 epoch 训练 + 验证
   - 保存 last checkpoint（含 epoch / optimizer / 历史最佳指标，可随时断点续训）
   - 验证指标变好时，另存一份 best checkpoint
8. 训练结束后：加载 best checkpoint，在测试集上评估
9. 模型加载与使用

与 001 的关键差异：
- 001 只有 train + test，没有验证集，训练过程中看不到泛化能力；
  本文件切出 val，每个 epoch 都能观察 train/val 的差距（判断过拟合）。
- 001 只在训练全部结束后存一次权重，中途断掉只能从头再来；
  本文件每个 epoch 都存 last，且包含 optimizer 状态，支持续训。
- 001 用 print，信息只留在终端里；
  本文件用 logging，同时写文件，附带时间戳，方便事后对比多次实验。
"""

import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader, random_split

# ---------------------------------------------------------------------------
# 1. 超参数集中管理
# ---------------------------------------------------------------------------
CONFIG = {
    "batch_size": 64,
    "learning_rate": 1e-3,
    "num_epochs": 3,
    "dropout_rate": 0.1,
    "val_ratio": 0.1,          # 从 6 万张训练图中按比例切出验证集
    "seed": 42,                # 随机种子，保证数据划分、dropout、shuffle 可复现
    "log_interval": 100,       # 每隔多少个 batch 打印一次训练进度
    "data_dir": "./data",
    "output_dir": "./output",
    "resume": False,           # 改成 True 即可从 last checkpoint 断点续训
}

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def set_seed(seed: int) -> None:
    """固定所有能固定的随机源，让实验结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# 3. 日志配置
# ---------------------------------------------------------------------------
def setup_logger(log_dir: str, config: dict) -> logging.Logger:
    """配置同时输出到控制台和文件的 logger。

    注意点：
    - 用 logging 而不是 print：可以带时间戳和日志级别，且能一条命令开关全部输出；
    - 每次都新建 logger 的话，重复运行会累积 handler 导致日志翻倍，所以先清空；
    - FileHandler 要显式指定 encoding="utf-8"，否则中文在部分平台上会乱码。
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # 防止重复添加 handler 造成日志重复
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(
        os.path.join(log_dir, "train.log"), mode="a", encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    # 把本次实验的超参数落盘，事后回溯时才知道每个结果对应什么配置
    logger.info("=" * 60)
    logger.info("实验配置: %s", config)
    logger.info("使用设备: %s", DEVICE)

    return logger


# ---------------------------------------------------------------------------
# 5. 模型定义
# ---------------------------------------------------------------------------
class SimpleNN(nn.Module):
    def __init__(self, dropout_rate: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=32 * 7 * 7, out_features=128)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # 展平为一维向量
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# 6. 训练（train） / 评估（eval/test）函数
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, loss_fn, optimizer, epoch, num_epochs, logger, log_interval):
    """训练一个 epoch，返回整个 epoch 的平均 loss 和平均准确率。

    指标要按样本数加权累加，不能直接对 batch 指标求平均：
    最后一个 batch 往往不满，权重和前面不一样。
    """
    model.train()  # 训练模式：开启 dropout

    loss_sum = 0.0
    correct_sum = 0
    sample_count = 0

    for i, (images, labels) in enumerate(loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()          # 清空梯度
        logits = model(images)         # 前向传播
        loss = loss_fn(logits, labels)  # 计算损失

        loss.backward()                # 反向传播，计算梯度
        optimizer.step()               # 更新参数

        loss_sum += loss.item() * labels.size(0)
        correct_sum += (logits.argmax(dim=1) == labels).sum().item()
        sample_count += labels.size(0)

        if (i + 1) % log_interval == 0:
            logger.info(
                "Epoch [%d/%d] Step [%d/%d] Loss: %.4f Acc: %.4f",
                epoch + 1, num_epochs, i + 1, len(loader),
                loss_sum / sample_count, correct_sum / sample_count,
            )

    return loss_sum / sample_count, correct_sum / sample_count


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    """在验证集/测试集上评估，返回平均 loss 和准确率。

    两个要点：
    - model.eval()：关闭 dropout，让模型行为确定；
    - @torch.no_grad()：不建计算图，省显存也更快。
    注意装饰器形式的 no_grad 会让函数体内所有代码都不求梯度，
    所以评估函数不要在里面做训练相关操作。
    """
    model.eval()

    loss_sum = 0.0
    correct_sum = 0
    sample_count = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(images)
        loss = loss_fn(logits, labels)

        loss_sum += loss.item() * labels.size(0)
        correct_sum += (logits.argmax(dim=1) == labels).sum().item()
        sample_count += labels.size(0)

    return loss_sum / sample_count, correct_sum / sample_count


# ---------------------------------------------------------------------------
# 7. checkpoint 存取
# ---------------------------------------------------------------------------
def save_checkpoint(path, model, optimizer, epoch, best_val_acc, config):
    """保存可恢复的 checkpoint。

    只存 state_dict，不要 torch.save(model, path)：
    - 整个模型对象会把类定义路径也 pickle 进去，文件移动或改名后就加载失败；
    - 加载整个模型需要 weights_only=False，等于允许执行任意 pickle 代码，有安全风险。

    要能「原样恢复训练现场」，至少需要：
    - model_state_dict：模型参数
    - optimizer_state_dict：优化器状态（Adam 的动量、二阶矩估计都在里面）
    - epoch：下一个 epoch 从哪开始
    - best_val_acc：目前最好的验证指标，用于判断是否更新 best
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "config": config,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None):
    """加载 checkpoint；传入 optimizer 则一并恢复其状态，用于续训。

    weights_only=True 是 PyTorch 2.6 之后的默认值，只允许反序列化张量和基础类型，
    能挡住恶意构造的 pickle 文件，建议保持开启。
    """
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


# ---------------------------------------------------------------------------
# 4. 数据准备
# ---------------------------------------------------------------------------
def build_dataloaders(config):
    """构造 train / val / test 三个 DataLoader。

    MNIST 官方只提供 train(60000) 和 test(10000)，没有验证集，
    所以从 train 里切一份出来做 val。划分要用固定种子的 generator，
    否则每次运行切分结果不同，验证指标没法横向比较。
    """
    transform = transforms.Compose([
        transforms.PILToTensor(),                              # PIL 图像 -> 张量 HWC->CHW
        transforms.ToDtype(torch.float32, scale=True),          # 转 float32 并缩放到 [0, 1]
        transforms.Resize((28, 28)),                            # 统一图像尺寸
    ])

    train_dataset = torchvision.datasets.MNIST(
        root=config["data_dir"], train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root=config["data_dir"], train=False, download=True, transform=transform
    )

    val_size = int(len(train_dataset) * config["val_ratio"])
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config["seed"]),
    )

    train_loader = DataLoader(
        train_subset, batch_size=config["batch_size"], shuffle=True, drop_last=True
    )
    # 验证/测试不需要 shuffle：shuffle 只会打乱日志顺序，不影响指标
    val_loader = DataLoader(val_subset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    return train_loader, val_loader, test_loader


def main():
    set_seed(CONFIG["seed"])

    output_dir = CONFIG["output_dir"]
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    logger = setup_logger(os.path.join(output_dir, "logs"), CONFIG)

    # 数据
    train_loader, val_loader, test_loader = build_dataloaders(CONFIG)
    logger.info(
        "数据准备完成: train=%d, val=%d, test=%d",
        len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset),
    )

    # 模型、损失函数、优化器
    net = SimpleNN(dropout_rate=CONFIG["dropout_rate"]).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=CONFIG["learning_rate"])

    last_ckpt_path = os.path.join(ckpt_dir, "ckpt_last.pth")
    best_ckpt_path = os.path.join(ckpt_dir, "ckpt_best.pth")

    start_epoch = 0
    best_val_acc = 0.0

    # 断点续训：把 CONFIG["resume"] 改成 True，且 last checkpoint 存在时生效
    if CONFIG["resume"] and os.path.exists(last_ckpt_path):
        checkpoint = load_checkpoint(last_ckpt_path, net, optimizer)
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint["best_val_acc"]
        logger.info(
            "从 checkpoint 恢复: 已完成 %d 个 epoch, best_val_acc=%.4f, 从 epoch %d 继续",
            checkpoint["epoch"] + 1, best_val_acc, start_epoch + 1,
        )

    # 训练循环
    logger.info("开始训练...")
    for epoch in range(start_epoch, CONFIG["num_epochs"]):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            net, train_loader, loss_fn, optimizer,
            epoch, CONFIG["num_epochs"], logger, CONFIG["log_interval"],
        )
        val_loss, val_acc = evaluate(net, val_loader, loss_fn)

        # 验证准确率创新高时，另存一份 best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(net.state_dict(), best_ckpt_path)
            logger.info("验证准确率提升，已保存 best checkpoint: %s", best_ckpt_path)

        # 每个 epoch 都保存 last checkpoint，中断后可以原样恢复。
        # 必须放在更新 best_val_acc 之后：否则 checkpoint 里记录的是上一个
        # epoch 的旧指标，恢复后「是否创新高」的判断门槛会被压低。
        save_checkpoint(last_ckpt_path, net, optimizer, epoch, best_val_acc, CONFIG)

        logger.info(
            "Epoch [%d/%d] 完成, 用时 %.1fs | train Loss: %.4f Acc: %.4f | val Loss: %.4f Acc: %.4f",
            epoch + 1, CONFIG["num_epochs"], time.time() - start_time,
            train_loss, train_acc, val_loss, val_acc,
        )

        # 训练集和验证集准确率差距明显拉大，是过拟合的典型信号
        if train_acc - val_acc > 0.05:
            logger.warning(
                "train 与 val 准确率相差 %.4f，可能出现过拟合，考虑加正则或早停",
                train_acc - val_acc,
            )

    logger.info("训练结束, 最佳验证准确率: %.4f", best_val_acc)

    # 测试：加载 best checkpoint 再评估，而不是直接用训练到最后一步的模型
    logger.info("加载 best checkpoint 并在测试集上评估...")
    net.load_state_dict(torch.load(best_ckpt_path, map_location=DEVICE, weights_only=True))
    test_loss, test_acc = evaluate(net, test_loader, loss_fn)
    logger.info("测试集结果: Loss: %.4f Acc: %.4f", test_loss, test_acc)

    logger.info("全部完成。")


if __name__ == "__main__":
    main()
