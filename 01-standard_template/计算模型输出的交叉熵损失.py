# 计算模型输出的交叉熵损失.py
import torch
import torch.nn.functional as F
import torch.nn as nn

vocab_size = 10
pad_id = 0

# 每行是一条已补齐的序列；1 可看作 BOS，2 可看作 EOS，0 是 Padding。
input_ids = torch.tensor([
    [1, 4, 5, 2],
    [1, 7, 2, 0],
])

# 第 t 个位置的 logits 要预测下一个 token，因此标签向左移动一位。
# -100 表示 Padding 对应的目标不参与交叉熵计算。
labels = torch.tensor([
    [4, 5, 2, -100],
    [7, 2, -100, -100],
])

# 实际代码中这里应为 model(input_ids) 的输出，形状是 (B=2, T=4, V=10)。
logits = torch.randn(2, 4, vocab_size, requires_grad=True)

# F.cross_entropy 需要类别维度 V 位于第 2 维，标签形状与其余维度 (B, T) 对齐。
# criterion = nn.CrossEntropyLoss() 与 F.cross_entropy 基本同效。一个是模块式 API，一个是函数式 API
loss = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100)
print(f"一、使用 F.cross_entropy 计算的交叉熵损失为：")
print(loss.item())


"""
手动实现思路：
→ (1) Softmax 得到每个类别的概率
→ (2) 取出每个有效位置上“真实标签对应类别”的概率
→ (3) 取负对数
→ (4) 求平均
"""
print(f"\n二、手动计算的交叉滴损失：")
# (1) Softmax 得到每个类别的概率
softmax_op = nn.Softmax(dim=-1)
proba_dis = softmax_op(logits) # 在最后一维上对 logits 执行 Softmax 操作，得到概率分布

# (2) 取出每个有效位置上“真实标签对应类别”的概率
valid_mask = labels != -100 # 有效位置掩码
batch_indices, time_indices = torch.where(valid_mask) # 获取有效位置索引坐标
prob_list = []
for batch, time in zip(batch_indices, time_indices):
    # print(f"batch: {batch}, time: {time}")
    real_label = labels[batch][time] # 有效位置标签（正确词元ID）
    # 概率前面已经计算得到，此处根据batch和time坐标定位具体的位置。再根据real_label定位到正确下一个词元的概率
    pro = proba_dis[batch][time][real_label].item()
    prob_list.append(pro) # 添加到列表中，最后用于汇总统计
    # print("*"* 50)
print(f"prob_list: {prob_list}")

# (3) 取负对数
negative_log_res = -torch.log(torch.tensor(prob_list))
print(f"negative_log_res : {negative_log_res}")

# (4) 求平均
cross_entropy_loss = negative_log_res.mean()
print(f"cross_entropy_loss: {cross_entropy_loss.item()}")

print(f"\n两次计算得到的交叉熵是否相同： {torch.allclose(loss, cross_entropy_loss)}")