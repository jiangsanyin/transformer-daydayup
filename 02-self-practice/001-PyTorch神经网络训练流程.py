"""
基于手写数字数据集的数字识别神经网络的训练流程：

0. 导入必要的库和模块，包括PyTorch、torchvision等。
1. 定义一些超参数，如批次大小、学习率和训练轮数。
2. 数据准备（包括数据加载和数据预处理，数据预处理要提前定义好处理管道）
3. 定义神经网络模型
4. 定义损失函数和优化器
5. 训练模型（过程中输出损失与准确率）
6. 测试模型（过程中输出准确率）
7. 模型保存
8. 模型加载与使用
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms.v2 as transforms
import os

# 定义超参数
batch_size = 64
learning_rate = 0.001
num_epochs = 3

train_transform = transforms.Compose([
    transforms.PILToTensor(), # 将PIL图像转换为张量 HWC->CHW
    transforms.ToDtype(torch.float32, scale=True), # 将张量转换为float32类型，并将像素值缩放到[0, 1]范围
    transforms.Resize((28, 28)), # 将图像大小调整为28x28
])

test_transform = transforms.Compose([
    transforms.PILToTensor(), # 将PIL图像转换为张量 HWC->CHW
    transforms.ToDtype(torch.float32, scale=True), # 将张量转换为float32类型，并将像素值缩放到[0, 1]范围
    transforms.Resize((28, 28)), # 将图像大小调整为28x28
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


class SimpleNN(nn.Module):
    def __init__(self, dropout_rate=0.1):
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

net = SimpleNN(dropout_rate=0.1)  # 实例化神经网络模型

loss_fn = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
optimizer = optim.Adam(net.parameters(), lr=learning_rate)  # 定义Adam优化器

print("Starting training...")
net.train()  # 设置模型为训练模式
for epoch in range(num_epochs):
    sample_count = 0
    correct_count = 0
    loss_sum = 0.0
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()  # 清空梯度
        predict = net(images)  # 前向传播，得到预测结果
        loss = loss_fn(predict, labels)  # 计算损失
        correct_count += (predict.argmax(dim=1) == labels).sum().item()  # 计算正确预测的数量
        sample_count += labels.size(0)  # 计算总样本数
        loss_sum += loss.item() * labels.size(0)  # 累加损失值
        if (i + 1) % 100 == 0:  # 每100个批次输出一次训练结果
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}]", \
                  f"Loss: {loss_sum / sample_count:.4f}, Accuracy: {correct_count / sample_count:.4f}")

        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 反向传播，更新参数
    print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{len(train_loader)}/{len(train_loader)}]", \
                      f"Loss: {loss_sum / sample_count:.4f}, Accuracy: {correct_count / sample_count:.4f}")

print("Training finished.")

print("\nStarting testing...")
net.eval()  # 设置模型为评估模式
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

with torch.no_grad():  # 在测试阶段不需要计算梯度
    sample_count = 0
    correct_count = 0
    for i, (images, labels) in enumerate(test_loader):
        predict = net(images) # 前向传播，得到预测结果
        correct_count += (predict.argmax(dim=1) == labels).sum().item()  # 计算正确预测的数量
        sample_count += labels.size(0)  # 计算总样本数
        if (i + 1) % 10 == 0:  # 每10个批次输出一次测试结果
            print(f"Step [{i + 1}/{len(test_loader)}], Accuracy: {correct_count / sample_count:.4f}")
    print(f"Final Accuracy on the test dataset: {correct_count / sample_count:.4f}")

print("Testing finished.")



print("\nSaving the model...")
save_dir = './output'
os.makedirs(save_dir, exist_ok=True)

save_path1 = f'{save_dir}/simple_nn_whole.pth'
torch.save(net, save_path1)  # 保存模型参数
print(f"Model whole saved to {save_path1}")

save_path2 = f'{save_dir}/simple_nn_state_dict.pth'
torch.save(net.state_dict(), save_path2)  # 保存模型的状态字典
print(f"Model state_dict saved to {save_path2}")


print("\nLoading the model and testing...")
# model = torch.load(save_path1, weights_only=False)  # 加载整个模型
model = SimpleNN(dropout_rate=0.1)  # 实例化神经网络模型
model.load_state_dict(torch.load(save_path2))  # 加载模型的状态字典
model.eval()  # 设置模型为评估模式
with torch.no_grad():  # 在测试阶段不需要计算梯度
    sample_count = 0
    correct_count = 0
    for i, (images, labels) in enumerate(test_loader):
        predict = model(images) # 前向传播，得到预测结果
        correct_count += (predict.argmax(dim=1) == labels).sum().item()  # 计算正确预测的数量
        sample_count += labels.size(0)  # 计算总样本数
        if (i + 1) % 10 == 0:  # 每10个批次输出一次测试结果
            print(f"Step [{i + 1}/{len(test_loader)}], Accuracy: {correct_count / sample_count:.4f}")
    print(f"Final Accuracy on the test dataset: {correct_count / sample_count:.4f}")

print("All done.")