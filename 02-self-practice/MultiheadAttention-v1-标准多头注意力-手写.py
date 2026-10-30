"""
此为手写针对 Transformer 模型框架中标准多头注意力机制实现。

大致流程或步骤如下：
1. 模拟输入一批序列，每个序列的最大长度是20，序列中每个词向量用256维向量表示，每批是32个序列。
2. 以下功能都封装在一个自定义多头注意力类 MultiheadAttention 中：
    2.1 构造函数中对一些必要的属性进行初始化赋值。
    2.2 求取原始 Q、K、V
    2.3 将原始 Q、K、V 划分给多个注意力头（此处用8个注意力头），并变更数据形状[batch_size, head_num, seq_len, hid_dim]
    2.4 计算原始注意力得分，然后softmax得到注意力权重，并进行dropout操作
    2.5 将注意力权重与V做矩阵乘法，得到原始注意力矩阵
    2.6 将多个注意力头结果进行合并，此时数据形状为[batch_size, seq_len, hid_dim]
    2.7 将合并后的注意力头结果经过一层全连接层做线性变换，融合各头信息。
    2.8 返回结果

"""

import torch
import torch.nn as nn

class MultiheadAttention(nn.Module):
    """
    手写多头注意力标准实现类
    """

    def __init__(self, hid_dim, head_num, dropout):
        super().__init__()

        # 单个词向量用多少维向量表示
        self.hid_dim = hid_dim
        # 多少个注意力头
        self.head_num = head_num
        # 断言两者能够整除
        assert (self.hid_dim % self.head_num == 0)

        # 如下三个线性层中都包含一个权重矩阵与偏置项。分别用来表示 W^Q、W^K、W^V
        self.wq_ = nn.Linear(hid_dim, hid_dim)
        self.wk_ = nn.Linear(hid_dim, hid_dim)
        self.wv_ = nn.Linear(hid_dim, hid_dim)

        # 随机丢弃一定比例的注意力权重值，防止过拟合
        self.dropout = nn.Dropout(dropout)

        # 最后的全连接层，融合各头信息
        self.fc = nn.Linear(hid_dim, hid_dim)
        # 对原始注意力得分进行缩放，防止注意力权重过于尖锐
        self.scale = (self.hid_dim // self.head_num) ** 0.5


    def forward(self, input, mask=None):
        # input 此时的形状是：[batch_size, seq_len, hid_dim] ， 此例中具体为：[32, 20, 256]

        # 2.2 求取原始 Q、K、V
        # input 的最后一维与线性层做线性变换
        Q = self.wq_(input) # Q的形状为：[batch_size, seq_len, hid_dim]
        K = self.wk_(input) # K的形状为：[batch_size, seq_len, hid_dim]
        V = self.wv_(input) # V的形状为：[batch_size, seq_len, hid_dim]
        
        # 2.3 将原始 Q、K、V 划分给多个注意力头（此处用8个注意力头），并变更数据形状[batch_size, head_num, seq_len, hid_dim]
        # [batch_size, seq_len, hid_dim] -> [batch_size, seq_len, self.head_num, self.hid_dim//self.head_num]
        #       -> [batch_size, self.head_num, seq_len, self.hid_dim//self.head_num]
        # 即 [32, 20, 256] -> [32, 20, 8, 32] -> [32, 8, 20, 32]
        batch_size = input.shape[0]
        # 线性变换操作的输出通常是连续存储的张量，所以此处不需要先执行 .contiguous() 操作
        Q = Q.view(batch_size, -1, self.head_num, self.hid_dim//self.head_num) \
            .permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.head_num, self.hid_dim//self.head_num) \
            .permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.head_num, self.hid_dim//self.head_num) \
            .permute(0, 2, 1, 3)

        # 2.4 计算原始注意力得分，然后softmax得到注意力权重
        # [32, 8, 20, 32] * [32, 8, 32, 20] => [32, 8, 20, 20]，最后除以缩放因子
        attention_score = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e10)

        # 在最后一个维度上做softmax操作
        attention = torch.softmax(attention_score, dim=-1)
        # 做 dropout 操作
        attention = self.dropout(attention)

        # 2.5 将注意力权重与V做矩阵乘法，得到原始注意力矩阵
        # [32, 8, 20, 20] * [32, 8, 20, 32] => [32, 8, 20, 32]
        z_ = torch.matmul(attention, V)

        # 2.6 将多个注意力头结果进行合并，此时数据形状为[batch_size, seq_len, hid_dim]
        # 变换维度，[32, 8, 20, 32] -> [32, 20, 8, 32]。并在底层上重新分配连续内存，保证 .view 操作安全
        z_ = z_.permute(0, 2, 1, 3).contiguous()
        # 将最后两维进行合并
        z_ = z_.view(batch_size, -1, self.head_num * (self.hid_dim//self.head_num))

        # 2.7 将合并后的注意力头结果经过一层全连接层做线性变换，融合各头信息
        z = self.fc(z_)

        return z



# 1. 构造输入
input = torch.rand(32, 20, 256)

attention = MultiheadAttention(256, 8, 0.1)
z = attention(input)
print(f"输入形状为：{input.shape}")
print(f"输入经过多头注意力层作用后，结果形状为：{z.shape}")