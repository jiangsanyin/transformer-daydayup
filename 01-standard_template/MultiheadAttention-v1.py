"""
实现 Transformer 的多头自注意力（Multi-Head Self-Attention）机制。

此文件实现的是标准的“缩放点积多头自注意力”核心计算。
输入是一批已经包含位置信息的词向量。程序先用三套不同的线性层生成
Query（查询）、Key（键）和 Value（值）；再让每个词的 Query 与全部词的
Key 计算相关性，以相关性为权重对 Value 加权求和。最后把多个注意力头的
结果拼接并融合，输出与输入序列等长、等维度的上下文表示。
"""

import torch
import torch.nn as nn

class MultiheadAttention(nn.Module):
    # hid_dim：每个词的输入、输出向量维度，例如 300
    # n_heads：并行计算的注意力头数量，例如 6
    # dropout：训练阶段随机丢弃注意力权重的比例，例如 0.1
    def __init__(self, hid_dim, n_heads, dropout):
        # 初始化 nn.Module，使线性层和 Dropout 能被 PyTorch 自动注册
        super(MultiheadAttention, self).__init__()
        # 保存超参数，供 forward 中拆分和拼接多个注意力头时使用
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        # 每个头的向量维度为 hid_dim // n_heads。
        # 例如 300 维拆成 6 个头时，每个头处理 50 维，因此 hid_dim 必须能被整除。
        assert hid_dim % n_heads == 0
        # 将每个词的输入向量映射为 Query，用于“当前词应该向哪些词查询信息”。
        # nn.Linear(hid_dim, hid_dim) 内含权重 [hid_dim, hid_dim] 和偏置 [hid_dim]。
        # 每个词的 [1, hid_dim] 向量经过此层后，形状仍为 [1, hid_dim]。
        self.w_q = nn.Linear(hid_dim, hid_dim)
        # 将每个词的输入向量映射为 Key，用于被其他词的 Query 匹配。
        # w_k 与 w_q 形状相同，但参数独立，学习到的变换也不同。
        self.w_k = nn.Linear(hid_dim, hid_dim)
        # 将每个词的输入向量映射为 Value，作为最终被加权汇总的信息内容。
        # w_v 同样拥有一套独立的权重和偏置。
        self.w_v = nn.Linear(hid_dim, hid_dim)
        # 拼接所有头的结果后，再用该线性层混合不同头提取到的信息。
        self.fc = nn.Linear(hid_dim, hid_dim)
        # 训练阶段会随机丢弃部分注意力权重，减少模型总依赖相同词对的风险。
        self.do = nn.Dropout(dropout)
        # 缩放系数为每个头维度的平方根，例如 sqrt(50)。
        # 除以它可避免向量维度较大时点积过大，使 Softmax 的分布过于极端。
        self.scale = (hid_dim // n_heads) ** 0.5

    def forward(self, query, mask=None):
        # 这是自注意力：Q、K、V 都来自同一个 query 输入。
        # query 形状为 [batch_size, seq_len, hid_dim]，示例为 [64, 12, 300]：
        # 64 个样本、每个样本 12 个词、每个词一个 300 维向量。
        # mask 是可选掩码，形状需能广播为 [batch_size, n_heads, seq_len, seq_len]。
        # mask 中为 0 的位置禁止被关注，可用于忽略补齐词（PAD）或屏蔽未来词。

        # 取第 0 维的长度，即当前批次中的样本数；示例中 bsz 为 64。
        bsz = query.shape[0]
        # nn.Linear 会对最后一维的每个词向量单独做相同的线性变换，
        # 前两个维度（批次大小和词数）不变。
        # 因此 Q、K、V 的形状均为 [64, 12, 300]，但其数值来自三套不同参数。
        Q = self.w_q(query)
        K = self.w_k(query)
        V = self.w_v(query)

        # 将每个 300 维向量拆成 6 个头，每个头的维度为 300 // 6 = 50。
        # view 中的 -1 表示让 PyTorch 根据元素总数自动推导 seq_len（本例为 12）。
        # 第一步：[64, 12, 300] -> [64, 12, 6, 50]
        # 第二步 permute(0, 2, 1, 3) 调换“词数”和“头数”维度：
        #        [64, 12, 6, 50] -> [64, 6, 12, 50]。
        # 这样每个注意力头都能独立处理一份 [12, 50] 的词向量序列。
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)

        # 第 1 步：计算缩放点积注意力的原始分数 QK^T / sqrt(head_dim)。
        # K.permute(0, 1, 3, 2) 只交换 K 的最后两个维度：
        # [64, 6, 12, 50] -> [64, 6, 50, 12]。
        # torch.matmul 保持前两维（样本和头）分别独立计算矩阵乘法：
        # 每个样本、每个头中：[12, 50] * [50, 12] -> [12, 12]。
        # 结果 attention 的 [i, j] 元素，是第 i 个词的 Query 和第 j 个词的 Key
        # 的 50 维点积，表示第 i 个词应关注第 j 个词的原始程度。
        # 所有样本和头合在一起的形状为 [64, 6, 12, 12]。
        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # 在归一化前处理掩码：把禁止关注的位置改为很小的负数。
        # 这些位置经过 Softmax 后的指数值接近 0，因此注意力权重也接近 0。
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        # 第 2 步：沿最后一维（每个 Query 可关注的全部 Key）执行 Softmax。
        # 此时每个 Query 对应的权重向量原本之和为 1，例如 [0.2, 0.5, 0.3]。
        # 随后在训练阶段执行 Dropout：约 dropout 比例的权重会置为 0，
        # 未置零的权重会按保留概率缩放，所以 Dropout 后单个权重向量之和不一定为 1。
        # 在 model.eval() 推理模式下 Dropout 不生效，权重向量之和仍为 1。
        # attention 形状保持为 [64, 6, 12, 12]。
        attention = self.do(torch.softmax(attention, dim=-1))

        # 第 3 步：用每个 Query 对各词的注意力权重，对对应 Value 加权求和。
        # 对某一个头而言：[12, 12] 中每一行的 12 个权重，
        # 会乘以 V 中对应的 12 个 [50] 维 Value 并累加，得到一个 [50] 维新向量。
        # 所有头一起计算：[64, 6, 12, 12] * [64, 6, 12, 50] -> [64, 6, 12, 50]。
        x = torch.matmul(attention, V)

        # 目前 x 的维度顺序是 [批次, 头数, 词数, 每头维度]。
        # 将它转为 [批次, 词数, 头数, 每头维度]，使同一个词的 6 个头相邻。
        # [64, 6, 12, 50] -> [64, 12, 6, 50]。
        # permute 只改变张量的观察顺序，内存布局可能不连续；
        # contiguous() 创建连续布局，确保后续 view 可以安全地重塑张量。
        x = x.permute(0, 2, 1, 3).contiguous()
        # 将同一个词的 6 个 [50] 维结果在最后一维拼接，恢复为 300 维。
        # [64, 12, 6, 50] -> [64, 12, 6 * 50] -> [64, 12, 300]。
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        # 对拼接结果进行最后一次线性变换，融合各头信息。
        # 输出形状仍是 [batch_size, seq_len, hid_dim]，本例为 [64, 12, 300]。
        x = self.fc(x)
        return x


# 构造一批随机词向量：64 个样本，每个样本包含 12 个词，每个词为 300 维。
# 实际使用时，这里通常是“词嵌入 + 位置嵌入”或前一层 Transformer 的输出。
x = torch.rand(64, 12, 300)
# 创建 6 头自注意力层，每个头处理 300 // 6 = 50 维特征；训练时丢弃 10% 注意力权重。
attention = MultiheadAttention(hid_dim=300, n_heads=6, dropout=0.1)
# 执行前向传播；输出与输入在批次大小、词数和向量维度上保持一致。
output = attention(x)
# 输出：torch.Size([64, 12, 300])
print(output.shape)


