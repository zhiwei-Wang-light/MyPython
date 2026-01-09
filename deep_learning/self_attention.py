# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------
# File Name:        self_attention.py
# Author:           wzw
# Version:          0.1
# Created:          2024/10/31
# Description:      自注意力机制,这里注意使用的是torch.bmm,采用mask的方法将padding的地方设置为负无穷
# ------------------------------------------------------------------
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SelfAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.W_Q = nn.Linear(embedding_dim, embedding_dim)  # 查询线性层
        self.W_K = nn.Linear(embedding_dim, embedding_dim)  # 键线性层
        self.W_V = nn.Linear(embedding_dim, embedding_dim)  # 值线性层

        # 创建填充掩码（0表示填充，1表示有效输入）
        self.mask = torch.tensor([[1, 1, 1, 0, 0],
                                  [1, 1, 0, 0, 0]]).unsqueeze(1).cuda()  # 形状为 (batch_size, 1, seq_length)

    def forward(self, inputs):
        # inputs 形状: (batch_size, seq_length, embedding_dim)
        Q = self.W_Q(inputs)  # (batch_size, seq_length, embedding_dim)
        K = self.W_K(inputs)  # (batch_size, seq_length, embedding_dim)
        V = self.W_V(inputs)  # (batch_size, seq_length, embedding_dim)

        # 计算注意力分数
        scores = torch.bmm(Q, K.transpose(1, 2)) / (
                    self.embedding_dim ** 0.5)  # (batch_size, seq_length, embedding_dim)
        scores = scores.masked_fill(self.mask == 0, float('-inf'))
        # 应用Softmax获取注意力权重
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)

        # 计算加权和
        output = torch.bmm(attention_weights, V)  # (batch_size, seq_length, embedding_dim)

        return output, attention_weights

# 使用示例
embedding_dim = 512
seq_length = 5
batch_size = 2
inputs = torch.rand(batch_size, seq_length, embedding_dim).cuda()  # 随机输入

self_attention = SelfAttention(embedding_dim).cuda()
output, attention_weights = self_attention(inputs)

print("Output:\n", output)
print("Attention Weights:\n", attention_weights)
