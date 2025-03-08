# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------
# File Name:        position_encoding_absolute.py
# Author:           wzw
# Version:          0.1
# Created:          2024/11/2
# Description:      绝对位置编码，分母的实现 torch.exp(torch.arrange(0,d_model,2)*(-log(10000)/d_model))
# ------------------------------------------------------------------
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        初始化位置编码。

        参数：
        - d_model (int): 嵌入向量的维度。
        - max_len (int): 序列的最大长度。
        """
        super(PositionalEncoding, self).__init__()

        # 创建一个 (max_len, d_model) 的位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 计算 PE(pos, 2i) 和 PE(pos, 2i+1)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度

        pe = pe.unsqueeze(0)  # 增加一个batch维度，形状为 (1, max_len, d_model)
        self.register_buffer('pe', pe)  # 将pe注册为缓冲区，避免其被视为模型参数

    def forward(self, x):
        """
        将位置编码加到输入的嵌入向量上。

        参数：
        - x (Tensor): 输入嵌入，形状为 (batch_size, seq_len, d_model)

        返回：
        - Tensor: 加入位置编码后的嵌入，形状为 (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        # 将位置编码裁剪到输入序列的长度，并添加到输入中
        x = x + self.pe[:, :seq_len, :]
        return x


# 示例用法
if __name__ == "__main__":
    d_model = 512  # 嵌入维度
    max_len = 50  # 序列最大长度
    batch_size = 2  # 批量大小
    seq_len = 50  # 当前序列长度

    # 初始化位置编码模块
    pos_encoding = PositionalEncoding(d_model, max_len)

    # 创建一个示例输入张量 (batch_size, seq_len, d_model)
    input_tensor = torch.zeros(batch_size, seq_len, d_model)

    # 应用位置编码
    output = pos_encoding(input_tensor)

    print("输入张量形状:", input_tensor.shape)  # 输出: torch.Size([2, 50, 512])
    print("输出张量形状:", output.shape)  # 输出: torch.Size([2, 50, 512])
    print("位置编码示例:", output[0, 0, :10])  # 输出第一个样本第一个位置的前10维位置编码
