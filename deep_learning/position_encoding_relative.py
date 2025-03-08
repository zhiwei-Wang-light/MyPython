# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------
# File Name:        position_encoding_relative.py
# Author:           wzw
# Version:          0.1
# Created:          2024/11/2
# Description:      相对位置编码,作用在K上
# ------------------------------------------------------------------
import math
import torch

def relative_position_encoding(seq_len, d_model):
    # 创建相对位置矩阵
    position = torch.arange(seq_len).unsqueeze(0)  # 1 x seq_len
    relative_positions = position - position.T  # seq_len x seq_len

    # 限制相对位置编码范围
    relative_positions = relative_positions.clamp(min=-seq_len + 1, max=seq_len - 1)

    # 计算每个位置的编码（按sin/cos方式）
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

    # 计算相对位置编码
    sin_vals = torch.sin(relative_positions.unsqueeze(-1) * div_term)  # 正弦部分
    cos_vals = torch.cos(relative_positions.unsqueeze(-1) * div_term)  # 余弦部分

    # 将正弦和余弦交替放在相同的维度上
    relative_position_encodings = torch.zeros((seq_len, d_model))
    relative_position_encodings[:, 0::2] = sin_vals.mean(dim=1)  # 偶数维度使用正弦
    relative_position_encodings[:, 1::2] = cos_vals.mean(dim=1)  # 奇数维度使用余弦

    return relative_position_encodings

# 示例使用
seq_len = 10
d_model = 16
rpe = relative_position_encoding(seq_len, d_model)
print(rpe.shape)  # 应输出 [seq_len, d_model]

