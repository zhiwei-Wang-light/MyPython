# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------
# File Name:        ppl.py
# Author:           wzw
# Version:          0.1
# Created:          2024/10/26
# Description:      困惑度计算,困惑度越小,说明模型对语言的理解能力更强,困惑度的其中一种表达方式
# ------------------------------------------------------------------
import numpy as np
from sklearn.metrics import log_loss

def calculate_perplexity(predicted_probs, true_labels):
    # 计算交叉熵
    cross_entropy = log_loss(true_labels, predicted_probs)
    # 计算困惑度
    perplexity = np.exp(cross_entropy)
    return perplexity

# 假设我们预测了 5 个字的概率
predicted_probs = np.array([[0.1, 0.9],  # 第一个字的概率分布
                             [0.7, 0.3],  # 第二个字的概率分布
                             [0.2, 0.8],  # 第三个字的概率分布
                             [0.4, 0.6],  # 第四个字的概率分布
                             [0.6, 0.4]]) # 第五个字的概率分布

# 真实标签（假设字的真实类别用 0 和 1 表示）
true_labels = np.array([1, 0, 1, 1, 0])  # 真实标签

# 计算困惑度
perplexity = calculate_perplexity(predicted_probs, true_labels)
print(f"困惑度: {perplexity:.4f}")
