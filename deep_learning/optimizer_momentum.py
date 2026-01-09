# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------
# File Name:        optimizer_momentum.py
# Author:           wzw
# Version:          0.1
# Created:          2024/11/2
# Description:      一阶动量优化器
# ------------------------------------------------------------------
import numpy as np

# y=theta*x**2+theta*2*x+1
# y'=2*theta*x+2*theta
beta = 0.9
lr = 0.1
l2_lambda = 0.1
m= 0
x = np.random.rand()
theta = np.random.rand()
for epoch in range(100):
    y = theta * x ** 2 + theta * 2 * x
    print("逐渐减小的y值:", y)
    # loss=y**2+0.5*l2_lambda * theta**2
    grad = 2 * y * (x ** 2 + 2 * x) + l2_lambda * theta
    m = beta * m + (1 - beta) * grad
    v_mat = m / (1 - beta ** (epoch + 1))
    theta = theta - lr * v_mat
