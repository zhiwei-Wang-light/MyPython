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
v = 0
x = np.random.rand()
theta = np.random.rand()
y = theta * x ** 2 + theta * 2 * x
print("初始的y值:", y)
for epoch in range(100):
    v = beta * v + (1 - beta) * (x ** 2 + 2 * x)
    v_mat = v / (1 - beta ** (epoch+1))
    theta = theta - lr * v_mat
    y = theta * x ** 2 + theta * 2 * x
    print("逐渐减小的y值:", y)
