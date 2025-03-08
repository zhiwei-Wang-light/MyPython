# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------
# File Name:        optimizer_adam.py
# Author:           wzw
# Version:          0.1
# Created:          2024/11/4
# Description:      adam优化器
# ------------------------------------------------------------------
import numpy as np
import math

# y=theta*x**2+theta*2*x+1
# y'=2*theta*x+2*theta
beta_m = 0.9
beta_v = 0.999
lr = 0.1
eps = 1e-8
m = 0
v = 0
x = np.random.rand()
theta = np.random.rand()
y = theta * x ** 2 + theta * 2 * x
print("初始的y值:", y)
for epoch in range(100):
    m = beta_m * m + (1 - beta_m) * (x ** 2 + 2 * x)
    v = beta_v * v + (1 - beta_v) * (x ** 2 + 2 * x) **2
    m_hat = m / (1 - beta_m ** (epoch+1))
    v_hat = v / (1 - beta_v ** (epoch+1))
    theta = theta - lr * m_hat / (math.sqrt(v_hat) + eps)
    y = theta * x ** 2 + theta * 2 * x
    print("逐渐减小的y值:", y)
