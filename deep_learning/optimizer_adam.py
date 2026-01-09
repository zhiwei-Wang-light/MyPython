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
l2_lambda = 0.1
x = np.random.rand()
theta = np.random.rand()
for epoch in range(100):
    y = theta * x ** 2 + theta * 2 * x
    # loss=y**2+0.5*l2_lambda * theta**2
    grad = 2 * y * (x ** 2 + 2 * x) + l2_lambda * theta
    print("逐渐减小的y值:", y ** 2)
    m = beta_m * m + (1 - beta_m) * grad
    v = beta_v * v + (1 - beta_v) * grad ** 2
    m_hat = m / (1 - beta_m ** (epoch + 1))
    v_hat = v / (1 - beta_v ** (epoch + 1))
    theta = theta - lr * m_hat / (math.sqrt(v_hat) + eps)
    print("theta:", theta)
