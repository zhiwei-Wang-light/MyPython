# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------
# File Name:        kalman_filter.py
# Author:           wzw
# Version:          0.1
# Created:          2026/2/3
# Description:      卡尔曼滤波
# ------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
Z=np.arange(0,100)
# 设置标准差为scale的噪声
noise=np.random.normal(loc=0,scale=5,size=100)
# 观测值，存在误差
Z=Z+noise
# 预测状态距离与速度初始均为0
xTrue=np.array([[0],[0]])
# 初始化观测距离
xObservation=0
# 状态协方差矩阵,shape=(n,n)
pPre=np.array([[1, 0], [0, 1]])
# 过程协方差噪声,shape=(n,n)
Q = np.array([[0.0001, 0], [0, 0.0001]])
# 观测转移矩阵,shape=(n,1)
H = np.array([[1,0]])
# 测量噪声方差
R = np.diag([25])
xpre=[]
xcor=[]
for i in range(100):
    # 预测部分
    # 时间间隔为1s
    dt=1
    # 状态矩阵
    F=np.array([[1,dt],[0,1]])
    # 控制矩阵
    B=np.array([[(dt**2)/2],[dt]])
    # 加速度设为0
    a=0
    # 状态预测,shape=(2,1)
    xTrue=np.dot(F,xTrue)+np.dot(B,a)
    # 协方差矩阵，shape=(2,2)
    pPre= (F @ pPre) @ F.transpose([1, 0]) + Q
    # 增益矩阵,shape=(2,1)
    kt=(pPre@H.transpose([1,0]))@(np.linalg.inv(H @ pPre @ (H.transpose([1, 0])) + R))
    # 观测数据
    xObservation=Z[i]
    print("xObservation",xObservation)
    # print("xObservation shape",xObservation.shape)
    xTrue=xTrue+kt@(xObservation-H@xTrue)
    xpre.append(xObservation)
    print("xTrue", xTrue[0])
    xcor.append(xTrue[0][0])
    pPre=(np.eye(2)-kt@H)@pPre
plt.plot(range(len(Z)), xpre, label = 'Measurements')
plt.plot(range(len(Z)), xcor, label = 'Kalman Filter Prediction')
plt.legend()
plt.show()