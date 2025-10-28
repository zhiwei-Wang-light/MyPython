# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------
# File Name:        image_scale.py
# Author:           wzw
# Version:          0.1
# Created:          2025/10/28
# Description:      图像缩放
# ------------------------------------------------------------------
import cv2
import numpy as np


def resize_and_pad(imageA, K_a, target_width, target_height, pad_value=0):
    """
    将图像A调整到目标分辨率（通过填充保持宽高比），并更新内参矩阵。

    参数：
        imageA: 输入图像，形状 (H_a, W_a, C)。
        K_a: 原始内参矩阵 (3x3)。
        target_width: 目标宽度（图像B的宽度）。
        target_height: 目标高度（图像B的高度）。
        pad_value: 填充值（默认为0）。

    返回：
        imageA_padded: 填充后的图像，形状 (target_height, target_width, C)。
        K_a_new: 调整后的内参矩阵 (3x3)。
    """
    # 原始图像的宽高
    h_a, w_a = imageA.shape[:2]

    # 计算缩放比例（保持宽高比）
    scale = min(target_width / w_a, target_height / h_a)

    # 缩放图像
    new_w = int(w_a * scale)
    new_h = int(h_a * scale)
    imageA_resized = cv2.resize(imageA, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 计算填充量（左右或上下填充）
    pad_w = target_width - new_w
    pad_h = target_height - new_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2

    # 填充图像
    imageA_padded = cv2.copyMakeBorder(
        imageA_resized,
        pad_top, pad_h - pad_top,  # 上、下填充
        pad_left, pad_w - pad_left,  # 左、右填充
        cv2.BORDER_CONSTANT,
        value=pad_value
    )
    cv2.namedWindow('imageA_padded', cv2.WINDOW_NORMAL)
    cv2.imshow('imageA_padded', imageA_padded)

    # 等待按键关闭窗口（按任意键关闭所有窗口）
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 调整内参矩阵
    K_a_new = K_a.copy()
    K_a_new[0, 0] *= scale  # fx
    K_a_new[1, 1] *= scale  # fy
    K_a_new[0, 2] = (K_a[0, 2] - 0.5) * scale + pad_left + 0.5  # cx
    K_a_new[1, 2] = (K_a[1, 2] - 0.5) * scale + pad_top + 0.5  # cy

    return imageA_padded, K_a_new


# 假设 imageA 和 imageB 是输入图像
imageA = cv2.imread(
    "/home/jd/wangzhiwei225/JDCode/RobotTaskSupervisor/SourceVision/标定数据/格口相机1/check_data/target/rgb_0.png")  # 形状 (H_a, W_a, 3)
imageB = cv2.imread(
    "/home/jd/wangzhiwei225/JDCode/RobotTaskSupervisor/SourceVision/标定数据/格口相机1/check_data/target/gray_0.png")  # 形状 (H_b, W_b, 3)

# 图像A的原始内参矩阵（示例）
K_a = np.array([
    [500, 0, 320],  # fx, 0, cx
    [0, 500, 240],  # 0, fy, cy
    [0, 0, 1]
])

# 调用函数调整图像A
target_width = imageB.shape[1]
target_height = imageB.shape[0]
imageA_padded, K_a_new = resize_and_pad(imageA, K_a, target_width, target_height)

print("调整后的内参矩阵:\n", K_a_new)
cv2.imwrite("imageA_padded.jpg", imageA_padded)
