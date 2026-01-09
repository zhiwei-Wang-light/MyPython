# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------
# File Name:        sift_match.py
# Author:           wzw
# Version:          0.1
# Created:          2025/10/28
# Description:      特征点匹配
# ------------------------------------------------------------------
import cv2
import numpy as np
import os


def translate_image(image, x, y):
    """
    平移图像
    :param image: 输入图像
    :param x: 水平平移量（正数向右，负数向左）
    :param y: 垂直平移量（正数向下，负数向上）
    :return: 平移后的图像
    """
    # 构建平移矩阵
    M = np.float32([[1, 0, x], [0, 1, y]])

    # 获取图像尺寸
    (h, w) = image.shape[:2]

    # 应用仿射变换
    shifted = cv2.warpAffine(image, M, (w, h))

    return shifted


def feature_matching_with_bf(image_path1, image_path2, window_size=(1200, 600)):
    # 读取图像并转换为灰度
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
    # 应用直方图均衡化
    # img1 = cv2.equalizeHist(img1)
    # img2 = cv2.equalizeHist(img2)
    # img2=translate_image(img2,2,2)

    if img1 is None or img2 is None:
        print("Error: Could not read images")
        return None

    # 初始化SIFT检测器
    sift = cv2.SIFT_create()

    # 检测关键点和计算描述符
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 初始化暴力匹配器（使用L2距离）
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # 进行匹配
    matches = bf.match(des1, des2)

    # 按匹配距离（得分）升序排序
    matches = sorted(matches, key=lambda x: x.distance)

    # 获取前10个最佳匹配
    top10_matches = matches[:200]
    # 准备用于RANSAC的点
    src_pts = np.float32([kp1[m.queryIdx].pt for m in top10_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in top10_matches]).reshape(-1, 1, 2)

    # 使用RANSAC计算单应性矩阵并筛选离群点
    if len(src_pts) >= 4:  # 至少需要4个点来计算单应性矩阵
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 1.0)
        mask = mask.ravel().tolist()
    else:
        print("Warning: Not enough points for RANSAC (need at least 4)")
        mask = [1] * len(top10_matches)  # 如果点数不足，将所有点视为内点

    # 筛选内点
    inlier_matches = [top10_matches[i] for i in range(len(top10_matches)) if mask[i]]
    outlier_matches = [top10_matches[i] for i in range(len(top10_matches)) if not mask[i]]
    # print("inlier_matches:", inlier_matches)
    # print("outlier_matches", outlier_matches)
    final_matches = inlier_matches
    # 计算匹配点对的像素坐标和欧氏距离
    matched_points = []
    for m in final_matches:
        # 获取匹配点的像素坐标
        pt1 = kp1[m.queryIdx].pt  # 图像1中的点
        pt2 = kp2[m.trainIdx].pt  # 图像2中的点

        # 计算欧氏距离
        distance = np.linalg.norm(np.array(pt1) - np.array(pt2))

        matched_points.append({
            'point1': pt1,  # (x1, y1)
            'point2': pt2,  # (x2, y2)
            'pixel_distance': distance  # 像素距离
        })

    # 计算最终匹配点的平均欧氏距离
    avg_distance = sum([m.distance for m in final_matches]) / len(final_matches)

    # 绘制内点
    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, inlier_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        matchColor=(0, 255, 0),  # 绿色线条
        singlePointColor=(255, 0, 0)  # 蓝色关键点
    )
    # 绘制外点
    cv2.drawMatches(
        img1, kp1, img2, kp2, outlier_matches, img_matches,
        matchColor=(0, 0, 255),  # 红色线条
        singlePointColor=(255, 0, 0),  # 蓝色关键点
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG
    )

    # 控制显示窗口大小
    h, w = img_matches.shape[:2]
    scale = min(window_size[0] / w, window_size[1] / h)
    resized_img = cv2.resize(img_matches, (int(w * scale), int(h * scale)))
    # 显示结果
    cv2.namedWindow('Top 10 Matches', cv2.WINDOW_NORMAL)
    cv2.imshow('Top 10 Matches', resized_img)

    cv2.imwrite('sift_match.png', resized_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 打印结果
    # print(f"Total matches found: {len(matches)}")
    # print(f"Average descriptor distance of top 10 matches: {avg_distance:.2f}")
    # print("\nTop 10 matched point pairs (pixel coordinates and distances):")
    # for i, mp in enumerate(matched_points):
    #     print(f"Match {i + 1}: Point1={mp['point1']}, Point2={mp['point2']}, Distance={mp['pixel_distance']:.2f}px")

    return {
        'matches': matches,
        'top10_matches': top10_matches,
        'inlier_matches': inlier_matches,
        'outlier_matches': outlier_matches,
        'matched_points': matched_points,  # 新增：包含坐标和距离的匹配点对
        'avg_descriptor_distance': avg_distance,  # 描述符距离的平均值
        'avg_pixel_distance': np.mean([mp['pixel_distance'] for mp in matched_points]),  # 像素距离的平均值
        'keypoints1': kp1,
        'keypoints2': kp2
    }

from typing import Tuple, Union
def resize_and_pad(image: np.ndarray,
                   K: np.ndarray,
                   target_size: Tuple[int, int],
                   pad_value: Union[int, Tuple[int, int, int]] = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    调整图像大小并添加填充

    Args:
        image: 输入图像 (H, W, C) 或 (H, W)
        K: 相机内参矩阵 (3x3)
        target_size: 目标尺寸 (width, height)
        pad_value: 填充值，对于彩色图像可以是三元组 (B, G, R)

    Returns:
        padded_image: 调整大小并填充后的图像
        K_new: 更新后的相机内参矩阵
    """
    target_width, target_height = target_size
    h, w = image.shape[:2]

    # 计算缩放比例
    scale = min(target_width / w, target_height / h)

    # 计算新尺寸
    new_w = int(w * scale)
    new_h = int(h * scale)

    # 调整图像大小
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 计算填充
    pad_w = target_width - new_w
    pad_h = target_height - new_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2

    # 添加填充
    padded_image = cv2.copyMakeBorder(
        resized_image,
        pad_top,
        pad_h - pad_top,
        pad_left,
        pad_w - pad_left,
        cv2.BORDER_CONSTANT,
        value=pad_value
    )

    # 更新相机内参矩阵
    K_new = K.copy()
    K_new[0, 0] *= scale  # fx
    K_new[1, 1] *= scale  # fy
    K_new[0, 2] = (K[0, 2] - 0.5) * scale + pad_left + 0.5  # cx
    K_new[1, 2] = (K[1, 2] - 0.5) * scale + pad_top + 0.5  # cy

    return padded_image, K_new

# 使用示例
if __name__ == "__main__":
    image1_path = f"/home/jd/wangzhiwei225_data/bev数据/xm_xb_01/handeye_data_0105/home/jd/RobotTaskSupervisor2/SourceVision/handeye_data/1/rgb_227.png"  # 替换为你的第一张图像路径
    image2_path = f"/home/jd/wangzhiwei225_data/bev数据/xm_xb_01/handeye_data_0105/home/jd/RobotTaskSupervisor2/SourceVision/handeye_data/1/rgb_228.png"  # 替换为你的第二张图像路径
    # K = np.array([[500.0, 0, 320.0],
    #               [0, 500.0, 180.0],
    #               [0, 0, 1.0]])
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    # img1, K = resize_and_pad(img1, K, (img2.shape[1], img2.shape[0]), 0)
    cv2.imwrite("track.png", img1)
    cv2.imwrite("seg.png",img2)
    result = feature_matching_with_bf("track.png", "seg.png")

    if result is not None:
        print(f"Average pixel distance: {result['avg_pixel_distance']:.2f}px")
    # root_path="/home/jd/wangzhiwei225_data/data_tw_5_1_20250805/data/images/4"
    # pathes=os.listdir("/home/jd/wangzhiwei225_data/data_tw_5_1_20250805/data/images/4")
    # rgb_pathes=[]
    # for path in pathes:
    #     if "package-recover_4_192.168.1.33_192.168.1.33_2_RGB" in path:
    #         rgb_pathes.append(path)
    # with open("/home/jd/wangzhiwei225_data/data_tw_5_1_20250805/data/images/offset_error_rgb.txt", "w") as file:
    #     for i in range(len(rgb_pathes)):
    #         source_path=os.path.join(root_path, rgb_pathes[i])
    #         for j in range(i+1,len(rgb_pathes)):
    #             target_path=os.path.join(root_path, rgb_pathes[j])
    #             result = feature_matching_with_bf(source_path, target_path)
    #             if result is not None:
    #                 print(f"Average pixel distance: {result['avg_pixel_distance']:.2f}px")
    #                 file.write(f"{rgb_pathes[i]} {rgb_pathes[j]} Average pixel distance: {result['avg_pixel_distance']:.2f}px\n")
    # file.close()
