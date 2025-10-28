# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------
# File Name:        stereo_vision.py
# Author:           wzw
# Version:          0.1
# Created:          2025/10/28
# Description:      立体视觉
# ------------------------------------------------------------------
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def generate_depth_map_and_pointcloud(left_rectified, right_rectified, Q):
    # 转换为灰度图
    if len(left_rectified.shape) == 3:
        left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
    else:
        left_gray = left_rectified

    if len(right_rectified.shape) == 3:
        right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
    else:
        right_gray = right_rectified

    # 设置立体匹配参数

    # 越低越零碎
    window_size = 7
    min_disp = 0
    # 越大精度越高
    num_disp = 4 * 16
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,  # 3 ~ 11
        P1=8 * window_size * window_size,
        P2=32 * window_size * window_size,
        disp12MaxDiff=100,
        preFilterCap=63,
        uniquenessRatio=10,  # 5 ~ 15
        speckleWindowSize=100,  # 50 ~ 200
        speckleRange=1  # 1 or 2
    )
    # 计算视差图

    disp_left_sgbm = stereo.compute(left_gray, right_gray).astype(np.float32)
    right_matcher = cv2.ximgproc.createRightMatcher(stereo)
    disp_right_sgbm = right_matcher.compute(right_gray, left_gray).astype(np.float32)
    # 接上例，添加 WLS 滤波
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
    # wls_filter.setLambda(8000.0)
    # wls_filter.setSigmaColor(1.5)
    filtered_disp = wls_filter.filter(disp_left_sgbm, left_rectified, disparity_map_right=disp_right_sgbm).astype(
        np.float32)
    filtered_disp = cv2.bilateralFilter(filtered_disp, 5, 75, 75)  # 双边滤波

    disp_final = filtered_disp / 16.0
    rectified = left_rectified
    print("Q matrix:\n", Q)

    # 计算3D点坐标
    points3D = cv2.reprojectImageTo3D(disp_final, Q, handleMissingValues=True)
    points3D = points3D
    # 提取深度图（mm单位）
    depth_map = points3D[:, :, 2] * 1000.0

    # 创建点云
    # 只保留有效视差的点
    mask = (disp_final > min_disp) & (disp_final < num_disp)
    points = points3D[mask]
    colors = rectified[mask] if len(rectified.shape) == 3 else \
        cv2.cvtColor(rectified, cv2.COLOR_GRAY2BGR)[mask]

    # 可选：进一步过滤深度范围（例如0.5m-10m）
    z_mask = (points[:, 2] > 2) & (points[:, 2] < 4.5)
    points = points[z_mask]
    colors = colors[z_mask]

    # 创建Open3D点云
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors[:, [2, 1, 0]] / 255.0)  # BGR to RGB

    return depth_map, cloud, disp_final


def visualize_pointcloud(cloud):
    # 可视化点云
    o3d.visualization.draw_geometries([cloud],
                                      window_name="Point Cloud",
                                      width=800,
                                      height=600)
    o3d.io.write_point_cloud(
        "/home/jd/wangzhiwei225/JDCode/RobotTaskSupervisor/SourceVision/ETH/point_cloud.pcd",
        cloud)


def visualize_depth_map(depth_map):
    plt.imshow(depth_map, cmap='jet')
    plt.colorbar()
    plt.title("Depth Map")
    plt.show()


def visualize_disparity(disp):
    plt.imshow(disp, cmap='jet')
    plt.colorbar()
    plt.title("Disparity Map")
    plt.show()


# 示例用法
if __name__ == "__main__":
    # 假设你已经有了校正后的左右图像和Q矩阵
    left_rectified = cv2.imread(
        "/home/jd/wangzhiwei225/JDCode/RobotTaskSupervisor/SourceVision/ETH/rectified_left.jpg")
    right_rectified = cv2.imread(
        "/home/jd/wangzhiwei225/JDCode/RobotTaskSupervisor/SourceVision/ETH/rectified_right.jpg")
    Q = np.array([[1, 0, 0, -442.8679990768433],
                  [0, 1, 0, -718.6932213306427],
                  [0, 0, 0, 1786.079242249438],
                  [0, 0, 11.50990700047632, -0]
                  ]

                 )  # 示例Q矩阵
    depth_map, point_cloud, disparity = generate_depth_map_and_pointcloud(
        left_rectified, right_rectified, Q)

    # 可视化结果
    # visualize_depth_map(depth_map)
    visualize_disparity(disparity)
    visualize_pointcloud(point_cloud)
