import cv2
import numpy as np
import open3d as o3d
import os


def read_depth_from_xml(depth_image_path):
    """
    从XML文件读取深度数据

    参数:
        depth_image_path: XML文件路径

    返回:
        depth_image: 深度图像数据
    """
    # depth_image = None
    fs = cv2.FileStorage(depth_image_path, cv2.FILE_STORAGE_READ)

    if fs.isOpened():
        node = fs.getNode("PointMap")
        depth_image = node.mat()
        print(f"深度数据形状: {depth_image.shape}")
    else:
        raise ValueError("无法打开深度图像文件")

    fs.release()
    # depth_image = cv2.imread(depth_image_path, -1)
    return depth_image


def read_camera_parameters(camera_intr_path):
    """
    从XML文件读取相机参数

    参数:
        camera_intr_path: 相机参数文件路径

    返回:
        intr_matrix: 相机内参矩阵
        dist_coeffs: 畸变系数
    """
    fs = cv2.FileStorage(camera_intr_path, cv2.FILE_STORAGE_READ)

    if fs.isOpened():
        intr_matrix = fs.getNode("Intrinsic").mat()
        dist_coeffs = fs.getNode("Distortion").mat()
        print(f"内参矩阵: {intr_matrix}")
        print(f"畸变系数: {dist_coeffs}")
    else:
        raise ValueError("无法打开相机参数文件")

    fs.release()
    return intr_matrix, dist_coeffs


def generate_point_cloud_from_rgb_depth(rgb_image_path, depth_image_path, output_ply_path,
                                        intr_matrix, dist_coeffs, depth_scale=1000.0):
    """
    从RGB和深度图生成彩色点云并保存为PLY文件

    参数:
        rgb_image_path: RGB图像路径
        depth_image_path: 深度图路径
        output_ply_path: 输出PLY文件路径
        intr_matrix: 相机内参矩阵
        dist_coeffs: 畸变系数
        depth_scale: 深度图的缩放因子
    """
    # 读取RGB图像
    rgb_image = cv2.imread(rgb_image_path)
    if rgb_image is None:
        raise ValueError("无法加载RGB图像")

    # 读取深度数据
    depth_image = read_depth_from_xml(depth_image_path)

    # 校正RGB图像畸变
    rgb_image = cv2.undistort(rgb_image, intr_matrix, dist_coeffs, None, intr_matrix)

    # 将BGR转换为RGB
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    # 提取内参
    fx, fy, cx, cy = intr_matrix[0, 0], intr_matrix[1, 1], intr_matrix[0, 2], intr_matrix[1, 2]

    # 获取图像尺寸
    height, width = depth_image.shape[:2]

    # 创建点云
    points = []
    colors = []

    for v in range(height):
        for u in range(width):
            # 获取深度值并转换为米
            depth = depth_image[v, u, 2] / depth_scale  # 假设Z通道包含深度信息
            if np.isnan(depth) or depth >= 6 or depth <= 0.0:
                continue  # 跳过无效深度

            # 计算3D坐标
            z = depth
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            points.append([x, y, z])
            colors.append(rgb_image[v, u] / 255.0)  # 归一化颜色

    # 转换为numpy数组
    points = np.array(points)
    colors = np.array(colors)

    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 可视化点云
    o3d.visualization.draw_geometries([pcd])

    # 保存点云
    o3d.io.write_point_cloud(output_ply_path, pcd)
    print(f"彩色点云已保存到 {output_ply_path}")


def generate_point_cloud_from_depth_only(depth_image_path, output_ply_path, depth_scale=1000.0):
    """
    仅从深度图生成点云并保存为PLY文件

    参数:
        depth_image_path: 深度图路径
        output_ply_path: 输出PLY文件路径
        depth_scale: 深度图的缩放因子
    """
    # 读取深度数据
    depth_image = read_depth_from_xml(depth_image_path)

    # 获取图像尺寸
    height, width, channel = depth_image.shape

    # 提取有效的3D点
    points = []
    for v in range(height):
        for u in range(width):
            depth_point = depth_image[v, u] / depth_scale
            # 检查点是否有效（非NaN且非零）
            if not np.any(np.isnan(depth_point)) and np.any(depth_point != 0):
                points.append(depth_point)

    # 转换为numpy数组
    points = np.array(points)

    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 可视化点云
    o3d.visualization.draw_geometries([pcd])

    # 保存点云
    o3d.io.write_point_cloud(output_ply_path, pcd)
    print(f"深度点云已保存到 {output_ply_path}")


def main():
    """
    主函数 - 示例使用
    """
    # 文件路径配置
    base_path = "/home/jd/wangzhiwei225/JDCode/RobotTaskSupervisor2/SourceVision/handeye_data/2/"
    target_path="/home/jd/wangzhiwei225/JDCode/RobotTaskSupervisor2/SourceVision/handeye_data/"
    rgb_image_path = os.path.join(base_path, "rgb_22.png")
    depth_image_path = os.path.join(base_path, "depth_22.xml")
    camera_intr_path = os.path.join(base_path, "TYRgbCameraParameters.xml")
    output_ply_path = os.path.join(target_path, "RGB2_22.pcd")

    # try:
    # 读取相机参数
    intr_matrix, dist_coeffs = read_camera_parameters(camera_intr_path)

    # 生成彩色点云
    generate_point_cloud_from_rgb_depth(
        rgb_image_path,
        depth_image_path,
        output_ply_path,
        intr_matrix,
        dist_coeffs,
        depth_scale=1000.0
    )

    # generate_point_cloud_from_depth_only("/home/jd/wangzhiwei225_data/标定数据/handeye_data/l3/6/depth_0.xml","/home/jd/wangzhiwei225_data/标定数据/handeye_data/l3/6/depth_0.pcd")

    # except Exception as e:
    #     print(f"处理过程中发生错误: {e}")


if __name__ == "__main__":
    main()
