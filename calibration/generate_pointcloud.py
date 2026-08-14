import cv2
import numpy as np
import open3d as o3d
import os


def read_depth_from_xml(depth_image_path):
    fs = cv2.FileStorage(depth_image_path, cv2.FILE_STORAGE_READ)
    if fs.isOpened():
        node = fs.getNode("PointMap")
        depth_image = node.mat()
        print(f"深度数据形状: {depth_image.shape}")
    else:
        raise ValueError("无法打开深度图像文件")
    fs.release()
    return depth_image


def read_camera_parameters(camera_intr_path):
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
            if np.isnan(depth):
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
    # # 可视化点云
    # o3d.visualization.draw_geometries([pcd])
    # 保存点云
    o3d.io.write_point_cloud(output_ply_path, pcd)
    print(f"彩色点云已保存到 {output_ply_path}")


def generate_point_cloud_from_depth_only(depth_image_path, output_ply_path, depth_scale=1000.0):
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


import os
import argparse
import sys


# 假设这些是你的原有函数
# from your_module import read_camera_parameters, generate_point_cloud_from_rgb_depth, generate_point_cloud_from_depth_only


def parse_arguments():
    """
    解析命令行参数

    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description="点云生成工具 - 支持带纹理和不带纹理两种模式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 生成带纹理的点云（RGB+深度）
  python script.py -r /path/to/rgb.png -d /path/to/depth.xml -o output.pcd -c /path/to/camera.xml

  # 生成不带纹理的点云（仅深度）
  python script.py -d /path/to/depth.xml -o output.pcd --depth-only

  # 生成带纹理的点云，并指定深度缩放因子
  python script.py -r /path/to/rgb.png -d /path/to/depth.xml -o output.pcd -c camera.xml -s 5000.0
        """
    )

    # 必需参数
    parser.add_argument(
        '-d', '--depth',
        type=str,
        required=True,
        help='深度图像路径（支持 .xml 或 .png 格式）'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='输出点云文件路径（建议使用 .pcd 或 .ply 格式）'
    )

    # 模式选择（互斥组）
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--rgb',
        type=str,
        help='RGB图像路径（启用带纹理模式）'
    )
    mode_group.add_argument(
        '--depth-only',
        action='store_true',
        help='仅深度模式（生成不带纹理的点云）'
    )

    # 可选参数
    parser.add_argument(
        '-c', '--camera',
        type=str,
        default=None,
        help='相机参数XML文件路径（带纹理模式必需，深度模式可选）'
    )

    parser.add_argument(
        '-s', '--scale',
        type=float,
        default=1000.0,
        help='深度缩放因子，默认: 1000.0'
    )

    parser.add_argument(
        '--skip-distortion',
        action='store_true',
        help='跳过畸变校正（默认会进行校正）'
    )

    return parser.parse_args()


def validate_arguments(args):
    """
    验证参数的合法性

    Args:
        args: 命令行参数对象

    Raises:
        ValueError: 参数不合法时抛出
    """
    # 检查深度文件是否存在
    if not os.path.exists(args.depth):
        raise ValueError(f"深度文件不存在: {args.depth}")

    # 带纹理模式必须提供相机参数
    if args.rgb and not args.camera:
        raise ValueError("带纹理模式（--rgb）必须提供相机参数文件（-c）")

    # 带纹理模式检查RGB文件是否存在
    if args.rgb and not os.path.exists(args.rgb):
        raise ValueError(f"RGB文件不存在: {args.rgb}")

    # 检查相机参数文件是否存在（如果提供了）
    if args.camera and not os.path.exists(args.camera):
        raise ValueError(f"相机参数文件不存在: {args.camera}")

    # 检查输出目录是否存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        raise ValueError(f"输出目录不存在: {output_dir}")


def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()

    try:
        # 验证参数
        validate_arguments(args)

        # 打印配置信息
        print("=" * 50)
        print("点云生成配置:")
        print(f"  模式: {'带纹理 (RGB)' if args.rgb else '仅深度'}")
        print(f"  深度文件: {args.depth}")
        print(f"  输出文件: {args.output}")
        if args.rgb:
            print(f"  RGB文件: {args.rgb}")
        if args.camera:
            print(f"  相机参数: {args.camera}")
        print(f"  深度缩放因子: {args.scale}")
        print(f"  畸变校正: {'跳过' if args.skip_distortion else '启用'}")
        print("=" * 50)

        # 读取相机参数（如果需要）
        intr_matrix = None
        dist_coeffs = None
        if args.camera:
            # 假设你有这个函数
            intr_matrix, dist_coeffs = read_camera_parameters(args.camera)
            print(f"✓ 已读取相机参数: {args.camera}")

        # 根据模式生成点云
        if args.rgb:
            # 带纹理模式
            print("🔄 正在生成带纹理的点云...")
            generate_point_cloud_from_rgb_depth(
                rgb_image_path=args.rgb,
                depth_image_path=args.depth,
                output_ply_path=args.output,
                intr_matrix=intr_matrix,
                dist_coeffs=dist_coeffs,
                depth_scale=args.scale
            )
            print("✅ 带纹理点云生成完成！")
        else:
            # 仅深度模式
            print("🔄 正在生成不带纹理的点云...")
            generate_point_cloud_from_depth_only(
                depth_image_path=args.depth,
                output_ply_path=args.output,
                depth_scale=args.scale,
            )
            print("✅ 不带纹理点云生成完成！")

        print(f"📁 输出文件: {args.output}")
        return 0

    except Exception as e:
        print(f"❌ 处理过程中发生错误: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    exit(main())
