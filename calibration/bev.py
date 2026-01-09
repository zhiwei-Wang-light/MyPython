import cv2
import numpy as np

def compute_homography(K, R, t, ground_z=0):
    """
    根据相机内外参计算单应性矩阵 H
    K: 3x3 相机内参矩阵
    R: 3x3 旋转矩阵
    t: 3x1 平移向量
    ground_z: 地平面高度（默认 0）
    """
    # 提取旋转矩阵的列
    r1 = R[:, 0]
    r2 = R[:, 1]
    r3 = R[:, 2]

    # 平面方程 n^T X + d = 0，这里假设地面法向量为 [0, 0, 1]
    n = np.array([0, 0, 1])
    d = ground_z - np.dot(n, t.flatten())

    # 单应性矩阵公式: H = K * (R - t * n^T / d) * K^-1
    H = K @ (np.column_stack((r1, r2, t.flatten())) - np.outer(t, n) / d) @ np.linalg.inv(K)
    return H

def bev_transform(image, H, output_size=(500, 800)):
    """
    执行 BEV 透视变换
    image: 输入图像
    H: 单应性矩阵
    output_size: 输出 BEV 图像大小 (宽, 高)
    """
    return cv2.warpPerspective(image, H, output_size)

if __name__ == "__main__":
    # 读取图像
    img = cv2.imread("road.jpg")
    if img is None:
        raise FileNotFoundError("未找到 road.jpg，请确保路径正确")

    # 示例相机内参（需根据实际标定结果替换）
    K = np.array([[1000, 0, 640],
                  [0, 1000, 360],
                  [0, 0, 1]], dtype=np.float32)

    # 示例外参（假设相机俯视一定角度）
    R = cv2.Rodrigues(np.array([0.2, 0, 0]))[0]  # 绕 X 轴旋转 0.2 弧度
    t = np.array([[0], [1.5], [5]])  # 相机位置（单位：米）

    # 计算单应性矩阵
    H = compute_homography(K, R, t)

    # 执行 BEV 变换
    bev_img = bev_transform(img, H)

    # 显示结果
    cv2.imshow("Original", img)
    cv2.imshow("BEV", bev_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
