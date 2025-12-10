import cv2
import numpy as np
import matplotlib.pyplot as plt


# ---------- Step 1: 手动选点 ----------
def get_points_from_images(img1, img2, num_points=8):
    points_img1, points_img2 = [], []

    def click_event_img1(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points_img1) < num_points:
            points_img1.append((x, y))
            cv2.circle(temp_img1, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Image 1', temp_img1)

    def click_event_img2(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points_img2) < num_points:
            points_img2.append((x, y))
            cv2.circle(temp_img2, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Image 2', temp_img2)

    temp_img1 = img1.copy()
    temp_img2 = img2.copy()
    cv2.namedWindow('Image 1', cv2.WINDOW_NORMAL)
    cv2.imshow('Image 1', temp_img1)
    cv2.setMouseCallback('Image 1', click_event_img1)
    print(f"请在 Image 1 中选择 {num_points} 个点")
    while len(points_img1) < num_points:
        cv2.waitKey(1)
    cv2.destroyWindow('Image 1')
    cv2.namedWindow('Image 2', cv2.WINDOW_NORMAL)
    cv2.imshow('Image 2', temp_img2)
    cv2.setMouseCallback('Image 2', click_event_img2)
    print(f"请在 Image 2 中选择对应 {num_points} 个点")
    while len(points_img2) < num_points:
        cv2.waitKey(1)
    cv2.destroyWindow('Image 2')

    return np.array(points_img1, dtype=np.float32), np.array(points_img2, dtype=np.float32)


# ---------- Step 2: 极线矫正 ----------
def rectify_images(img1, img2, pts1, pts2):
    # 计算基础矩阵
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
    print("基础矩阵 F:\n", F)
    pts1_3d = np.concatenate((pts1, np.ones((8, 1))), axis=1)
    pts2_3d = np.concatenate((pts2, np.ones((8, 1))), axis=1)
    # 立体矫正得到变换矩阵
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    _, H1, H2 = cv2.stereoRectifyUncalibrated(
        pts1, pts2, F, imgSize=(w2, h2)
    )
    print("H1", H1)
    print("H2", H2)
    H1 = np.asarray([[-2.88278930e-04, 3.40304033e-04, 5.31401091e-01],
                     [-4.23444531e-04, -2.24536746e-04, 1.27041247e+00],
                     [-5.00483051e-08, -8.05506150e-08, 7.72145734e-04]])
    H2 = np.asarray([[1.25455950e+00, 6.66630628e-01, -9.94210591e+02],
                     [-1.58449363e-01, 1.04821435e+00, 1.44570530e+02],
                     [3.03501905e-04, 1.61270682e-04, 4.63372490e-01]])
    print((H1 @ pts1_3d.T).T / (H1 @ pts1_3d.T).T[:, 2:])
    print((H2 @ pts2_3d.T).T / (H2 @ pts2_3d.T).T[:, 2:])
    img1_rectified = cv2.warpPerspective(img1, H1, (w2, h2))
    img2_rectified = cv2.warpPerspective(img2, H2, (w2, h2))

    return img1_rectified, img2_rectified


# ---------- Step 3: 画平行线 ----------
def draw_parallel_lines(img, num_lines=10, color=(0, 255, 0)):
    # img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_color = img
    h, w, c = img.shape
    step = h // (num_lines + 1)
    for i in range(1, num_lines + 1):
        y = i * step
        img_color = cv2.line(img_color, (0, y), (w, y), color, 1)
    return img_color


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


# ---------- Step 4: 主程序 ----------
if __name__ == "__main__":
    image1_path = "/home/jd/wangzhiwei225_data/标定数据/merge/track_1.jpeg"
    image2_path = "/home/jd/wangzhiwei225_data/标定数据/merge/seg_1.png"
    K = np.array([[500.0, 0, 320.0],
                  [0, 500.0, 180.0],
                  [0, 0, 1.0]])
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    img1, K = resize_and_pad(img1, K, (img2.shape[1], img2.shape[0]), 0)
    # 手动选择对应点
    pts1, pts2 = get_points_from_images(img1, img2, num_points=8)

    # 极线矫正
    img1_rect, img2_rect = rectify_images(img1, img2, pts1, pts2)

    # 画平行线
    img1_lines = draw_parallel_lines(img1_rect, num_lines=15)
    img2_lines = draw_parallel_lines(img2_rect, num_lines=15)

    # 可视化
    plt.figure(figsize=(15, 7))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img1_lines, cv2.COLOR_BGR2RGB))
    plt.title('Rectified Image 1 with Parallel Lines')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(img2_lines, cv2.COLOR_BGR2RGB))
    plt.title('Rectified Image 2 with Parallel Lines')
    plt.axis('off')
    plt.show()
