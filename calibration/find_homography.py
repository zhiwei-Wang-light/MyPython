import cv2
import numpy as np
import matplotlib.pyplot as plt


# ---------- Step 1: 手动选点 ----------
def get_points_from_images(img1, img2, num_points=4):
    """
    手动选择两张图像中的对应点
    左图为 image1，右图为 image2
    """
    points_img1 = []
    points_img2 = []

    def click_event_img1(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points_img1) < num_points:
            points_img1.append((x, y))
            cv2.circle(temp_img1, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Image 1 - Select Points', temp_img1)

    def click_event_img2(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points_img2) < num_points:
            points_img2.append((x, y))
            cv2.circle(temp_img2, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Image 2 - Select Points', temp_img2)

    # 显示图像 1
    temp_img1 = img1.copy()
    cv2.imshow('Image 1 - Select Points', temp_img1)
    cv2.setMouseCallback('Image 1 - Select Points', click_event_img1)

    print(f"请在 Image 1 中依次选择 {num_points} 个对应点")
    while len(points_img1) < num_points:
        cv2.waitKey(1)

    cv2.destroyWindow('Image 1 - Select Points')

    # 显示图像 2
    temp_img2 = img2.copy()
    cv2.imshow('Image 2 - Select Points', temp_img2)
    cv2.setMouseCallback('Image 2 - Select Points', click_event_img2)

    print(f"请在 Image 2 中依次选择与 Image 1 对应的 {num_points} 个点")
    while len(points_img2) < num_points:
        cv2.waitKey(1)

    cv2.destroyWindow('Image 2 - Select Points')

    return np.array(points_img1, dtype=np.float32), np.array(points_img2, dtype=np.float32)


# ---------- Step 2: 可视化变换关系 ----------
def visualize_homography(image1, image2, H):
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    corners1 = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]], dtype=np.float32)
    corners1_transformed = cv2.perspectiveTransform(corners1.reshape(1, -1, 2), H).reshape(-1, 2)

    img_vis = image2.copy()
    cv2.polylines(img_vis, [np.int32(corners1_transformed)], isClosed=True, color=(0, 255, 0), thickness=3)
    cv2.polylines(img_vis, [np.int32([[0, 0], [w2, 0], [w2, h2], [0, h2]])], isClosed=True, color=(0, 0, 255), thickness=2)

    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
    plt.title('�� 变换后的 Image1 边界，�� Image2 边界')
    plt.axis('off')
    plt.show()


# ---------- Step 3: 拼接融合 ----------
def stitch_images_with_homography(image1, image2, H, blend_ratio=0.5):
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    corners1 = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]], dtype=np.float32)
    corners1_transformed = cv2.perspectiveTransform(corners1.reshape(1, -1, 2), H).reshape(-1, 2)

    all_corners = np.vstack([corners1_transformed, [[0, 0], [w2, 0], [w2, h2], [0, h2]]])
    x_min, y_min = np.min(all_corners, axis=0).astype(int)
    x_max, y_max = np.max(all_corners, axis=0).astype(int)

    translation_x = -x_min if x_min < 0 else 0
    translation_y = -y_min if y_min < 0 else 0
    width = int(x_max - x_min) if x_min < 0 else x_max + translation_x
    height = int(y_max - y_min) if y_min < 0 else y_max + translation_y

    T = np.array([[1, 0, translation_x],
                  [0, 1, translation_y],
                  [0, 0, 1]], dtype=np.float64)

    H_adjusted = T @ H
    result = cv2.warpPerspective(image1, H_adjusted, (width, height))

    y_start, y_end = translation_y, translation_y + h2
    x_start, x_end = translation_x, translation_x + w2

    result[y_start:y_end, x_start:x_end] = cv2.addWeighted(
        result[y_start:y_end, x_start:x_end],
        blend_ratio,
        image2,
        1 - blend_ratio,
        0
    )

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Stitched Image using Manually Selected Points')
    plt.axis('off')
    plt.show()

    cv2.imwrite('stitched_manual_points.png', result)
    print("✅ 拼接完成，结果已保存为 stitched_manual_points.png")


# ---------- Step 4: 主程序 ----------
if __name__ == "__main__":
    image1_path = "/home/jd/wangzhiwei225_data/标定数据/20251112/track_14.jpeg"
    image2_path = "/home/jd/wangzhiwei225_data/标定数据/20251112/seg_14.png"

    img1 = cv2.imread(image1_path, 0)
    img2 = cv2.imread(image2_path, 0)

    # 手动选择对应点
    src_pts, dst_pts = get_points_from_images(img1, img2, num_points=4)

    print("选取的点对：")
    for i in range(len(src_pts)):
        print(f"点{i+1}: 图1 {src_pts[i]} -> 图2 {dst_pts[i]}")

    # 使用RANSAC计算单应矩阵
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    # H=cv2.getPerspectiveTransform(src_pts,dst_pts)
    print("\n计算得到的单应矩阵 H：\n", H)
    # 可视化H效果
    # visualize_homography(img1, img2, H)

    # 拼接结果
    stitch_images_with_homography(img1, img2, H, blend_ratio=0.5)
