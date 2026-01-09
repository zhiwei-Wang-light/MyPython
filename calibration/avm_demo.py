import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_points_from_images(img, num_points=4):
    points_img = []

    def click_event_img(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points_img) < num_points:
            points_img.append((x, y))
            cv2.circle(temp_img, (x, y), 5, (0, 255, 0), 1)
            cv2.namedWindow('Image 1 - Select Points', cv2.WINDOW_NORMAL)
            cv2.imshow('Image 1 - Select Points', temp_img)

    temp_img = img.copy()
    cv2.namedWindow('Image 1 - Select Points', cv2.WINDOW_NORMAL)
    cv2.imshow('Image 1 - Select Points', temp_img)
    cv2.setMouseCallback('Image 1 - Select Points', click_event_img)

    print(f"请在 Image 中依次选择 {num_points} 个对应点")
    while len(points_img) < num_points:
        cv2.waitKey(1)
    cv2.destroyWindow('Image 1 - Select Points')
    return np.array(points_img, dtype=np.float32)


def blend(srcImg, warpImg, savename=None):
    """
    图片融合，
    """
    rows, cols = srcImg.shape[:2]
    # 找到左右重叠区域
    global left, right
    for col in range(0, cols):
        if srcImg[:, col].any() and warpImg[:, col].any():
            left = col
            break
    for col in range(cols - 1, 0, -1):
        if srcImg[:, col].any() and warpImg[:, col].any():
            right = col
            break
    res = np.zeros([rows, cols, 3], np.uint8)
    alpha = np.zeros((rows, right - left, 3), dtype=np.float32)
    for row in range(0, rows):
        for col in range(left, right):
            if not srcImg[row, col].any():  # src不存在
                alpha[row, col - left, :] = 0
            elif not warpImg[row, col].any():  # warpImg 不存在
                alpha[row, col - left, :] = 1
            else:  # src 和warp都存在
                srcImgLen = float(abs(col - left))
                testImgLen = float(abs(col - right))
                alpha[row, col - left, :] = testImgLen / (srcImgLen + testImgLen)

    res[:, :left] = srcImg[:, :left]
    res[:, right:] = warpImg[:, right:]
    res[:, left:right] = np.clip(srcImg[:, left:right] * alpha + warpImg[:, left:right] * (np.ones_like(alpha) - alpha),
                                 0, 255)

    # opencv is bgr, matplotlib is rgb
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    if savename is not None:
        plt.imsave(savename, res)
    return res


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
    return result


if __name__ == "__main__":
    w = 640
    h = 480
    dst_w=200
    dst_h=175
    index=7
    image1 = cv2.imread(f"/home/jd/wangzhiwei225_data/标定数据/handeye_data/l1/{index}/rgb_0.png")
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    src_points = get_points_from_images(image1)
    dst_points = np.array([[w / 2 - dst_w, h / 2 - dst_h], [w / 2 - dst_w, h / 2 + dst_h], [w / 2 + dst_w, h / 2 + dst_h],
                           [w / 2 + dst_w, h / 2 - dst_h]])
    H1, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC)
    # H1 = np.asarray([[-1.64559430e+00 , 1.04196585e+00 , 6.96272633e+02],
    #  [-2.72306564e-01 ,-7.41326336e-01 , 5.30230623e+02],
    #  [-1.38169713e-03 , 3.92917695e-03 , 1.00000000e+00]])
    image1 = cv2.warpPerspective(image1, H1, (640, 480))
    cv2.imwrite(f"/home/jd/wangzhiwei225_data/标定数据/handeye_data/l1/{index}/rgb_0_bev.png", image1)
    print("H1", H1)
    image2 = cv2.imread(f"/home/jd/wangzhiwei225_data/标定数据/handeye_data/l2/{index}/rgb_0.png")
    image2=cv2.cvtColor(image2,cv2.COLOR_BGR2RGB)
    src_points = get_points_from_images(image2)
    dst_points = np.array([[w / 2 - dst_w, h / 2 - dst_h], [w / 2 - dst_w, h / 2 + dst_h], [w / 2 + dst_w, h / 2 + dst_h],
                           [w / 2 + dst_w, h / 2 - dst_h]])
    H2, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC)
    # H2 = np.asarray([[-4.53447324e-01, 6.80638245e+00, -3.64053534e+02],
    #                  [-2.19449574e+00, 2.01064803e+00, 8.74565621e+02],
    #                  [-8.45572974e-04, 9.036635e-03, 1.00000000e+00]])
    image2 = cv2.warpPerspective(image2, H2, (640, 480))
    cv2.imwrite(f"/home/jd/wangzhiwei225_data/标定数据/handeye_data/l2/{index}/rgb_0_bev.png", image2)
    print("H2", H2)
    image3 = cv2.imread(f"/home/jd/wangzhiwei225_data/标定数据/handeye_data/l3/{index}/rgb_0.png")
    image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
    src_points = get_points_from_images(image3)
    dst_points = np.array([[w / 2 - dst_w, h / 2 - dst_h], [w / 2 - dst_w, h / 2 + dst_h], [w / 2 + dst_w, h / 2 + dst_h],
                           [w / 2 + dst_w, h / 2 - dst_h]])
    H3, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC)
    # H3 = np.asarray([[-2.60249333e-16, -6.05263158e-01, 5.65000000e+02],
    #                  [9.21052632e-01, 3.78289474e-01, -1.57203947e+02],
    #                  [-7.77819720e-19, 2.63157895e-03, 1.00000000e+00]])
    image3 = cv2.warpPerspective(image3, H3, (640, 480))
    cv2.imwrite(f"/home/jd/wangzhiwei225_data/标定数据/handeye_data/l3/{index}/rgb_0_bev.png", image3)
    print("H3", H3)
    image23 = blend(image2, image3)
    # # image123 = add_weight(image13, image2, 1)
    cv2.namedWindow('avm', cv2.WINDOW_NORMAL)
    cv2.imshow('avm', image23)
    cv2.waitKey(0)
