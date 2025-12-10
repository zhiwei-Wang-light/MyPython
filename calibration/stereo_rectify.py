import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from PIL import Image

c1_images_names = sorted(
    glob.glob('/home/jd/wangzhiwei225_data/标定数据/标定数据_XM_LC1_Left/标定数据/格口相机1/handeye_data/gray_*.png'))
c2_images_names = sorted(
    glob.glob('/home/jd/wangzhiwei225_data/标定数据/标定数据_XM_LC1_Left/标定数据/格口相机1/handeye_data/rgb_*.png'))
print(c1_images_names)
print(c2_images_names)
c1_images = []
c2_images = []
for im1, im2 in zip(c1_images_names, c2_images_names):
    _im = cv2.imread(im1, 1)
    c1_images.append(_im)

    _im = cv2.imread(im2, 1)
    c2_images.append(_im)


def find_corners(image_names, rotate=True):
    images = []
    for imname in image_names:
        im = cv2.imread(imname, 1)
        images.append(im)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    rows = 7  # number of checkerboard rows.
    columns = 6  # number of checkerboard columns.

    # Pixel coordinates of checkerboards
    imgpoints = []  # 2d points in image plane.
    valid_indices = []  # 存储成功检测到角点的图像索引

    for i, frame in enumerate(images):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # find the checkerboard
        ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)

        if ret == True:
            # Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)

            # opencv can attempt to improve the checkerboard coordinates
            corners = cv2.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            imgpoints.append(corners)
            valid_indices.append(i)  # 记录成功检测的图像索引

    return imgpoints, valid_indices


# 获取两个相机的角点数据和有效索引
imgpoints1, valid_indices1 = find_corners(c1_images_names, False)
imgpoints2, valid_indices2 = find_corners(c2_images_names, False)
print(valid_indices1)
print(valid_indices2)
# 找出两个相机都成功检测到角点的共同索引
common_indices = set(valid_indices1) & set(valid_indices2)
print(common_indices)
# 只保留共同有效的角点
imgpoints1_common = [imgpoints1[valid_indices1.index(i)] for i in common_indices]
imgpoints2_common = [imgpoints2[valid_indices2.index(i)] for i in common_indices]

# 将角点坐标转换为float32格式
points1all = np.array([pt.ravel() for pts in imgpoints1_common for pt in pts], dtype=np.float32)
points2all = np.array([pt.ravel() for pts in imgpoints2_common for pt in pts], dtype=np.float32)
print(points1all.shape)
print(points2all.shape)
# 计算基础矩阵
F, mask = cv2.findFundamentalMat(points1all, points2all, cv2.FM_RANSAC)

# 图像大小
img_size = (640, 480)

img1 = cv2.imread(
    '/home/jd/wangzhiwei225_data/标定数据/标定数据_XM_LC1_Left/标定数据/格口相机1/handeye_data/gray_0.png', 1)
img2 = cv2.imread('/home/jd/wangzhiwei225_data/标定数据/标定数据_XM_LC1_Left/标定数据/格口相机1/handeye_data/rgb_0.png',
                  1)


gray_left = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 找到棋盘格角点
ret_left, corners_left = cv2.findChessboardCorners(gray_left, (7, 6), None)
ret_right, corners_right = cv2.findChessboardCorners(gray_right, (7, 6), None)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

print(ret_left, ret_right)

if ret_left and ret_right:
    # 提高角点的精度
    corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
    corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

# 将角点坐标转换为float32格式
points1 = np.array([pt[0] for pt in corners_left], dtype=np.float32).reshape(-1, 2)
points2 = np.array([pt[0] for pt in corners_right], dtype=np.float32).reshape(-1, 2)

# 进行立体校正
retval, H1, H2 = cv2.stereoRectifyUncalibrated(points1, points2, F, img_size)


if retval:
    # 计算校正映射
    img1_rectified = cv2.warpPerspective(img1, H1, img_size)
    img2_rectified = cv2.warpPerspective(img2, H2, img_size)

    # 显示校正后的图像
    cv2.imwrite('Rectified Image 1.png', img1_rectified)
    cv2.imwrite('Rectified Image 2.png', img2_rectified)
else:
    print("立体校正失败")

plt.figure(figsize=(20, 20))

for i in range(0, 1):  # 以第一对图片为例
    im_L = Image.fromarray(img1_rectified)  # numpy 转 image类
    im_R = Image.fromarray(img2_rectified)  # numpy 转 image 类

    width = im_L.size[0] * 2
    height = im_L.size[1]

    img_compare = Image.new('RGBA', (width, height))
    img_compare.paste(im_L, box=(0, 0))
    img_compare.paste(im_R, box=(640, 0))

    # 在已经极线对齐的图片上均匀画线
    for i in range(1, 20):
        len = 480 / 20
        plt.axhline(y=i * len, color='r', linestyle='-')
    plt.imshow(img_compare)
    plt.savefig('epipolar_lines_ro.png', bbox_inches='tight', pad_inches=0)
    plt.show()
