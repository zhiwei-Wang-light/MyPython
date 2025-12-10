# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------
# File Name:        realsense_demo.py
# Author:           wzw
# Version:          0.1
# Created:          2025/10/28
# Description:      调用realsense相机
# ------------------------------------------------------------------
import pyrealsense2 as rs
import time
import numpy as np
import cv2 as cv2

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
time.sleep(0.1)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
time.sleep(0.1)
cfg = pipeline.start(config)
time.sleep(0.1)
dev = cfg.get_device()
time.sleep(0.1)
depth_sensor = dev.first_depth_sensor()
time.sleep(0.1)
align = rs.align(rs.stream.color)
time.sleep(0.1)
color_sensor = dev.first_color_sensor()
time.sleep(0.1)
profile = cfg.get_stream(rs.stream.color)
time.sleep(0.1)
intrinsics = profile.as_video_stream_profile().get_intrinsics()
fx = intrinsics.fx
fy = intrinsics.fy
cx = intrinsics.ppx
cy = intrinsics.ppy
intr = [fx, fy, cx, cy]
depth_scale = depth_sensor.get_depth_scale()
for i in range(10):
    time.sleep(0.03)
    unused_frames = pipeline.wait_for_frames()
frames = pipeline.wait_for_frames()
frames_align = align.process(frames)
color_frame = frames_align.get_color_frame()
depth_frame = frames_align.get_depth_frame()
color_image = np.asanyarray(color_frame.get_data())
depth_image = np.asanyarray(depth_frame.get_data())
# 创建两个独立窗口并显示图像
cv2.namedWindow('Image 1', cv2.WINDOW_NORMAL)  # 可调整窗口大小
cv2.imshow('Image 1', color_image)
cv2.imwrite("./0925_data/color_1.png", color_image)
cv2.namedWindow('Image 2', cv2.WINDOW_NORMAL)
cv2.imshow('Image 2', depth_image)
cv2.imwrite("./0925_data/depth_1.png", depth_image)

# 等待按键关闭窗口（按任意键关闭所有窗口）
cv2.waitKey(0)
cv2.destroyAllWindows()
