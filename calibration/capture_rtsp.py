#!/usr/bin/env python3
"""
RTSP多相机同步采集与显示 - 简化版
支持多线程同时采集多个RTSP流
"""

import cv2
import numpy as np
import time
import sys
import threading
from collections import deque
import signal
import argparse


class RTSPSyncCapture:
    def __init__(self, urls, target_fps=30, buffer_size=1):
        """
        初始化多相机采集
        urls: RTSP URL列表
        target_fps: 目标帧率
        buffer_size: 缓冲区大小 (1=最低延迟)
        """
        self.urls = urls
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.buffer_size = buffer_size

        # 存储相机数据
        self.cameras = []
        self.running = False
        self.fps_stats = {}

        # 初始化相机信息
        for i, url in enumerate(urls):
            self.cameras.append({
                'id': i,
                'url': url,
                'name': f'Camera_{i + 1}',
                'cap': None,
                'frame': None,
                'fps': 0,
                'frame_count': 0,
                'last_time': time.time(),
                'status': 'pending',
                'lock': threading.Lock()
            })

    def _open_camera(self, url):
        """打开单个RTSP相机"""
        # 使用FFmpeg后端
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            # 尝试默认后端
            cap = cv2.VideoCapture(url)

        if not cap.isOpened():
            return None

        # 低延迟优化
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 清空缓冲区
        for _ in range(3):
            cap.grab()

        return cap

    def _capture_single(self, cam):
        """单个相机的采集线程"""
        cap = self._open_camera(cam['url'])

        if cap is None:
            cam['status'] = 'error'
            print(f"❌ 相机 {cam['name']} 打开失败")
            return

        cam['cap'] = cap
        cam['status'] = 'connected'
        print(f"✓ 相机 {cam['name']} 连接成功")

        # 采集循环
        while self.running:
            try:
                # grab + retrieve 分离降低延迟
                if cap.grab():
                    ret, frame = cap.retrieve()

                    if ret and frame is not None:
                        with cam['lock']:
                            cam['frame'] = frame.copy()
                            cam['status'] = 'streaming'
                            cam['frame_count'] += 1

                            # 计算FPS
                            now = time.time()
                            if now - cam['last_time'] >= 1.0:
                                cam['fps'] = cam['frame_count'] / (now - cam['last_time'])
                                cam['frame_count'] = 0
                                cam['last_time'] = now
                    else:
                        cam['status'] = 'error'
                else:
                    # 读取失败，尝试重连
                    cam['status'] = 'reconnecting'
                    time.sleep(1)

                    # 重新打开
                    cap.release()
                    new_cap = self._open_camera(cam['url'])
                    if new_cap:
                        cap = new_cap
                        cam['cap'] = cap
                        cam['status'] = 'streaming'
                        print(f"🔄 相机 {cam['name']} 重连成功")
                    else:
                        print(f"⚠️ 相机 {cam['name']} 重连失败")

            except Exception as e:
                print(f"⚠️ 相机 {cam['name']} 异常: {e}")
                time.sleep(0.1)

    def start(self):
        """启动所有相机采集"""
        if not self.urls:
            print("❌ 没有URL")
            return False

        print(f"\n📷 启动 {len(self.urls)} 个RTSP相机")
        print(f"🎯 目标帧率: {self.target_fps} FPS")
        print("-" * 50)

        self.running = True

        # 为每个相机创建采集线程
        threads = []
        for cam in self.cameras:
            t = threading.Thread(target=self._capture_single, args=(cam,))
            t.daemon = True
            t.start()
            threads.append(t)
            time.sleep(0.1)  # 错开连接时间

        return True

    def get_frames(self):
        """获取所有相机的最新帧（同步）"""
        frames = []
        for cam in self.cameras:
            with cam['lock']:
                if cam['frame'] is not None:
                    frames.append({
                        'frame': cam['frame'].copy(),
                        'name': cam['name'],
                        'fps': cam['fps'],
                        'status': cam['status']
                    })
                else:
                    # 无帧时的占位图
                    h, w = 480, 640
                    blank = np.zeros((h, w, 3), dtype=np.uint8)
                    status_text = cam['status'].upper()
                    cv2.putText(blank, status_text, (w // 2 - 80, h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    frames.append({
                        'frame': blank,
                        'name': cam['name'],
                        'fps': 0,
                        'status': cam['status']
                    })
        return frames

    def show_grid(self, cols=2, window_name='RTSP Cameras'):
        """网格显示所有相机"""
        if not self.running:
            print("❌ 相机未启动")
            return

        print(f"\n🖥️  开始显示 (网格 {cols}列)")
        print("   - 'q' 退出")
        print("   - 'r' 重连所有")
        print("   - 'f' 全屏切换\n")

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        display_fps = 0
        last_fps_time = time.time()

        while self.running:
            try:
                frames = self.get_frames()
                grid = self._make_grid(frames, cols)

                if grid is not None:
                    # 显示FPS
                    display_fps += 1
                    now = time.time()
                    if now - last_fps_time >= 1.0:
                        print(f"\r📊 显示FPS: {display_fps}", end='')
                        display_fps = 0
                        last_fps_time = now

                    cv2.imshow(window_name, grid)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    self.stop()
                    break
                elif key == ord('r'):
                    self.reconnect_all()
                elif key == ord('f'):
                    self._toggle_fullscreen(window_name)

            except Exception as e:
                print(f"\n⚠️ 显示错误: {e}")
                break

        cv2.destroyAllWindows()

    def _make_grid(self, frames, cols=2):
        """创建网格视图"""
        if not frames:
            return None

        n = len(frames)
        rows = (n + cols - 1) // cols

        # 每个格子大小
        cell_w, cell_h = 640, 480

        # 调整帧大小并添加信息
        cell_frames = []
        for data in frames:
            frame = data['frame']

            # 调整大小
            if frame.shape[:2] != (cell_h, cell_w):
                try:
                    frame = cv2.resize(frame, (cell_w, cell_h))
                except:
                    frame = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)

            # 添加信息栏
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (cell_w, 45), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            # 显示信息
            name = data['name'][:12]
            fps = data['fps']
            status = data['status']

            # 状态颜色
            color = (0, 255, 0) if status == 'streaming' else \
                (0, 255, 255) if status == 'reconnecting' else (0, 0, 255)

            cv2.putText(frame, name, (5, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"{fps:.1f} FPS", (5, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(frame, status.upper(), (cell_w - 110, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            cell_frames.append(frame)

        # 填充空白
        while len(cell_frames) < rows * cols:
            blank = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
            cv2.putText(blank, "EMPTY", (cell_w // 2 - 40, cell_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            cell_frames.append(blank)

        # 组合网格
        grid_rows = []
        for r in range(rows):
            start = r * cols
            row = np.hstack(cell_frames[start:start + cols])
            grid_rows.append(row)

        return np.vstack(grid_rows)

    def _toggle_fullscreen(self, window_name):
        """切换全屏"""
        try:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
        except:
            pass

    def reconnect_all(self):
        """重连所有相机"""
        print("\n🔄 重连所有相机...")
        for cam in self.cameras:
            if cam['cap']:
                try:
                    cam['cap'].release()
                except:
                    pass
            cam['cap'] = None
            cam['status'] = 'reconnecting'

        # 重新创建采集线程
        self.running = False
        time.sleep(0.5)
        self.running = True

        for cam in self.cameras:
            t = threading.Thread(target=self._capture_single, args=(cam,))
            t.daemon = True
            t.start()

    def stop(self):
        """停止所有相机"""
        self.running = False
        for cam in self.cameras:
            if cam['cap']:
                try:
                    cam['cap'].release()
                except:
                    pass
        cv2.destroyAllWindows()
        print("\n✅ 所有相机已关闭")


def main():
    parser = argparse.ArgumentParser(
        description='RTSP多相机同步采集工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 两个相机
   python capture_rtsp.py rtsp://192.168.1.10:554/user=admin_password=_channel=1_stream=0.sdp rtsp://192.168.1.11:554/user=admin_password=_channel=1_stream=0.sdp

        """
    )
    parser.add_argument('urls', type=str, nargs='+', help='RTSP URL列表')
    parser.add_argument('-c', '--cols', type=int, default=2, help='网格列数 (默认2)')
    parser.add_argument('-f', '--fps', type=int, default=30, help='目标帧率 (默认30)')
    parser.add_argument('-b', '--buffer', type=int, default=5, help='缓冲区大小 (默认1)')

    args = parser.parse_args()

    if not args.urls:
        print("❌ 请提供至少一个RTSP URL")
        return

    # 创建管理器
    capture = RTSPSyncCapture(
        urls=args.urls,
        target_fps=args.fps,
        buffer_size=args.buffer
    )

    # 启动采集
    if not capture.start():
        return

    # 显示网格
    capture.show_grid(cols=args.cols)


if __name__ == "__main__":
    def signal_handler(sig, frame):
        print("\n正在退出...")
        cv2.destroyAllWindows()
        sys.exit(0)


    signal.signal(signal.SIGINT, signal_handler)
    main()