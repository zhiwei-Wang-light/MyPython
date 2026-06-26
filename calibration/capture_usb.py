#!/usr/bin/env python3
"""
多USB相机同步采集与显示（多线程优化版 - Windows/Linux兼容）
支持通过udev稳定路径打开相机
"""

import cv2
import numpy as np
import time
import sys
import os
from pathlib import Path
import threading
from collections import deque
import signal
import platform
import re

# 根据操作系统选择不同的相机后端
IS_WINDOWS = platform.system() == 'Windows'
if IS_WINDOWS:
    CAMERA_BACKEND = cv2.CAP_DSHOW  # Windows使用DirectShow
else:
    CAMERA_BACKEND = cv2.CAP_V4L2  # Linux使用V4L2


class MultiUSBCameraManager:
    def __init__(self, target_fps=30, max_width=640, max_height=480):
        """
        初始化多相机管理器
        target_fps: 目标帧率
        max_width, max_height: 显示时缩放的最大尺寸
        """
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.max_width = max_width
        self.max_height = max_height
        self.cameras = []
        self.running = False
        self.fps_stats = {}
        self.display_fps = 0
        self.last_fps_time = time.time()
        self.is_windows = IS_WINDOWS
        self.caps = []
        self.frame_lock = threading.Lock()  # 帧数据锁
        self.capture_threads = []  # 采集线程列表

    def find_cameras(self):
        """查找所有USB相机（Windows兼容）"""
        cameras = []

        if self.is_windows:
            print("🔍 正在扫描Windows相机 (索引0-10)...")
            for index in range(10):
                try:
                    cap = cv2.VideoCapture(index, CAMERA_BACKEND)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            if fps <= 0:
                                fps = 30

                            name = self._get_windows_camera_name(index, cap)

                            cameras.append({
                                'index': index,
                                'name': name,
                                'width': width,
                                'height': height,
                                'fps': fps,
                                'path': f"camera_{index}",
                                'stable_path': None,
                                'stable_name': None
                            })
                            print(f"✓ 发现相机: 索引 {index} -> {name} ({width}x{height} @ {fps:.1f}fps)")
                    cap.release()
                except Exception as e:
                    continue
        else:
            print("🔍 正在扫描Linux相机 (通过udev稳定路径)...")
            by_path_dir = Path('/dev/v4l/by-path/')
            if by_path_dir.exists():
                for dev_link in by_path_dir.glob('*video*'):
                    try:
                        stable_name = dev_link.name
                        real_path = dev_link.resolve()
                        cap = cv2.VideoCapture(str(real_path), CAMERA_BACKEND)
                        if cap.isOpened():
                            ret, frame = cap.read()
                            if ret:
                                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                fps = cap.get(cv2.CAP_PROP_FPS)
                                if fps <= 0:
                                    fps = 30

                                index_match = re.search(r'video(\d+)', str(real_path))
                                index = int(index_match.group(1)) if index_match else -1
                                name = self._get_linux_device_name(real_path)

                                cameras.append({
                                    'path': str(real_path),
                                    'stable_path': str(dev_link),
                                    'stable_name': stable_name,
                                    'index': index,
                                    'name': name,
                                    'width': width,
                                    'height': height,
                                    'fps': fps
                                })
                                print(f"✓ 发现相机: {stable_name} -> {name} ({width}x{height} @ {fps:.1f}fps)")
                        cap.release()
                    except Exception as e:
                        continue

            if not cameras:
                print("  → 未在 /dev/v4l/by-path/ 找到相机，尝试扫描 /dev/video*...")
                video_devices = sorted(Path('/dev').glob('video*'))
                for dev in video_devices:
                    try:
                        cap = cv2.VideoCapture(str(dev), CAMERA_BACKEND)
                        if cap.isOpened():
                            ret, frame = cap.read()
                            if ret:
                                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                fps = cap.get(cv2.CAP_PROP_FPS)
                                if fps <= 0:
                                    fps = 30

                                index = int(dev.name.replace('video', ''))
                                name = self._get_linux_device_name(dev)
                                stable_path = self._find_stable_path_by_index(index)

                                cameras.append({
                                    'path': str(dev),
                                    'stable_path': stable_path,
                                    'stable_name': Path(stable_path).name if stable_path else None,
                                    'index': index,
                                    'name': name,
                                    'width': width,
                                    'height': height,
                                    'fps': fps
                                })
                                print(f"✓ 发现相机: {dev} -> {name} ({width}x{height} @ {fps:.1f}fps)")
                        cap.release()
                    except Exception as e:
                        continue

        return cameras

    def _find_stable_path_by_index(self, index):
        """根据video索引查找对应的稳定路径"""
        by_path_dir = Path('/dev/v4l/by-path/')
        if by_path_dir.exists():
            for dev_link in by_path_dir.glob('*video*'):
                try:
                    real_path = dev_link.resolve()
                    if f'video{index}' in str(real_path):
                        return str(dev_link)
                except:
                    continue
        return None

    def _get_windows_camera_name(self, index, cap):
        """Windows下获取相机名称"""
        try:
            name = cap.get(cv2.CAP_PROP_POS_MSEC)
            if name:
                return str(name)
        except:
            pass

        try:
            import win32com.client
            wmi = win32com.client.GetObject("winmgmts:")
            cameras = wmi.ExecQuery("SELECT * FROM Win32_PnPEntity WHERE ConfigManagerErrorCode = 0")
            for cam in cameras:
                if cam.Name and ("camera" in cam.Name.lower() or "webcam" in cam.Name.lower()):
                    return cam.Name
        except:
            pass

        return f"Camera {index}"

    def _get_linux_device_name(self, dev_path):
        """Linux下通过udev获取设备名称"""
        try:
            import subprocess
            result = subprocess.run(
                ['udevadm', 'info', '-q', 'property', '-n', str(dev_path)],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                vendor = ""
                model = ""
                for line in lines:
                    if line.startswith('ID_VENDOR='):
                        vendor = line.split('=', 1)[1].strip()
                    elif line.startswith('ID_MODEL='):
                        model = line.split('=', 1)[1].strip()
                if vendor and model:
                    return f"{vendor}_{model}"
                elif model:
                    return model
                elif vendor:
                    return vendor
        except:
            pass
        return dev_path.name

    def open_camera_by_stable_path(self, stable_path):
        """通过稳定路径打开相机"""
        if self.is_windows:
            print("⚠️ Windows不支持稳定路径，请使用索引")
            return False

        if not stable_path.startswith('/'):
            full_path = f"/dev/v4l/by-path/{stable_path}"
        else:
            full_path = stable_path

        print(f"📷 尝试打开相机: {full_path}")

        if not Path(full_path).exists():
            print(f"❌ 路径不存在: {full_path}")
            return False

        try:
            real_path = Path(full_path).resolve()
            print(f"  → 实际设备: {real_path}")
        except:
            print(f"❌ 无法解析路径: {full_path}")
            return False

        cap = cv2.VideoCapture(str(real_path), CAMERA_BACKEND)
        if not cap.isOpened():
            print(f"❌ 无法打开相机: {full_path}")
            return False

        ret, frame = cap.read()
        if not ret:
            print(f"❌ 无法从相机读取数据: {full_path}")
            cap.release()
            return False

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30

        index_match = re.search(r'video(\d+)', str(real_path))
        index = int(index_match.group(1)) if index_match else -1
        name = self._get_linux_device_name(real_path)

        cam_info = {
            'path': str(real_path),
            'stable_path': full_path,
            'stable_name': Path(full_path).name,
            'index': index,
            'name': name,
            'width': width,
            'height': height,
            'fps': fps
        }

        self.cameras.append(cam_info)
        print(f"✓ 成功打开相机: {name} ({width}x{height} @ {fps:.1f}fps)")
        cap.release()
        return True

    def _open_single_camera(self, cam):
        """打开单个相机并配置参数"""
        # 优先使用稳定路径
        if not self.is_windows and cam.get('stable_path'):
            open_path = cam['stable_path']
            if not open_path.startswith('/'):
                open_path = f"/dev/v4l/by-path/{open_path}"
            try:
                real_path = Path(open_path).resolve()
                cap = cv2.VideoCapture(str(real_path), CAMERA_BACKEND)
            except:
                cap = cv2.VideoCapture(cam['path'], CAMERA_BACKEND)
        else:
            if self.is_windows:
                cap = cv2.VideoCapture(cam['index'], CAMERA_BACKEND)
            else:
                cap = cv2.VideoCapture(cam['path'], CAMERA_BACKEND)

        if not cap.isOpened():
            return None

        # === 低延迟优化 ===
        try:
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        except:
            pass

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)
        except:
            pass

        try:
            codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            cap.set(cv2.CAP_PROP_FOURCC, codec)
        except:
            pass

        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        except:
            pass

        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_width <= 0 or actual_height <= 0:
            actual_width = 640
            actual_height = 480

        return cap, actual_width, actual_height

    def _capture_single_camera(self, cap_info):
        """
        单个相机的采集线程
        使用grab()和retrieve()分离降低延迟
        """
        cap = cap_info['cap']
        cam_id = cap_info['cam_id']

        while self.running:
            try:
                # grab()快速获取下一帧
                if cap.grab():
                    ret, frame = cap.retrieve()
                    if ret and frame is not None:
                        # 使用锁更新帧数据
                        with self.frame_lock:
                            cap_info['last_frame'] = frame
                            cap_info['frame_count'] += 1

                            # 计算FPS
                            current_time = time.time()
                            if current_time - cap_info['last_time'] >= 1.0:
                                fps = cap_info['frame_count'] / (current_time - cap_info['last_time'])
                                if cam_id not in self.fps_stats:
                                    self.fps_stats[cam_id] = deque(maxlen=30)
                                self.fps_stats[cam_id].append(fps)
                                cap_info['info']['current_fps'] = np.mean(self.fps_stats[cam_id]) if self.fps_stats[
                                    cam_id] else 0
                                cap_info['frame_count'] = 0
                                cap_info['last_time'] = current_time
                    else:
                        # 读取失败，显示No Signal
                        with self.frame_lock:
                            if cap_info.get('last_frame') is None:
                                h = cap_info.get('actual_height', 480)
                                w = cap_info.get('actual_width', 640)
                                blank = np.zeros((h, w, 3), dtype=np.uint8)
                                cv2.putText(blank, "No Signal", (50, h // 2),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                cap_info['last_frame'] = blank
                else:
                    # grab失败，短暂等待
                    time.sleep(0.001)

            except Exception as e:
                # 出错时短暂等待
                time.sleep(0.01)

    def start_capture(self, selected_cameras=None):
        """启动多相机同步采集"""
        # 如果cameras为空，先查找
        if not self.cameras:
            if selected_cameras is None:
                self.cameras = self.find_cameras()
            else:
                all_cams = self.find_cameras()
                self.cameras = []
                for sel in selected_cameras:
                    for cam in all_cams:
                        if (isinstance(sel, int) and cam['index'] == sel) or \
                                (isinstance(sel, str) and cam.get('stable_name') == sel):
                            self.cameras.append(cam)
                            break

        if not self.cameras:
            print("❌ 未找到任何USB相机！")
            return False

        print(f"\n📷 总共发现 {len(self.cameras)} 个相机")
        print(f"🎯 目标帧率: {self.target_fps} FPS")
        print(f"💻 操作系统: {'Windows' if self.is_windows else 'Linux'}")
        print("⚡ 启动多线程同步采集...\n")

        self.caps = []
        self.fps_stats = {}
        self.capture_threads = []

        # 打开所有相机
        for cam in self.cameras:
            cam_id = cam.get('stable_name', str(cam['index']))

            result = self._open_single_camera(cam)
            if result is None:
                print(f"⚠️  警告: 无法打开相机 {cam.get('stable_name', cam.get('path', cam.get('index')))}")
                continue

            cap, actual_width, actual_height = result

            cap_info = {
                'cap': cap,
                'info': cam,
                'frame_count': 0,
                'last_time': time.time(),
                'last_frame': None,
                'actual_width': actual_width,
                'actual_height': actual_height,
                'cam_id': cam_id
            }

            self.caps.append(cap_info)
            self.fps_stats[cam_id] = deque(maxlen=30)
            print(f"✓ 相机 {cam_id} 打开成功")

        if not self.caps:
            print("❌ 没有成功打开任何相机")
            return False

        self.running = True

        # 为每个相机创建独立的采集线程
        for i, cap_info in enumerate(self.caps):
            os.makedirs(f"data/camera_{i}", exist_ok=True)
            thread = threading.Thread(target=self._capture_single_camera, args=(cap_info,), daemon=True)
            thread.start()
            self.capture_threads.append(thread)
            # 错开启动时间，避免同时占用资源
            time.sleep(0.05)

        print(f"\n✅ 所有相机采集线程已启动 (共 {len(self.caps)} 个)")
        return True

    def get_frames(self):
        """
        获取所有相机的最新帧（同步）
        使用锁确保数据一致性
        """
        frames = []
        with self.frame_lock:
            for cap_info in self.caps:
                frame = cap_info.get('last_frame')
                info = cap_info['info']

                if frame is None:
                    h = cap_info.get('actual_height', 480)
                    w = cap_info.get('actual_width', 640)
                    blank = np.zeros((h, w, 3), dtype=np.uint8)
                    cv2.putText(blank, "Waiting...", (50, h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    frames.append({
                        'frame': blank,
                        'info': info,
                        'frame_count': cap_info['frame_count']
                    })
                else:
                    frames.append({
                        'frame': frame.copy(),
                        'info': info,
                        'frame_count': cap_info['frame_count']
                    })
        return frames

    def show_cameras(self, use_grid=True, grid_cols=2):
        """显示所有相机"""
        if not self.running:
            print("❌ 相机未启动")
            return

        print(f"\n🖥️  开始显示 ({'网格' if use_grid else '独立窗口'}视图)")
        print("   - 按 'q' 或 'ESC' 退出")
        print("   - 按 'r' 降低分辨率")
        print("   - 显示实时FPS")

        if use_grid:
            cv2.namedWindow('USB Cameras', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('USB Cameras', 800, 600)

            while self.running:
                try:
                    frames = self.get_frames()
                    grid = self._create_grid(frames, grid_cols)

                    if grid is not None:
                        self.display_fps += 1
                        current = time.time()
                        if current - self.last_fps_time >= 1.0:
                            print(f"\r📊 显示帧率: {self.display_fps} FPS", end='')
                            self.display_fps = 0
                            self.last_fps_time = current

                        cv2.imshow('USB Cameras', grid)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        self.stop()
                        break
                    elif key == ord('r'):
                        self._reduce_resolution()
                except Exception as e:
                    print(f"显示循环错误: {e}")
                    break

            cv2.destroyAllWindows()
        else:
            windows = []
            for i, cap_info in enumerate(self.caps):
                info = cap_info['info']
                window_name = f"Camera {i}: {info['name'][:20]}"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 640, 480)
                windows.append(window_name)

            while self.running:
                try:
                    frames = self.get_frames()

                    for i, frame_data in enumerate(frames):
                        if i < len(windows):
                            frame = frame_data['frame']
                            info = frame_data['info']

                            current_fps = info.get('current_fps', 0)
                            cv2.putText(frame, f"{current_fps:.1f} FPS", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                            stable_name = info.get('stable_name', '')
                            if stable_name:
                                cv2.putText(frame, f"{stable_name[:30]}", (10, 55),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                            cv2.imshow(windows[i], frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        self.stop()
                        break
                    elif key == ord('r'):
                        self._reduce_resolution()
                except Exception as e:
                    print(f"显示循环错误: {e}")
                    break

            cv2.destroyAllWindows()

    def _create_grid(self, cameras, grid_cols=2):
        """创建网格视图"""
        if not cameras:
            return None

        n_cams = len(cameras)
        rows = (n_cams + grid_cols - 1) // grid_cols

        target_w = 640
        target_h = 480

        resized_frames = []
        for i, cam in enumerate(cameras):
            frame = cam['frame']
            if frame is not None and cam['frame_count'] % 30 == 0:
                cv2.imwrite(f"data/camera_{i}/{cam['frame_count']}.jpg", frame)
            if frame is None or frame.size == 0:
                frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)

            try:
                resized = cv2.resize(frame, (target_w, target_h))
            except:
                resized = np.zeros((target_h, target_w, 3), dtype=np.uint8)

            info = cam['info']
            current_fps = info.get('current_fps', 0)

            overlay = resized.copy()
            cv2.rectangle(overlay, (0, 0), (target_w, 65), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, resized, 0.4, 0, resized)

            name = info['name'][:12] if len(info['name']) > 12 else info['name']
            cv2.putText(resized, f"{name}", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(resized, f"{current_fps:.1f} FPS", (5, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(resized, f"{info['width']}x{info['height']}", (5, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            stable_name = info.get('stable_name', '')
            if stable_name:
                short_name = stable_name[:20] + '...' if len(stable_name) > 20 else stable_name
                cv2.putText(resized, f"{short_name}", (5, 62),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, (200, 200, 200), 1)

            resized_frames.append(resized)

        while len(resized_frames) < rows * grid_cols:
            blank = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            cv2.putText(blank, "Empty", (target_w // 2 - 30, target_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            resized_frames.append(blank)

        grid_rows = []
        for r in range(rows):
            row_start = r * grid_cols
            row_frames = resized_frames[row_start:row_start + grid_cols]
            row = np.hstack(row_frames)
            grid_rows.append(row)

        return np.vstack(grid_rows)

    def _reduce_resolution(self):
        """降低所有相机的分辨率"""
        print("\n📉 降低分辨率以提高速度...")
        for cap_info in self.caps:
            try:
                cap_info['cap'].set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                cap_info['cap'].set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                print(f"  → 相机 {cap_info['cam_id']}: 降至 320x240")
            except:
                print(f"  → 相机 {cap_info['cam_id']}: 降低分辨率失败")

    def stop(self):
        """停止所有相机"""
        self.running = False

        # 等待所有采集线程结束
        if hasattr(self, 'capture_threads'):
            for thread in self.capture_threads:
                try:
                    thread.join(timeout=0.5)
                except:
                    pass

        # 释放相机资源
        if hasattr(self, 'caps'):
            for cap_info in self.caps:
                try:
                    cap_info['cap'].release()
                except:
                    pass

        cv2.destroyAllWindows()
        print("\n✅ 所有相机已关闭")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='多USB相机同步采集与显示工具')
    parser.add_argument('-g', '--grid', action='store_true', default=True,
                        help='使用网格视图（默认）')
    parser.add_argument('-c', '--cols', type=int, default=2,
                        help='网格列数（默认2）')
    parser.add_argument('-i', '--index', type=int, nargs='+',
                        help='指定相机索引号')
    parser.add_argument('-s', '--stable-path', type=str, nargs='+',
                        help='通过稳定路径打开相机')
    parser.add_argument('-f', '--fps', type=int, default=30,
                        help='目标帧率（默认30）')
    parser.add_argument('--window', action='store_true',
                        help='使用独立窗口模式（不推荐）')

    args = parser.parse_args()

    manager = MultiUSBCameraManager(target_fps=args.fps)

    if args.stable_path:
        print(f"📷 使用稳定路径打开相机...")
        success = True
        for stable_path in args.stable_path:
            if not manager.open_camera_by_stable_path(stable_path):
                success = False
                break

        if not success:
            print("❌ 打开相机失败")
            return

        if not manager.start_capture():
            print("❌ 启动相机采集失败")
            return
    else:
        all_cameras = manager.find_cameras()

        if not all_cameras:
            print("❌ 未找到任何USB相机")
            print("  提示: Windows下请确保相机已连接并安装了驱动")
            return

        print(f"\n📷 发现 {len(all_cameras)} 个相机:")
        for i, cam in enumerate(all_cameras):
            stable_info = f" [稳定路径: {cam.get('stable_name', 'N/A')}]" if cam.get('stable_name') else ""
            print(f"  [{i}] {cam['name']} ({cam['width']}x{cam['height']}){stable_info}")

        if not manager.start_capture(args.index):
            return

    manager.show_cameras(use_grid=not args.window, grid_cols=args.cols)


if __name__ == "__main__":
    def signal_handler(sig, frame):
        print("\n正在退出...")
        cv2.destroyAllWindows()
        sys.exit(0)


    signal.signal(signal.SIGINT, signal_handler)
    main()
