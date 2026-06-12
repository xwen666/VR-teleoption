import os
import csv
import cv2
import json
import time
import shutil
import signal
import threading
from pathlib import Path

import numpy as np
import pyrealsense2 as rs
from Robotic_Arm.rm_robot_interface import *


# =========================
# 配置区
# =========================
SAVE_ROOT = Path("./dataset_raw")
TASK_TEXT = "pick up the cup"

# RealSense
CAMERA_NAME = "front"
IMG_W = 640
IMG_H = 480
CAM_FPS = 30
JPEG_QUALITY = 95

# Robot
ROBOT_IP = "192.168.1.18"
ROBOT_PORT = 8080
ROBOT_STATE_HZ = 100.0

# 预览
SHOW_WINDOW = True
WINDOW_NAME = "RealSense Recording"

# 按键
KEY_QUIT = ord("q")
KEY_SAVE_NEXT = ord("d")   # 备用：d
KEY_RERECORD = ord("a")    # 备用：a


# =========================
# 全局停止信号
# =========================
stop_event = threading.Event()


def now_ts():
    return time.time()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def remove_dir_if_exists(p: Path):
    if p.exists() and p.is_dir():
        shutil.rmtree(p)


def is_left_key(key: int) -> bool:
    # OpenCV 在不同系统上方向键键码可能不同
    return key in [2424832, 81, 65361, KEY_RERECORD]


def is_right_key(key: int) -> bool:
    return key in [2555904, 83, 65363, KEY_SAVE_NEXT]


class RMRobotLogger:
    def __init__(self, ip: str, port: int, hz: float):
        self.ip = ip
        self.port = port
        self.dt = 1.0 / hz

        self.arm = None
        self.handle = None

        self.thread = None
        self.csv_file = None
        self.csv_writer = None
        self.lock = threading.Lock()
        self.logging_stop = threading.Event()
        self.is_running = False

    def connect(self):
        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.handle = self.arm.rm_create_robot_arm(self.ip, self.port)
        print(f"[Robot] connected, handle id = {self.handle.id}")

    def start_episode(self, csv_path: Path):
        self.stop_episode()

        self.csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "timestamp",
            "ret_code",
            "joint_1_deg", "joint_2_deg", "joint_3_deg",
            "joint_4_deg", "joint_5_deg", "joint_6_deg",
            "pose_x_m", "pose_y_m", "pose_z_m",
            "pose_rx_rad", "pose_ry_rad", "pose_rz_rad",
            "err_len", "err_codes"
        ])
        self.csv_file.flush()

        self.logging_stop.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        self.is_running = True

    def _loop(self):
        while not stop_event.is_set() and not self.logging_stop.is_set():
            t0 = time.perf_counter()
            ts = now_ts()

            try:
                ret, state = self.arm.rm_get_current_arm_state()

                joint = state.get("joint", [])
                pose = state.get("pose", [])
                err = state.get("err", {})

                joint = (joint + [""] * 6)[:6]
                pose = (pose + [""] * 6)[:6]

                err_len = err.get("err_len", "")
                err_codes = err.get("err", [])
                if isinstance(err_codes, list):
                    err_codes = "|".join(map(str, err_codes))
                else:
                    err_codes = str(err_codes)

                with self.lock:
                    if self.csv_writer is not None:
                        self.csv_writer.writerow([
                            ts,
                            ret,
                            joint[0], joint[1], joint[2], joint[3], joint[4], joint[5],
                            pose[0], pose[1], pose[2], pose[3], pose[4], pose[5],
                            err_len,
                            err_codes
                        ])
                        self.csv_file.flush()

            except Exception as e:
                print(f"[Robot] read error: {e}")

            elapsed = time.perf_counter() - t0
            sleep_time = self.dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop_episode(self):
        if self.is_running:
            self.logging_stop.set()
            if self.thread is not None:
                self.thread.join(timeout=1.0)
            self.thread = None

            if self.csv_file is not None:
                self.csv_file.close()

            self.csv_file = None
            self.csv_writer = None
            self.is_running = False

    def close(self):
        self.stop_episode()
        try:
            if self.arm is not None:
                self.arm.rm_delete_robot_arm()
                print("[Robot] disconnected")
        except Exception as e:
            print(f"[Robot] disconnect error: {e}")


class EpisodeRecorder:
    def __init__(self, save_root: Path, camera_name: str):
        self.save_root = save_root
        self.camera_name = camera_name
        ensure_dir(self.save_root)

        self.episode_idx = 1
        self.ep_dir = None
        self.img_dir = None
        self.frames_csv_path = None
        self.meta_json_path = None

        self.frames_file = None
        self.frames_writer = None
        self.frame_idx = 0
        self.started_ts = None

    def episode_name(self):
        return f"episode_{self.episode_idx:06d}"

    def current_episode_dir(self):
        return self.save_root / self.episode_name()

    def start_new_episode(self):
        self.ep_dir = self.current_episode_dir()
        self.img_dir = self.ep_dir / self.camera_name
        self.frames_csv_path = self.ep_dir / "frames.csv"
        self.meta_json_path = self.ep_dir / "meta.json"

        remove_dir_if_exists(self.ep_dir)
        ensure_dir(self.img_dir)

        self.frames_file = open(self.frames_csv_path, "w", newline="", encoding="utf-8")
        self.frames_writer = csv.writer(self.frames_file)
        self.frames_writer.writerow(["frame_idx", "timestamp", "image_path"])
        self.frames_file.flush()

        self._save_meta()
        self.frame_idx = 0
        self.started_ts = now_ts()

        print(f"[Episode] start {self.episode_name()}")

    def _save_meta(self):
        meta = {
            "task": TASK_TEXT,
            "camera_name": CAMERA_NAME,
            "image_width": IMG_W,
            "image_height": IMG_H,
            "camera_fps": CAM_FPS,
            "robot_state_hz": ROBOT_STATE_HZ,
            "image_format": "jpg",
            "image_dir": self.camera_name,
            "timestamp_unit": "seconds",
            "robot_joint_unit": "degree",
            "robot_pose_unit": {
                "xyz": "meter",
                "rxyz": "radian"
            }
        }
        with open(self.meta_json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def save_frame(self, image: np.ndarray, ts: float):
        img_name = f"{self.frame_idx:06d}.jpg"
        img_path = self.img_dir / img_name

        ok = cv2.imwrite(
            str(img_path),
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )
        if not ok:
            print(f"[Camera] failed to save {img_path}")
            return False

        self.frames_writer.writerow([self.frame_idx, ts, f"{self.camera_name}/{img_name}"])
        self.frames_file.flush()
        self.frame_idx += 1
        return True

    def close_episode(self):
        if self.frames_file is not None:
            self.frames_file.close()
            self.frames_file = None
            self.frames_writer = None

    def finalize_and_next(self):
        self.close_episode()
        saved_name = self.episode_name()
        saved_frames = self.frame_idx
        print(f"[Episode] saved {saved_name}, frames={saved_frames}")
        self.episode_idx += 1
        self.start_new_episode()

    def discard_and_restart(self):
        old_name = self.episode_name()
        self.close_episode()
        remove_dir_if_exists(self.ep_dir)
        print(f"[Episode] discarded {old_name}, restart same episode")
        self.start_new_episode()


def signal_handler(sig, frame):
    print("\n[Main] stop signal received")
    stop_event.set()


def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    ensure_dir(SAVE_ROOT)

    # 机器人
    robot_logger = RMRobotLogger(
        ip=ROBOT_IP,
        port=ROBOT_PORT,
        hz=ROBOT_STATE_HZ
    )
    robot_logger.connect()

    # episode 管理
    ep = EpisodeRecorder(SAVE_ROOT, CAMERA_NAME)
    ep.start_new_episode()
    robot_logger.start_episode(ep.ep_dir / "robot.csv")

    # 相机
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, IMG_W, IMG_H, rs.format.bgr8, CAM_FPS)
    profile = pipeline.start(config)
    print(f"[Camera] RealSense started: {IMG_W}x{IMG_H} @ {CAM_FPS}Hz")

    print("[Control] 右方向键/d: 保存并进入下一条")
    print("[Control] 左方向键/a: 删除并重录本条")
    print("[Control] q: 退出")

    frame_counter = 0
    fps_t0 = time.time()

    try:
        while not stop_event.is_set():
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            ts = now_ts()
            color_image = np.asanyarray(color_frame.get_data())

            # 保存当前帧
            ep.save_frame(color_image, ts)

            # 预览
            if SHOW_WINDOW:
                vis = color_image.copy()
                tip1 = f"{ep.episode_name()} | frame={ep.frame_idx}"
                tip2 = "RIGHT/d save&next | LEFT/a redo | q quit"

                cv2.putText(vis, tip1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(vis, tip2, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

                cv2.imshow(WINDOW_NAME, vis)
                key = cv2.waitKeyEx(1)

                if key == KEY_QUIT:
                    stop_event.set()
                    break

                elif is_right_key(key):
                    robot_logger.stop_episode()
                    ep.finalize_and_next()
                    robot_logger.start_episode(ep.ep_dir / "robot.csv")

                elif is_left_key(key):
                    robot_logger.stop_episode()
                    ep.discard_and_restart()
                    robot_logger.start_episode(ep.ep_dir / "robot.csv")

            frame_counter += 1
            now = time.time()
            if now - fps_t0 >= 1.0:
                actual_fps = frame_counter / (now - fps_t0)
                print(f"[Camera] {ep.episode_name()} | frame={ep.frame_idx} | actual_fps={actual_fps:.2f}")
                frame_counter = 0
                fps_t0 = now

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        robot_logger.close()
        ep.close_episode()
        print("[Main] finished")


if __name__ == "__main__":
    main()