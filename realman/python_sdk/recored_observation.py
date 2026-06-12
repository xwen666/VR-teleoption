import os
import csv
import cv2
import json
import time
import math
import struct
import shutil
import signal
import threading
from pathlib import Path

import numpy as np
import pyrealsense2 as rs
from datasets import Dataset, Features, Image, Sequence, Value
from Robotic_Arm.rm_robot_interface import *


# =========================
# 配置区
# =========================
SAVE_ROOT = Path("./lerobot_dataset")
TASK_TEXT = "pick up the cup"

# RealSense
CAMERA_NAME = "front"
IMG_W = 640
IMG_H = 480
CAM_FPS = 30
JPEG_QUALITY = 95
CAMERA_REQUIRED = False

# Robot
ROBOT_IP = "192.168.1.18"
ROBOT_PORT = 8080
ROBOT_STATE_HZ = 100.0

# Revo2 Hand
HAND_OBS_ENABLED = True
HAND_DEVICE_ID = 126
HAND_PORT = 1
HAND_BAUDRATE = 460800
HAND_TIMEOUT = 20
HAND_UNIT_MODE_REG = 937
HAND_MIN_POS_REG_START = 946
HAND_MAX_POS_REG_START = 952
HAND_ACTUAL_POS_REG_START = 2000
HAND_NUM_JOINTS = 6
HAND_CONFIG_REFRESH_SEC = 1.0
HAND_DEFAULT_MIN_DEG = [0.0] * HAND_NUM_JOINTS
HAND_DEFAULT_MAX_DEG = [59.0, 90.0, 81.0, 81.0, 81.0, 81.0]
HAND_JOINT_LABELS = [
    "thumb_flex",
    "thumb_aux",
    "index",
    "middle",
    "ring",
    "pinky",
]
HAND_OPEN_POS = [0, 0, 0, 0, 0, 0]
HAND_CLOSE_POS = [500, 500, 1000, 1000, 1000, 1000]
HAND_GRASP_POS = [400, 0, 1000, 1000, 1000, 1000]
HAND_DEFAULT_SPEED = 600

# 预览
SHOW_WINDOW = True
WINDOW_NAME = "RealSense Recording"

# 按键
KEY_QUIT = ord("q")
KEY_SAVE_NEXT = ord("d")   # 备用：d
KEY_RERECORD = ord("a")    # 备用：a
KEY_HAND_OPEN = ord("o")
KEY_HAND_CLOSE = ord("c")
KEY_HAND_GRASP = ord("g")

CHUNK_SIZE = 1000


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


def list_realsense_devices():
    ctx = rs.context()
    devices = []
    for dev in ctx.query_devices():
        info = {}
        for camera_info in [
            rs.camera_info.name,
            rs.camera_info.serial_number,
            rs.camera_info.product_id,
            rs.camera_info.usb_type_descriptor,
        ]:
            try:
                info[str(camera_info)] = dev.get_info(camera_info)
            except Exception:
                continue
        devices.append(info)
    return devices


def start_realsense_pipeline(pipeline: rs.pipeline, config: rs.config):
    devices = list_realsense_devices()
    if not devices:
        raise RuntimeError(
            "No RealSense device detected by librealsense. "
            "Please check whether the camera is physically connected, "
            "whether the container was started after the camera was plugged in, "
            "and whether the host can see the RealSense device."
        )

    try:
        return pipeline.start(config)
    except RuntimeError as exc:
        raise RuntimeError(
            "RealSense device was enumerated but the stream failed to start. "
            "This is often caused by USB bandwidth/cable/power issues, or by the "
            "container not getting the correct USB/video devices."
        ) from exc


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

        self.hand_ready = False
        self.hand_unit_mode = 0
        self.hand_min_deg = HAND_DEFAULT_MIN_DEG.copy()
        self.hand_max_deg = HAND_DEFAULT_MAX_DEG.copy()
        self.last_hand_config_refresh = 0.0
        self.last_hand_joint_rad = [0.0] * HAND_NUM_JOINTS
        self.last_action_ee_pose = [0.0] * 6
        self.last_action_hand_rad = [0.0] * HAND_NUM_JOINTS
        self.last_action_valid = False

    def connect(self):
        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.handle = self.arm.rm_create_robot_arm(self.ip, self.port)
        if self.handle.id == -1:
            raise RuntimeError(f"failed to connect robot arm at {self.ip}:{self.port}")
        print(f"[Robot] connected, handle id = {self.handle.id}")

        if HAND_OBS_ENABLED and self.handle.id != -1:
            hand_ret = self.arm.rm_set_modbus_mode(HAND_PORT, HAND_BAUDRATE, HAND_TIMEOUT)
            self.hand_ready = hand_ret == 0
            print(
                f"[Hand] rm_set_modbus_mode(port={HAND_PORT}, baudrate={HAND_BAUDRATE}, "
                f"timeout={HAND_TIMEOUT}) -> {hand_ret}"
            )
            if self.hand_ready:
                self._refresh_hand_config(force=True)
        self.seed_action_from_current_state()

    @staticmethod
    def _to_float_list(values, size: int, default: float = 0.0):
        padded = list(values) if isinstance(values, list) else []
        padded = (padded + [default] * size)[:size]
        return [float(v) for v in padded]

    @staticmethod
    def _deg_list_to_rad(values):
        return [math.radians(float(v)) for v in values]

    def _make_hand_rw_params(self, address: int, num: int | None = None):
        return rm_peripheral_read_write_params_t(
            port=HAND_PORT,
            address=address,
            device=HAND_DEVICE_ID,
            num=num,
        )

    def _decode_u16_registers(self, raw_data, num: int):
        if len(raw_data) == num:
            return [int(value) for value in raw_data]
        if len(raw_data) < num * 2:
            raise ValueError(f"expected {num} registers or {num * 2} bytes, got {len(raw_data)}")
        raw_bytes = bytes((value & 0xFF) for value in raw_data[: num * 2])
        return list(struct.unpack(f">{num}H", raw_bytes))

    def _read_hand_single_holding(self, address: int):
        return self.arm.rm_read_holding_registers(self._make_hand_rw_params(address))

    def _write_hand_single_holding(self, address: int, value: int):
        return self.arm.rm_write_single_register(self._make_hand_rw_params(address), value)

    def _write_hand_multiple_holding(self, address: int, values: list[int]):
        return self.arm.rm_write_registers(self._make_hand_rw_params(address, len(values)), values)

    def _read_hand_multiple_holding(self, address: int, num: int):
        ret, raw_data = self.arm.rm_read_multiple_holding_registers(
            self._make_hand_rw_params(address, num)
        )
        if ret != 0:
            values = []
            for offset in range(num):
                single_ret, single_value = self._read_hand_single_holding(address + offset)
                if single_ret != 0:
                    return ret, []
                values.append(single_value)
            return 0, values
        return ret, self._decode_u16_registers(raw_data, num)

    def _read_hand_multiple_input(self, address: int, num: int):
        ret, raw_data = self.arm.rm_read_multiple_input_registers(
            self._make_hand_rw_params(address, num)
        )
        if ret != 0:
            values = []
            for offset in range(num):
                single_ret, single_value = self.arm.rm_read_input_registers(
                    self._make_hand_rw_params(address + offset)
                )
                if single_ret != 0:
                    return ret, []
                values.append(single_value)
            return 0, values
        return ret, self._decode_u16_registers(raw_data, num)

    def set_hand_unit_mode(self, mode: int = 0):
        if not self.hand_ready:
            return -1
        ret = self._write_hand_single_holding(HAND_UNIT_MODE_REG, mode)
        if ret == 0:
            self.hand_unit_mode = mode
        print(f"[Hand] set unit mode {mode} -> {ret}")
        return ret

    def set_single_hand_joint(self, joint_idx: int, position: int, speed: int = HAND_DEFAULT_SPEED):
        if not self.hand_ready:
            return -1
        ret = self._write_hand_multiple_holding(1055, [joint_idx, position, speed])
        print(f"[Hand] set joint {joint_idx} pos={position} speed={speed} -> {ret}")
        return ret

    def set_hand_positions(self, positions: list[int], speed: int = HAND_DEFAULT_SPEED):
        if not self.hand_ready:
            return -1
        if len(positions) != HAND_NUM_JOINTS:
            raise ValueError(f"expected {HAND_NUM_JOINTS} hand positions, got {len(positions)}")

        # 三代控制器 + 末端Modbus这里最稳的方式是逐指发 1055
        for joint_idx, position in enumerate(positions):
            ret = self.set_single_hand_joint(joint_idx, position, speed)
            if ret != 0:
                return ret
        self.last_action_hand_rad = self._normalized_positions_to_radians(positions)
        self.last_action_valid = True
        return 0

    def open_hand(self, speed: int = HAND_DEFAULT_SPEED):
        return self.set_hand_positions(HAND_OPEN_POS, speed)

    def close_hand(self, speed: int = HAND_DEFAULT_SPEED):
        return self.set_hand_positions(HAND_CLOSE_POS, speed)

    def grasp_hand(self, speed: int = HAND_DEFAULT_SPEED):
        return self.set_hand_positions(HAND_GRASP_POS, speed)

    def _refresh_hand_config(self, force: bool = False):
        if not self.hand_ready:
            return

        now = time.monotonic()
        if not force and now - self.last_hand_config_refresh < HAND_CONFIG_REFRESH_SEC:
            return

        unit_ret, unit_mode = self._read_hand_single_holding(HAND_UNIT_MODE_REG)
        if unit_ret == 0:
            self.hand_unit_mode = unit_mode
        else:
            print(f"[Hand] read unit mode error: ret={unit_ret}")

        min_ret, min_deg = self._read_hand_multiple_holding(HAND_MIN_POS_REG_START, HAND_NUM_JOINTS)
        if min_ret == 0:
            self.hand_min_deg = [float(value) for value in min_deg]
        else:
            print(f"[Hand] read min position config error: ret={min_ret}")

        max_ret, max_deg = self._read_hand_multiple_holding(HAND_MAX_POS_REG_START, HAND_NUM_JOINTS)
        if max_ret == 0:
            self.hand_max_deg = [float(value) for value in max_deg]
        else:
            print(f"[Hand] read max position config error: ret={max_ret}")

        self.last_hand_config_refresh = now

    def _normalized_to_degree(self, raw_value: int, joint_idx: int) -> float:
        min_deg = self.hand_min_deg[joint_idx]
        max_deg = self.hand_max_deg[joint_idx]
        return min_deg + (float(raw_value) / 1000.0) * (max_deg - min_deg)

    def _normalized_positions_to_radians(self, positions: list[int]):
        self._refresh_hand_config()
        return [
            math.radians(self._normalized_to_degree(raw_value, idx))
            for idx, raw_value in enumerate(positions)
        ]

    def read_hand_joint_radians(self):
        if not self.hand_ready:
            return "", "", self.last_hand_joint_rad.copy()

        self._refresh_hand_config()

        ret, raw_positions = self._read_hand_multiple_input(HAND_ACTUAL_POS_REG_START, HAND_NUM_JOINTS)
        if ret != 0:
            return ret, self.hand_unit_mode, self.last_hand_joint_rad.copy()

        hand_joints_rad = []
        for idx, raw_value in enumerate(raw_positions):
            if self.hand_unit_mode == 1:
                degree = raw_value / 10.0
            else:
                degree = self._normalized_to_degree(raw_value, idx)
            hand_joints_rad.append(math.radians(degree))

        self.last_hand_joint_rad = hand_joints_rad.copy()
        return ret, self.hand_unit_mode, hand_joints_rad

    def get_current_observation(self):
        ret, state = self.arm.rm_get_current_arm_state()
        arm_joint_deg = self._to_float_list(state.get("joint", []), 6)
        arm_joint_rad = self._deg_list_to_rad(arm_joint_deg)
        ee_pose = self._to_float_list(state.get("pose", []), 6)

        hand_ret = ""
        hand_unit_mode = self.hand_unit_mode
        hand_joint_rad = self.last_hand_joint_rad.copy()
        if HAND_OBS_ENABLED:
            hand_ret, hand_unit_mode, hand_joint_rad = self.read_hand_joint_radians()

        if not self.last_action_valid:
            self.last_action_ee_pose = ee_pose.copy()
            self.last_action_hand_rad = hand_joint_rad.copy()
            self.last_action_valid = True

        return {
            "robot_ret_code": ret,
            "arm_joint_deg": arm_joint_deg,
            "arm_joint_rad": arm_joint_rad,
            "ee_pose": ee_pose,
            "hand_ret_code": hand_ret,
            "hand_unit_mode": hand_unit_mode,
            "hand_joint_rad": hand_joint_rad,
            "observation_state": arm_joint_rad + hand_joint_rad,
            "action": self.get_action_vector(),
            "action_ee_pose": self.last_action_ee_pose.copy(),
            "action_hand_joint_rad": self.last_action_hand_rad.copy(),
        }

    def seed_action_from_current_state(self):
        obs = self.get_current_observation()
        self.last_action_ee_pose = obs["ee_pose"].copy()
        self.last_action_hand_rad = obs["hand_joint_rad"].copy()
        self.last_action_valid = True

    def get_action_vector(self):
        return self.last_action_ee_pose.copy() + self.last_action_hand_rad.copy()

    def movej_record(
        self, joint_deg: list[float], v: int, r: int = 0, connect: int = 0, block: int = 1
    ):
        ee_pose = self.arm.rm_algo_forward_kinematics(joint_deg, 1)
        self.last_action_ee_pose = self._to_float_list(ee_pose, 6)
        self.last_action_valid = True
        ret = self.arm.rm_movej(joint_deg, v, r, connect, block)
        print(f"[Robot] rm_movej target={joint_deg} -> {ret}")
        return ret

    def movel_record(
        self, pose: list[float], v: int, r: int = 0, connect: int = 0, block: int = 1
    ):
        self.last_action_ee_pose = self._to_float_list(pose, 6)
        self.last_action_valid = True
        ret = self.arm.rm_movel(pose, v, r, connect, block)
        print(f"[Robot] rm_movel target={pose} -> {ret}")
        return ret

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
            "err_len", "err_codes",
            "hand_ret_code", "hand_unit_mode",
            "hand_thumb_flex_rad", "hand_thumb_aux_rad", "hand_index_rad",
            "hand_middle_rad", "hand_ring_rad", "hand_pinky_rad",
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

                hand_ret = ""
                hand_unit_mode = ""
                hand_joints_rad = [""] * HAND_NUM_JOINTS
                if HAND_OBS_ENABLED:
                    hand_ret, hand_unit_mode, hand_joints_rad = self.read_hand_joint_radians()

                with self.lock:
                    if self.csv_writer is not None:
                        self.csv_writer.writerow([
                            ts,
                            ret,
                            joint[0], joint[1], joint[2], joint[3], joint[4], joint[5],
                            pose[0], pose[1], pose[2], pose[3], pose[4], pose[5],
                            err_len,
                            err_codes,
                            hand_ret,
                            hand_unit_mode,
                            hand_joints_rad[0], hand_joints_rad[1], hand_joints_rad[2],
                            hand_joints_rad[3], hand_joints_rad[4], hand_joints_rad[5],
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
            if self.hand_ready and self.arm is not None:
                hand_close_ret = self.arm.rm_close_modbus_mode(HAND_PORT)
                print(f"[Hand] rm_close_modbus_mode({HAND_PORT}) -> {hand_close_ret}")
            if self.arm is not None:
                self.arm.rm_delete_robot_arm()
                print("[Robot] disconnected")
        except Exception as e:
            print(f"[Robot] disconnect error: {e}")


class LeRobotDatasetRecorder:
    def __init__(self, save_root: Path, camera_name: str, task_text: str):
        self.save_root = save_root
        self.camera_name = camera_name
        self.task_text = task_text

        self.meta_dir = self.save_root / "meta"
        self.data_dir = self.save_root / "data"
        self.images_dir = self.save_root / "images"

        ensure_dir(self.save_root)
        ensure_dir(self.meta_dir)
        ensure_dir(self.data_dir)
        ensure_dir(self.images_dir)

        self.tasks_path = self.meta_dir / "tasks.jsonl"
        self.episodes_path = self.meta_dir / "episodes.jsonl"
        self.episodes_stats_path = self.meta_dir / "episodes_stats.jsonl"
        self.info_path = self.meta_dir / "info.json"

        self.task_to_index = self._load_tasks()
        self.task_index = self._ensure_task(self.task_text)

        self.next_episode_index = self._load_next_episode_index()
        self.next_global_index = self._load_next_global_index()

        self.current_episode_index = None
        self.current_rows = []
        self.frame_idx = 0
        self.started_ts = None
        self.current_image_dir = None
        self.current_has_camera = False
        self.has_any_camera_episode = False

    def _load_jsonl(self, path: Path):
        if not path.exists():
            return []
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _append_jsonl(self, path: Path, record: dict):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load_tasks(self):
        task_to_index = {}
        for item in self._load_jsonl(self.tasks_path):
            task_to_index[item["task"]] = item["task_index"]
        return task_to_index

    def _ensure_task(self, task: str):
        if task in self.task_to_index:
            return self.task_to_index[task]
        task_index = len(self.task_to_index)
        self.task_to_index[task] = task_index
        self._append_jsonl(
            self.tasks_path,
            {
                "task_index": task_index,
                "task": task,
            },
        )
        return task_index

    def _load_next_episode_index(self):
        return len(self._load_jsonl(self.episodes_path))

    def _load_next_global_index(self):
        total = 0
        for item in self._load_jsonl(self.episodes_path):
            total += int(item.get("length", 0))
        return total

    def _episode_name(self, episode_index: int):
        return f"episode_{episode_index:06d}"

    def _chunk_index(self, episode_index: int):
        return episode_index // CHUNK_SIZE

    def _rel_data_path(self, episode_index: int):
        return (
            Path("data")
            / f"chunk-{self._chunk_index(episode_index):03d}"
            / f"{self._episode_name(episode_index)}.parquet"
        )

    def _abs_data_path(self, episode_index: int):
        path = self.save_root / self._rel_data_path(episode_index)
        ensure_dir(path.parent)
        return path

    def _rel_image_dir(self, episode_index: int):
        return (
            Path("images")
            / f"chunk-{self._chunk_index(episode_index):03d}"
            / self.camera_name
            / self._episode_name(episode_index)
        )

    def _abs_image_dir(self, episode_index: int):
        path = self.save_root / self._rel_image_dir(episode_index)
        ensure_dir(path)
        return path

    def _build_features(self):
        features = {
            "timestamp": Value("float32"),
            "frame_index": Value("int64"),
            "episode_index": Value("int64"),
            "index": Value("int64"),
            "task_index": Value("int64"),
            "task": Value("string"),
            "next.done": Value("bool"),
            "observation.state": Sequence(Value("float32"), length=12),
            "observation.ee_pose": Sequence(Value("float32"), length=6),
            "observation.arm_joint_rad": Sequence(Value("float32"), length=6),
            "observation.hand_joint_rad": Sequence(Value("float32"), length=6),
            "action": Sequence(Value("float32"), length=12),
            "action.ee_pose": Sequence(Value("float32"), length=6),
            "action.hand_joint_rad": Sequence(Value("float32"), length=6),
            "robot_ret_code": Value("int64"),
            "hand_ret_code": Value("string"),
            "hand_unit_mode": Value("string"),
        }
        if self.current_has_camera:
            features["observation.images.front"] = Image()
        return Features(features)

    def _compute_vector_stats(self, rows: list[dict], key: str):
        values = np.asarray([row[key] for row in rows], dtype=np.float32)
        return {
            "mean": values.mean(axis=0).astype(float).tolist(),
            "min": values.min(axis=0).astype(float).tolist(),
            "max": values.max(axis=0).astype(float).tolist(),
        }

    def _write_info_json(self):
        total_episodes = self.next_episode_index
        total_chunks = max(1, math.ceil(max(total_episodes, 1) / CHUNK_SIZE))
        features = {
            "timestamp": {"dtype": "float32"},
            "frame_index": {"dtype": "int64"},
            "episode_index": {"dtype": "int64"},
            "index": {"dtype": "int64"},
            "task_index": {"dtype": "int64"},
            "task": {"dtype": "string"},
            "next.done": {"dtype": "bool"},
            "observation.state": {"dtype": "float32", "shape": [12]},
            "observation.ee_pose": {"dtype": "float32", "shape": [6]},
            "observation.arm_joint_rad": {"dtype": "float32", "shape": [6]},
            "observation.hand_joint_rad": {"dtype": "float32", "shape": [6]},
            "action": {"dtype": "float32", "shape": [12]},
            "action.ee_pose": {"dtype": "float32", "shape": [6]},
            "action.hand_joint_rad": {"dtype": "float32", "shape": [6]},
            "robot_ret_code": {"dtype": "int64"},
            "hand_ret_code": {"dtype": "string"},
            "hand_unit_mode": {"dtype": "string"},
        }
        if self.has_any_camera_episode:
            features["observation.images.front"] = {"dtype": "image", "shape": [IMG_H, IMG_W, 3]}

        info = {
            "codebase_version": "lerobot_v2.1_compatible_local",
            "robot_type": "realman_bm65b_revo2",
            "task": self.task_text,
            "total_episodes": total_episodes,
            "total_frames": self.next_global_index,
            "total_tasks": len(self.task_to_index),
            "total_videos": 0,
            "total_chunks": total_chunks,
            "chunks_size": CHUNK_SIZE,
            "fps": CAM_FPS if self.has_any_camera_episode else ROBOT_STATE_HZ,
            "splits": {"train": f"0:{total_episodes}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "image_path": (
                f"images/chunk-{{episode_chunk:03d}}/{self.camera_name}/"
                "episode_{episode_index:06d}/{frame_index:06d}.jpg"
            ),
            "features": features,
        }
        with open(self.info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

    def start_new_episode(self, has_camera: bool):
        self.current_episode_index = self.next_episode_index
        self.current_rows = []
        self.frame_idx = 0
        self.started_ts = now_ts()
        self.current_has_camera = has_camera
        self.has_any_camera_episode = self.has_any_camera_episode or has_camera
        self.current_image_dir = self._abs_image_dir(self.current_episode_index)
        print(f"[Episode] start {self._episode_name(self.current_episode_index)}")

    def record_step(self, image: np.ndarray | None, observation: dict):
        rel_image_path = None
        if image is not None:
            img_name = f"{self.frame_idx:06d}.jpg"
            abs_image_path = self.current_image_dir / img_name
            rel_image_path = self._rel_image_dir(self.current_episode_index) / img_name
            ok = cv2.imwrite(
                str(abs_image_path),
                image,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
            )
            if not ok:
                raise RuntimeError(f"failed to save image to {abs_image_path}")

        row = {
            "timestamp": float(now_ts() - self.started_ts),
            "frame_index": int(self.frame_idx),
            "episode_index": int(self.current_episode_index),
            "index": int(self.next_global_index + len(self.current_rows)),
            "task_index": int(self.task_index),
            "task": self.task_text,
            "next.done": False,
            "observation.state": [float(v) for v in observation["observation_state"]],
            "observation.ee_pose": [float(v) for v in observation["ee_pose"]],
            "observation.arm_joint_rad": [float(v) for v in observation["arm_joint_rad"]],
            "observation.hand_joint_rad": [float(v) for v in observation["hand_joint_rad"]],
            "action": [float(v) for v in observation["action"]],
            "action.ee_pose": [float(v) for v in observation["action_ee_pose"]],
            "action.hand_joint_rad": [float(v) for v in observation["action_hand_joint_rad"]],
            "robot_ret_code": int(observation["robot_ret_code"]),
            "hand_ret_code": str(observation["hand_ret_code"]),
            "hand_unit_mode": str(observation["hand_unit_mode"]),
        }
        if self.current_has_camera:
            row["observation.images.front"] = str(rel_image_path)

        self.current_rows.append(row)
        self.frame_idx += 1

    def _write_episode_dataset(self):
        if not self.current_rows:
            return
        self.current_rows[-1]["next.done"] = True
        features = self._build_features()
        dataset = Dataset.from_list(self.current_rows, features=features)
        dataset.to_parquet(str(self._abs_data_path(self.current_episode_index)))

    def _write_episode_metadata(self):
        if not self.current_rows:
            return
        length = len(self.current_rows)
        episode_meta = {
            "episode_index": self.current_episode_index,
            "episode_chunk": self._chunk_index(self.current_episode_index),
            "length": length,
            "task_index": self.task_index,
            "task": self.task_text,
            "data_path": str(self._rel_data_path(self.current_episode_index)),
        }
        self._append_jsonl(self.episodes_path, episode_meta)

        stats = {
            "episode_index": self.current_episode_index,
            "length": length,
            "task_index": self.task_index,
            "observation.state": self._compute_vector_stats(self.current_rows, "observation.state"),
            "action": self._compute_vector_stats(self.current_rows, "action"),
        }
        self._append_jsonl(self.episodes_stats_path, stats)

    def finalize_and_next(self, has_camera: bool):
        self._write_episode_dataset()
        self._write_episode_metadata()
        saved_name = self._episode_name(self.current_episode_index)
        saved_frames = len(self.current_rows)
        self.next_global_index += saved_frames
        self.next_episode_index += 1
        self._write_info_json()
        print(f"[Episode] saved {saved_name}, frames={saved_frames}")
        self.start_new_episode(has_camera=has_camera)

    def discard_and_restart(self, has_camera: bool):
        old_name = self._episode_name(self.current_episode_index)
        remove_dir_if_exists(self.current_image_dir)
        self.current_rows = []
        self.frame_idx = 0
        self.started_ts = now_ts()
        print(f"[Episode] discarded {old_name}, restart same episode")
        self.start_new_episode(has_camera=has_camera)

    def close_episode(self):
        self._write_info_json()


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
    if robot_logger.hand_ready:
        robot_logger.set_hand_unit_mode(0)

    # 相机
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, IMG_W, IMG_H, rs.format.bgr8, CAM_FPS)
    try:
        devices = list_realsense_devices()
        if devices:
            print(f"[Camera] detected RealSense devices: {devices}")
        profile = start_realsense_pipeline(pipeline, config)
        print(f"[Camera] RealSense started: {IMG_W}x{IMG_H} @ {CAM_FPS}Hz")
    except RuntimeError as e:
        print(f"[Camera] startup error: {e}")
        if CAMERA_REQUIRED:
            robot_logger.close()
            return
        print("[Camera] continue without camera because CAMERA_REQUIRED=False")
        profile = None

    # LeRobot 风格 episode 管理
    ep = LeRobotDatasetRecorder(SAVE_ROOT, CAMERA_NAME, TASK_TEXT)
    ep.start_new_episode(has_camera=profile is not None)

    print("[Control] 右方向键/d: 保存并进入下一条")
    print("[Control] 左方向键/a: 删除并重录本条")
    print("[Control] o: 张手 | c: 握拳 | g: 抓取预设")
    print("[Control] q: 退出")
    print("[RobotCtrl] Use robot_logger.movej_record()/movel_record() in code for arm actions")

    frame_counter = 0
    fps_t0 = time.time()

    try:
        while not stop_event.is_set():
            color_image = None
            if profile is not None:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                color_image = np.asanyarray(color_frame.get_data())

            observation = robot_logger.get_current_observation()
            ep.record_step(color_image, observation)

            # 预览
            if SHOW_WINDOW and color_image is not None:
                vis = color_image.copy()
                tip1 = f"{ep._episode_name(ep.current_episode_index)} | frame={ep.frame_idx}"
                tip2 = "RIGHT/d save | LEFT/a redo | o/c/g hand | q quit"

                cv2.putText(vis, tip1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(vis, tip2, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

                cv2.imshow(WINDOW_NAME, vis)
                key = cv2.waitKeyEx(1)

                if key == KEY_QUIT:
                    stop_event.set()
                    break

                elif is_right_key(key):
                    ep.finalize_and_next(has_camera=profile is not None)

                elif is_left_key(key):
                    ep.discard_and_restart(has_camera=profile is not None)

                elif key == KEY_HAND_OPEN:
                    robot_logger.open_hand()

                elif key == KEY_HAND_CLOSE:
                    robot_logger.close_hand()

                elif key == KEY_HAND_GRASP:
                    robot_logger.grasp_hand()

            if color_image is not None:
                frame_counter += 1
                now = time.time()
                if now - fps_t0 >= 1.0:
                    actual_fps = frame_counter / (now - fps_t0)
                    print(
                        f"[Camera] {ep._episode_name(ep.current_episode_index)} | "
                        f"frame={ep.frame_idx} | actual_fps={actual_fps:.2f}"
                    )
                    frame_counter = 0
                    fps_t0 = now
            else:
                time.sleep(max(0.0, 1.0 / ROBOT_STATE_HZ))

    finally:
        try:
            if profile is not None:
                pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        robot_logger.close()
        ep.close_episode()
        print("[Main] finished")


if __name__ == "__main__":
    main()
