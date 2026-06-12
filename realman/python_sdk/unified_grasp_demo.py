#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import os
import signal
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


ROOT_DIR = Path(__file__).resolve().parents[1]
RM_API_PYTHON_DIR = ROOT_DIR / "RM_API2" / "Python"
if str(RM_API_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(RM_API_PYTHON_DIR))

from Robotic_Arm.rm_ctypes_wrap import rm_peripheral_read_write_params_t  # noqa: E402
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e  # noqa: E402

WINDOW_NAME = "Unified Grasp Demo"
SNAPSHOT_DIR = ROOT_DIR / "test" / "captures"

HOME_JOINTS = [0, 0, 0, 0, 0, 0]
PREGRASP_JOINTS = [-101.0, 44.0, -13.0, -166.0, -109.2, 179.7]
GRASP_OFFSET_A = [-89, -78, -15, -176, -105, 179.0]

HAND_PORT = 1
HAND_BAUDRATE = 460800
HAND_TIMEOUT = 20
HAND_UNIT_MODE_REG = 937
HAND_MIN_POS_REG_START = 946
HAND_MAX_POS_REG_START = 952
HAND_ACTUAL_POS_REG_START = 2000
HAND_NUM_JOINTS = 6
HAND_DEFAULT_MIN_DEG = [0.0] * HAND_NUM_JOINTS
HAND_DEFAULT_MAX_DEG = [59.0, 90.0, 81.0, 81.0, 81.0, 81.0]

HAND_CMD_HOME = [960, 600, 60, 600, 980, 600, 980, 600, 980, 600, 980, 600]
HAND_CMD_RELEASE = [50, 700, 820, 500, 110, 500, 120, 500, 120, 500, 110, 440]
HAND_CMD_GRASP = [160, 600, 900, 560, 480, 560, 480, 560, 480, 560, 360, 500]

OVERLAY_FONT = cv2.FONT_HERSHEY_SIMPLEX


def list_realsense_devices() -> list[dict[str, str]]:
    ctx = rs.context()
    devices: list[dict[str, str]] = []
    for dev in ctx.query_devices():
        info: dict[str, str] = {}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive BM65 + Revo2 + RealSense grasp demo."
    )
    parser.add_argument("--arm-ip", default="192.168.1.18", help="RealMan arm controller IP")
    parser.add_argument("--arm-port", type=int, default=8080, help="RealMan arm controller TCP port")
    parser.add_argument("--hand-id", type=int, default=126, help="Revo2 Modbus device id")
    parser.add_argument(
        "--skip-camera",
        action="store_true",
        help="Run arm + hand demo without starting RealSense",
    )
    parser.add_argument("--width", type=int, default=640, help="RealSense color width")
    parser.add_argument("--height", type=int, default=480, help="RealSense color height")
    parser.add_argument("--fps", type=int, default=30, help="RealSense color FPS")
    parser.add_argument(
        "--move-speed",
        type=int,
        default=20,
        help="Arm move speed percentage used by rm_movej/rm_movel",
    )
    parser.add_argument(
        "--telemetry-hz",
        type=float,
        default=5.0,
        help="How often to refresh arm/hand telemetry in the preview loop",
    )
    parser.add_argument(
        "--grasp-offset",
        nargs=6,
        type=float,
        default=GRASP_OFFSET_A,
        metavar=("DX", "DY", "DZ", "DRX", "DRY", "DRZ"),
        help="Linear grasp offset applied at pre-grasp pose before hand close",
    )
    return parser.parse_args()


@dataclass
class Telemetry:
    arm_ret_code: int | str = ""
    hand_ret_code: int | str = ""
    hand_unit_mode: int | str = ""
    arm_joint_deg: list[float | str] = field(default_factory=lambda: [""] * 6)
    arm_pose: list[float | str] = field(default_factory=lambda: [""] * 6)
    hand_joint_rad: list[float | str] = field(default_factory=lambda: [""] * HAND_NUM_JOINTS)
    last_update_wall_ts: float = 0.0


class UnifiedGraspDemo:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.robot: RoboticArm | None = None
        self.handle = None
        self.pipeline: rs.pipeline | None = None
        self.telemetry = Telemetry()
        self.stop_requested = False
        self.telemetry_interval = 1.0 / max(args.telemetry_hz, 0.1)
        self.last_telemetry_refresh = 0.0
        self.hand_ready = False
        self.hand_min_deg = HAND_DEFAULT_MIN_DEG.copy()
        self.hand_max_deg = HAND_DEFAULT_MAX_DEG.copy()
        self.hand_unit_mode = 0
        self.camera_started = False

    def connect(self):
        self._connect_arm()
        if self.args.skip_camera:
            print("[Camera] skip requested via --skip-camera")
        else:
            self._connect_camera()
        self._setup_hand()
        self.refresh_telemetry(force=True)

    def _connect_arm(self):
        print(f"[Arm] connecting to {self.args.arm_ip}:{self.args.arm_port} ...")
        self.robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.handle = self.robot.rm_create_robot_arm(self.args.arm_ip, self.args.arm_port)
        print(f"[Arm] handle id = {self.handle.id}")
        if self.handle.id == -1:
            raise RuntimeError("failed to connect to the RealMan arm controller")

    def _connect_camera(self):
        print(f"[Camera] starting RealSense color {self.args.width}x{self.args.height}@{self.args.fps} ...")
        devices = list_realsense_devices()
        if not devices:
            video_nodes = sorted(
                name for name in os.listdir("/dev") if name.startswith("video")
            ) if os.path.isdir("/dev") else []
            raise RuntimeError(
                "No RealSense device detected by librealsense. "
                "Check whether the camera is physically connected, whether the container "
                "was started after plugging the camera in, and whether the host can see "
                f"the device via USB/video nodes. Visible /dev/video*: {video_nodes}"
            )
        print(f"[Camera] detected devices: {devices}")
        config = rs.config()
        config.enable_stream(
            rs.stream.color,
            self.args.width,
            self.args.height,
            rs.format.bgr8,
            self.args.fps,
        )
        self.pipeline = rs.pipeline()
        try:
            self.pipeline.start(config)
            self.camera_started = True
        except RuntimeError as exc:
            raise RuntimeError(
                "RealSense was enumerated but stream start failed. "
                "This is often caused by USB bandwidth, cable, power, or missing "
                "device passthrough inside the container."
            ) from exc

    def _setup_hand(self):
        assert self.robot is not None
        ret = self.robot.rm_set_modbus_mode(HAND_PORT, HAND_BAUDRATE, HAND_TIMEOUT)
        print(
            f"[Hand] rm_set_modbus_mode(port={HAND_PORT}, baudrate={HAND_BAUDRATE}, "
            f"timeout={HAND_TIMEOUT}) -> {ret}"
        )
        self.hand_ready = ret == 0
        if not self.hand_ready:
            raise RuntimeError("failed to open BM65 tool-port Modbus for Revo2")

        time.sleep(0.1)
        unit_ret = self.write_single_register(HAND_UNIT_MODE_REG, 0)
        print(f"[Hand] write reg 937(unit mode=normalized) -> {unit_ret}")
        if unit_ret != 0:
            raise RuntimeError("failed to set Revo2 unit mode to normalized")

        self.refresh_hand_config()

    def close(self):
        if self.pipeline is not None and self.camera_started:
            try:
                self.pipeline.stop()
                print("[Camera] stopped")
            except Exception as exc:
                print(f"[Camera] stop error: {exc}")
        elif self.pipeline is not None:
            print("[Camera] pipeline was created but never started; skip stop()")

        if self.pipeline is not None:
            self.pipeline = None
        self.camera_started = False

        if self.robot is not None and self.hand_ready:
            try:
                ret = self.robot.rm_close_modbus_mode(HAND_PORT)
                print(f"[Hand] rm_close_modbus_mode({HAND_PORT}) -> {ret}")
            except Exception as exc:
                print(f"[Hand] close error: {exc}")
            self.hand_ready = False

        if self.robot is not None:
            try:
                ret = self.robot.rm_delete_robot_arm()
                print(f"[Arm] rm_delete_robot_arm() -> {ret}")
            except Exception as exc:
                print(f"[Arm] disconnect error: {exc}")
            self.robot = None

        cv2.destroyAllWindows()

    def request_stop(self, *_):
        self.stop_requested = True

    def _make_hand_rw_params(self, address: int, num: int | None = None):
        return rm_peripheral_read_write_params_t(
            port=HAND_PORT,
            address=address,
            device=self.args.hand_id,
            num=num,
        )

    def _decode_u16_registers(self, raw_data: list[int], num: int) -> list[int]:
        if len(raw_data) == num:
            return [int(value) for value in raw_data]
        if len(raw_data) < num * 2:
            raise ValueError(f"expected {num} registers or {num * 2} bytes, got {len(raw_data)}")
        raw_bytes = bytes((value & 0xFF) for value in raw_data[: num * 2])
        return list(struct.unpack(f">{num}H", raw_bytes))

    def read_single_holding_register(self, address: int) -> tuple[int, int]:
        assert self.robot is not None
        return self.robot.rm_read_holding_registers(self._make_hand_rw_params(address))

    def read_multiple_holding_registers(self, address: int, num: int) -> tuple[int, list[int]]:
        assert self.robot is not None
        ret, raw = self.robot.rm_read_multiple_holding_registers(self._make_hand_rw_params(address, num))
        if ret != 0:
            return ret, []
        return ret, self._decode_u16_registers(raw, num)

    def read_multiple_input_registers(self, address: int, num: int) -> tuple[int, list[int]]:
        assert self.robot is not None
        ret, raw = self.robot.rm_read_multiple_input_registers(self._make_hand_rw_params(address, num))
        if ret != 0:
            return ret, []
        return ret, self._decode_u16_registers(raw, num)

    def write_single_register(self, address: int, value: int) -> int:
        assert self.robot is not None
        return self.robot.rm_write_single_register(self._make_hand_rw_params(address), value)

    def write_multi_registers(self, address: int, values: list[int]) -> int:
        assert self.robot is not None
        payload = [byte for value in values for byte in struct.pack(">h", value)]
        return self.robot.rm_write_registers(self._make_hand_rw_params(address, len(values)), payload)

    def write_hand_command(self, command_words: list[int]) -> int:
        if len(command_words) != 12:
            raise ValueError(f"expected 12 command words, got {len(command_words)}")

        first_batch = command_words[:10]
        last_finger = [5] + command_words[10:]

        print(f"[Hand] write 1010..1019 -> {first_batch}")
        ret = self.write_multi_registers(1010, first_batch)
        print(f"[Hand] rm_write_registers(1010, len=10) -> {ret}")
        if ret != 0:
            return ret

        time.sleep(0.05)

        print(f"[Hand] write 1052..1054 -> {last_finger}")
        ret = self.write_multi_registers(1052, last_finger)
        print(f"[Hand] rm_write_registers(1052, len=3) -> {ret}")
        return ret

    def refresh_hand_config(self):
        unit_ret, unit_mode = self.read_single_holding_register(HAND_UNIT_MODE_REG)
        if unit_ret == 0:
            self.hand_unit_mode = unit_mode

        min_ret, hand_min = self.read_multiple_holding_registers(HAND_MIN_POS_REG_START, HAND_NUM_JOINTS)
        if min_ret == 0:
            self.hand_min_deg = [float(value) for value in hand_min]

        max_ret, hand_max = self.read_multiple_holding_registers(HAND_MAX_POS_REG_START, HAND_NUM_JOINTS)
        if max_ret == 0:
            self.hand_max_deg = [float(value) for value in hand_max]

    def read_hand_joint_radians(self) -> tuple[int, list[float | str]]:
        ret, raw_positions = self.read_multiple_input_registers(HAND_ACTUAL_POS_REG_START, HAND_NUM_JOINTS)
        if ret != 0:
            return ret, [""] * HAND_NUM_JOINTS

        hand_joint_rad: list[float] = []
        for idx, raw_value in enumerate(raw_positions):
            if self.hand_unit_mode == 1:
                degree = raw_value / 10.0
            else:
                min_deg = self.hand_min_deg[idx]
                max_deg = self.hand_max_deg[idx]
                degree = min_deg + (float(raw_value) / 1000.0) * (max_deg - min_deg)
            hand_joint_rad.append(math.radians(degree))

        return ret, hand_joint_rad

    def refresh_telemetry(self, force: bool = False):
        if not force and time.monotonic() - self.last_telemetry_refresh < self.telemetry_interval:
            return

        assert self.robot is not None
        telemetry = Telemetry()

        arm_ret, arm_state = self.robot.rm_get_current_arm_state()
        telemetry.arm_ret_code = arm_ret
        if arm_ret == 0:
            telemetry.arm_joint_deg = (arm_state.get("joint", []) + [""] * 6)[:6]
            telemetry.arm_pose = (arm_state.get("pose", []) + [""] * 6)[:6]

        self.refresh_hand_config()
        hand_ret, hand_joint_rad = self.read_hand_joint_radians()
        telemetry.hand_ret_code = hand_ret
        telemetry.hand_joint_rad = hand_joint_rad
        telemetry.hand_unit_mode = self.hand_unit_mode
        telemetry.last_update_wall_ts = time.time()

        self.telemetry = telemetry
        self.last_telemetry_refresh = time.monotonic()

    def movej(self, joints: list[float], *, name: str):
        assert self.robot is not None
        print(f"[Arm] movej {name}: {joints}")
        ret = self.robot.rm_movej(joints, v=self.args.move_speed, r=0, connect=0, block=1)
        print(f"[Arm] rm_movej({name}) -> {ret}")
        if ret != 0:
            raise RuntimeError(f"rm_movej({name}) failed with code {ret}")

    def movel_offset(self, offset: list[float], *, name: str):
        assert self.robot is not None
        arm_ret, state = self.robot.rm_get_current_arm_state()
        if arm_ret != 0:
            raise RuntimeError(f"rm_get_current_arm_state failed with code {arm_ret}")
        pose = list(state["pose"])
        for idx, delta in enumerate(offset):
            pose[idx] += delta

        print(f"[Arm] movel {name}: offset={offset}")
        ret = self.robot.rm_movel(pose, self.args.move_speed, 0, 0, 1)
        print(f"[Arm] rm_movel({name}) -> {ret}")
        if ret != 0:
            raise RuntimeError(f"rm_movel({name}) failed with code {ret}")

    def go_home(self):
        self.write_hand_command(HAND_CMD_HOME)
        time.sleep(0.3)
        self.movej(HOME_JOINTS, name="home")
        self.refresh_telemetry(force=True)

    def go_pregrasp(self):
        self.movej(PREGRASP_JOINTS, name="pregrasp")
        self.refresh_telemetry(force=True)

    def hand_open(self):
        ret = self.write_hand_command(HAND_CMD_RELEASE)
        if ret != 0:
            raise RuntimeError(f"hand open failed with code {ret}")
        time.sleep(1.0)
        self.refresh_telemetry(force=True)

    def hand_close(self):
        ret = self.write_hand_command(HAND_CMD_GRASP)
        if ret != 0:
            raise RuntimeError(f"hand close failed with code {ret}")
        time.sleep(1.2)
        self.refresh_telemetry(force=True)

    def run_grasp_sequence(self):
        self.go_pregrasp()
        self.movel_offset(list(self.args.grasp_offset), name="grasp-approach")
        self.hand_close()
        self.movel_offset([-value for value in self.args.grasp_offset], name="grasp-retreat")
        self.refresh_telemetry(force=True)

    def run_release_sequence(self):
        self.go_pregrasp()
        self.movel_offset(list(self.args.grasp_offset), name="release-approach")
        self.hand_open()
        self.movel_offset([-value for value in self.args.grasp_offset], name="release-retreat")
        self.refresh_telemetry(force=True)

    def save_snapshot(self, frame: np.ndarray):
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = SNAPSHOT_DIR / f"{timestamp}.jpg"
        ok = cv2.imwrite(str(output_path), frame)
        print(f"[Camera] save snapshot -> {output_path} ({'ok' if ok else 'failed'})")

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        overlay = frame.copy()
        line_y = 28
        line_step = 24

        def put(text: str, color=(0, 255, 0)):
            nonlocal line_y
            cv2.putText(overlay, text, (16, line_y), OVERLAY_FONT, 0.62, color, 2)
            line_y += line_step

        put("Keys: h home | p pregrasp | o open | c close | g grasp | r release | s snapshot | q quit")
        put(
            f"Arm ret={self.telemetry.arm_ret_code} | Hand ret={self.telemetry.hand_ret_code} | "
            f"Hand mode={self.telemetry.hand_unit_mode}",
            color=(0, 255, 255),
        )

        arm_joints = ", ".join(
            f"{value:.1f}" if isinstance(value, (int, float)) else "--"
            for value in self.telemetry.arm_joint_deg
        )
        hand_joints = ", ".join(
            f"{value:.3f}" if isinstance(value, (int, float)) else "--"
            for value in self.telemetry.hand_joint_rad
        )
        pose_text = ", ".join(
            f"{value:.3f}" if isinstance(value, (int, float)) else "--"
            for value in self.telemetry.arm_pose[:3]
        )
        put(f"Arm joints deg: {arm_joints}")
        put(f"Arm pose xyz m: {pose_text}")
        put(f"Hand joints rad: {hand_joints}")
        if self.telemetry.last_update_wall_ts:
            put(
                f"Telemetry ts: {time.strftime('%H:%M:%S', time.localtime(self.telemetry.last_update_wall_ts))}",
                color=(255, 255, 0),
            )

        return overlay

    def run_action(self, label: str, action):
        print(f"\n[Action] {label}")
        try:
            action()
            print(f"[Action] {label} done")
        except Exception as exc:
            print(f"[Action] {label} failed: {exc}")

    def event_loop(self):
        print("[Control] h home | p pregrasp | o open | c close | g grasp | r release | s snapshot | q quit")
        while not self.stop_requested:
            color_image = None
            if self.pipeline is not None:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                color_image = np.asanyarray(color_frame.get_data())

            self.refresh_telemetry()
            if color_image is not None:
                vis = self.draw_overlay(color_image)
                cv2.imshow(WINDOW_NAME, vis)
                key = cv2.waitKeyEx(1)
            else:
                key = cv2.waitKeyEx(50)
            if key == -1:
                continue

            key &= 0xFFFFFFFF
            if key in (ord("q"), 27):
                self.stop_requested = True
            elif key == ord("h"):
                self.run_action("move home", self.go_home)
            elif key == ord("p"):
                self.run_action("move pregrasp", self.go_pregrasp)
            elif key == ord("o"):
                self.run_action("hand open", self.hand_open)
            elif key == ord("c"):
                self.run_action("hand close", self.hand_close)
            elif key == ord("g"):
                self.run_action("grasp sequence", self.run_grasp_sequence)
            elif key == ord("r"):
                self.run_action("release sequence", self.run_release_sequence)
            elif key == ord("s") and color_image is not None:
                self.save_snapshot(color_image)


def main():
    args = parse_args()
    demo = UnifiedGraspDemo(args)

    signal.signal(signal.SIGINT, demo.request_stop)
    signal.signal(signal.SIGTERM, demo.request_stop)

    try:
        demo.connect()
        demo.event_loop()
    finally:
        demo.close()


if __name__ == "__main__":
    main()
