#!/usr/bin/env python3
from __future__ import annotations

import math
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml


DEFAULT_ARM_IP = "192.168.1.18"
DEFAULT_ARM_PORT = 8080
DEFAULT_HAND_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent
    / "workspace/rm65_dex_ws/src/dex_hand_control/config/revo2_left_qpos.yaml"
)
HAND_DEVICE_ID = 126
HAND_PORT = 1
HAND_BAUDRATE = 460800
HAND_TIMEOUT = 20
HAND_UNIT_MODE_REG = 937
HAND_MIN_POS_REG_START = 946
HAND_MAX_POS_REG_START = 952
HAND_ACTUAL_POS_REG_START = 2000
HAND_POS_SPEED_REG_START = 1022
HAND_SINGLE_FINGER_REG_START = 1055
HAND_NUM_JOINTS = 6
HAND_CONFIG_REFRESH_SEC = 1.0
HAND_DEFAULT_MIN_DEG = [0.0] * HAND_NUM_JOINTS
HAND_DEFAULT_MAX_DEG = [59.0, 90.0, 81.0, 81.0, 81.0, 81.0]
HAND_DEFAULT_SPEED = 600
# Modbus actual-position registers expose thumb bend before thumb rotate.
# Reorder to the teleop qpos convention used across this repo:
# [thumb rotate, thumb bend, index, middle, ring, pinky].
HAND_READBACK_TO_TELEOP_ORDER = [1, 0, 2, 3, 4, 5]
HAND_TELEOP_TO_MODBUS_ORDER = [1, 0, 2, 3, 4, 5]


def can_import_sdk() -> bool:
    try:
        from Robotic_Arm.rm_robot_interface import RoboticArm  # noqa: F401
    except Exception:
        return False
    return True


def rad_to_deg(values: list[float] | np.ndarray) -> list[float]:
    return [math.degrees(float(value)) for value in values]


def deg_to_rad(values: list[float] | np.ndarray) -> list[float]:
    return [math.radians(float(value)) for value in values]


@dataclass
class HandTeleopConfig:
    lower: np.ndarray
    upper: np.ndarray
    max_delta: float
    follow_pos_min: int = 0
    follow_pos_max: int = 1000


def load_hand_teleop_config(config_path: Path) -> HandTeleopConfig:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    params = data["hand_qpos_node"]["ros__parameters"]
    return HandTeleopConfig(
        lower=np.array(params.get("lower", [0.0] * 6), dtype=np.float32),
        upper=np.array(params.get("upper", [1.0] * 6), dtype=np.float32),
        max_delta=float(params.get("max_delta", 0.08)),
    )


class RealmanSdkClient:
    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port
        self.robot = None
        self.handle = None
        self._modbus_params_type = None
        self._modbus_rtu_write_params_type = None
        self.hand_modbus_enabled = False
        self.hand_modbus_api_mode: Optional[str] = None
        self.hand_unit_mode = 0
        self.hand_min_deg = HAND_DEFAULT_MIN_DEG.copy()
        self.hand_max_deg = HAND_DEFAULT_MAX_DEG.copy()
        self.last_hand_config_refresh = 0.0
        self.last_hand_joint_rad = [0.0] * HAND_NUM_JOINTS
        self.last_hand_modbus_attempt_time = 0.0
        self.last_hand_read_error_time = 0.0

    def connect(self) -> None:
        from Robotic_Arm.rm_robot_interface import (
            RoboticArm,
            rm_modbus_rtu_write_params_t,
            rm_peripheral_read_write_params_t,
            rm_thread_mode_e,
        )

        self.robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.handle = self.robot.rm_create_robot_arm(self.ip, self.port)
        self._modbus_params_type = rm_peripheral_read_write_params_t
        self._modbus_rtu_write_params_type = rm_modbus_rtu_write_params_t
        handle_id = getattr(self.handle, "id", -1)
        print(f"[SDK] connected to {self.ip}:{self.port}, handle id = {handle_id}")
        if handle_id == -1:
            raise RuntimeError("failed to connect to the RealMan arm controller")

    def close(self) -> None:
        if self.robot is None:
            return
        try:
            if self.hand_modbus_enabled:
                try:
                    ret = self.robot.rm_close_modbus_mode(HAND_PORT)
                    print(f"[SDK] rm_close_modbus_mode({HAND_PORT}) -> {ret}")
                except Exception as exc:
                    print(f"[SDK] rm_close_modbus_mode({HAND_PORT}) failed: {exc}")
            ret = self.robot.rm_delete_robot_arm()
            print(f"[SDK] rm_delete_robot_arm() -> {ret}")
        finally:
            self.robot = None
            self.handle = None
            self._modbus_params_type = None
            self._modbus_rtu_write_params_type = None
            self.hand_modbus_enabled = False
            self.hand_modbus_api_mode = None

    def get_joint_deg(self) -> list[float]:
        if self.robot is None:
            raise RuntimeError("SDK client is not connected")

        ret, state = self.robot.rm_get_current_arm_state()
        if ret == 0 and isinstance(state, dict):
            joint = state.get("joint", [])
            if isinstance(joint, list) and len(joint) >= 6:
                return [float(value) for value in joint[:6]]

        ret, joint_deg = self.robot.rm_get_joint_degree()
        if ret != 0 or len(joint_deg) < 6:
            raise RuntimeError(f"failed to read current joint state via SDK, ret={ret}")
        return [float(value) for value in joint_deg[:6]]

    def movej_deg(self, target_deg: list[float], speed: int, block: int = 0) -> int:
        if self.robot is None:
            raise RuntimeError("SDK client is not connected")
        return int(self.robot.rm_movej(target_deg, speed, 0, 0, block))

    def movej_follow_deg(self, target_deg: list[float]) -> int:
        if self.robot is None:
            raise RuntimeError("SDK client is not connected")
        return int(self.robot.rm_movej_follow(target_deg))

    def movej_canfd_deg(
        self,
        target_deg: list[float],
        follow: bool = True,
        expand: float = 0.0,
        trajectory_mode: int = 0,
        radio: int = 0,
    ) -> int:
        if self.robot is None:
            raise RuntimeError("SDK client is not connected")
        return int(
            self.robot.rm_movej_canfd(
                target_deg,
                follow=follow,
                expand=expand,
                trajectory_mode=trajectory_mode,
                radio=radio,
            )
        )

    def check_self_collision(self, target_deg: list[float]) -> Optional[int]:
        if self.robot is None:
            raise RuntimeError("SDK client is not connected")
        try:
            padded = list(target_deg)
            if len(padded) < 7:
                padded.extend([0.0] * (7 - len(padded)))
            return int(self.robot.rm_algo_safety_robot_self_collision_detection(padded))
        except Exception as exc:
            print(f"[SDK] self-collision check skipped: {exc}")
            return None

    def set_hand_speed(self, speed: int) -> int:
        if self.robot is None:
            raise RuntimeError("SDK client is not connected")
        return int(self.robot.rm_set_hand_speed(int(speed)))

    def set_hand_force(self, force: int) -> int:
        if self.robot is None:
            raise RuntimeError("SDK client is not connected")
        return int(self.robot.rm_set_hand_force(int(force)))

    def set_hand_follow_pos(self, hand_pos: list[int], block: bool = False) -> int:
        if self.robot is None:
            raise RuntimeError("SDK client is not connected")
        return int(self.robot.rm_set_hand_follow_pos([int(value) for value in hand_pos], block))

    def set_hand_follow_angle(self, hand_angle: list[int], block: bool = False) -> int:
        if self.robot is None:
            raise RuntimeError("SDK client is not connected")
        return int(self.robot.rm_set_hand_follow_angle([int(value) for value in hand_angle], block))

    def enable_hand_modbus(self, strict: bool = False) -> bool:
        if self.robot is None:
            raise RuntimeError("SDK client is not connected")
        if self.hand_modbus_enabled:
            self._refresh_hand_config(force=False)
            return True

        now = time.monotonic()
        if not strict and now - self.last_hand_modbus_attempt_time < 1.0:
            return False
        self.last_hand_modbus_attempt_time = now

        ret = int(self.robot.rm_set_modbus_mode(HAND_PORT, HAND_BAUDRATE, HAND_TIMEOUT))
        if ret == 0:
            self.hand_modbus_enabled = True
            print(
                f"[SDK] rm_set_modbus_mode(port={HAND_PORT}, baudrate={HAND_BAUDRATE}, "
                f"timeout={HAND_TIMEOUT}) -> {ret}"
            )
            self._refresh_hand_config(force=True)
            return True

        message = (
            "[SDK] failed to enable hand Modbus readback: "
            f"ret={ret}, port={HAND_PORT}, device={HAND_DEVICE_ID}"
        )
        if strict:
            raise RuntimeError(message)
        print(message)
        return False

    def _make_hand_rw_params(self, address: int, num: Optional[int] = None):
        if self._modbus_params_type is None:
            raise RuntimeError("SDK Modbus parameter type is unavailable before connect().")
        return self._modbus_params_type(
            port=HAND_PORT,
            address=address,
            device=HAND_DEVICE_ID,
            num=num,
        )

    def _make_hand_rtu_write_params(self, address: int, data: list[int]):
        if self._modbus_rtu_write_params_type is None:
            raise RuntimeError("SDK Modbus RTU write parameter type is unavailable before connect().")
        return self._modbus_rtu_write_params_type(
            address=address,
            device=HAND_DEVICE_ID,
            type=1,
            num=len(data),
            data=[int(value) for value in data],
        )

    @staticmethod
    def _decode_u16_registers(raw_data, num: int) -> list[int]:
        if len(raw_data) == num:
            return [int(value) for value in raw_data]
        if len(raw_data) < num * 2:
            raise ValueError(f"expected {num} registers or {num * 2} bytes, got {len(raw_data)}")
        raw_bytes = bytes((value & 0xFF) for value in raw_data[: num * 2])
        return list(struct.unpack(f">{num}H", raw_bytes))

    def _read_hand_single_holding(self, address: int) -> tuple[int, int]:
        if self.robot is None:
            raise RuntimeError("SDK client is not connected")
        return self.robot.rm_read_holding_registers(self._make_hand_rw_params(address))

    def _read_hand_multiple_holding(self, address: int, num: int) -> tuple[int, list[int]]:
        if self.robot is None:
            raise RuntimeError("SDK client is not connected")
        ret, raw_data = self.robot.rm_read_multiple_holding_registers(
            self._make_hand_rw_params(address, num)
        )
        if ret != 0:
            values: list[int] = []
            for offset in range(num):
                single_ret, single_value = self._read_hand_single_holding(address + offset)
                if single_ret != 0:
                    return ret, []
                values.append(int(single_value))
            return 0, values
        return ret, self._decode_u16_registers(raw_data, num)

    def _read_hand_multiple_input(self, address: int, num: int) -> tuple[int, list[int]]:
        if self.robot is None:
            raise RuntimeError("SDK client is not connected")
        ret, raw_data = self.robot.rm_read_multiple_input_registers(
            self._make_hand_rw_params(address, num)
        )
        if ret != 0:
            values: list[int] = []
            for offset in range(num):
                single_ret, single_value = self.robot.rm_read_input_registers(
                    self._make_hand_rw_params(address + offset)
                )
                if single_ret != 0:
                    return ret, []
                values.append(int(single_value))
            return 0, values
        return ret, self._decode_u16_registers(raw_data, num)

    def _legacy_write_hand_holding_registers(self, address: int, values: list[int]) -> int:
        if self.robot is None:
            raise RuntimeError("SDK client is not connected")
        if len(values) == 1:
            return int(self.robot.rm_write_single_register(self._make_hand_rw_params(address), int(values[0])))
        if len(values) > 10:
            raise ValueError(
                f"legacy peripheral API supports at most 10 holding registers per write, got {len(values)}"
            )
        packed = [byte for value in values for byte in struct.pack(">h", int(value))]
        return int(self.robot.rm_write_registers(self._make_hand_rw_params(address, len(values)), packed))

    def _write_hand_holding_registers(self, address: int, values: list[int]) -> int:
        if self.robot is None:
            raise RuntimeError("SDK client is not connected")
        if self.hand_modbus_api_mode == "legacy":
            return self._legacy_write_hand_holding_registers(address, values)
        try:
            ret = int(
                self.robot.rm_write_modbus_rtu_registers(
                    self._make_hand_rtu_write_params(address, values)
                )
            )
        except Exception as exc:
            if self.hand_modbus_api_mode != "legacy":
                print(f"[SDK] RTU4 hand write unavailable, falling back to legacy peripheral API: {exc}")
            self.hand_modbus_api_mode = "legacy"
            return self._legacy_write_hand_holding_registers(address, values)
        if ret == -4:
            if self.hand_modbus_api_mode != "legacy":
                print("[SDK] RTU4 hand write unsupported on this controller, falling back to legacy peripheral API.")
            self.hand_modbus_api_mode = "legacy"
            return self._legacy_write_hand_holding_registers(address, values)
        if ret == 0:
            self.hand_modbus_api_mode = "rtu4"
        return ret

    def _set_hand_unit_mode(self, mode: int) -> int:
        ret = self._write_hand_holding_registers(HAND_UNIT_MODE_REG, [int(mode)])
        if ret == 0:
            self.hand_unit_mode = int(mode)
            self.last_hand_config_refresh = 0.0
        return ret

    def _set_hand_single_finger_position_and_speed(
        self, finger_id: int, position: int, speed: int
    ) -> int:
        return self._write_hand_holding_registers(
            HAND_SINGLE_FINGER_REG_START,
            [int(finger_id), int(position), int(speed)],
        )

    def _set_hand_positions_with_single_finger_commands(
        self,
        positions: list[int],
        speed: int,
    ) -> int:
        for finger_id, position in enumerate(positions):
            ret = self._set_hand_single_finger_position_and_speed(finger_id, position, speed)
            if ret != 0:
                return ret
        return 0

    def set_hand_modbus_positions(self, hand_pos_native: list[int], speed: int = HAND_DEFAULT_SPEED) -> int:
        if len(hand_pos_native) != HAND_NUM_JOINTS:
            raise ValueError(f"expected {HAND_NUM_JOINTS} hand positions, got {len(hand_pos_native)}")
        if not self.enable_hand_modbus(strict=False):
            return -1
        if self.hand_unit_mode != 0:
            unit_ret = self._set_hand_unit_mode(0)
            if unit_ret != 0:
                return unit_ret
        speed = int(HAND_DEFAULT_SPEED if speed < 0 else speed)
        clipped = [max(0, min(1000, int(value))) for value in hand_pos_native]
        if self.hand_modbus_api_mode == "legacy":
            return self._set_hand_positions_with_single_finger_commands(clipped, speed)
        speeds = [speed] * HAND_NUM_JOINTS
        try:
            ret = self._write_hand_holding_registers(HAND_POS_SPEED_REG_START, clipped + speeds)
        except ValueError:
            if self.hand_modbus_api_mode != "legacy":
                print(
                    "[SDK] Legacy hand write path cannot send 12 holding registers in one shot; "
                    "switching to per-finger Modbus commands."
                )
            self.hand_modbus_api_mode = "legacy"
            return self._set_hand_positions_with_single_finger_commands(clipped, speed)
        if ret == 0:
            return ret
        if self.hand_modbus_api_mode != "legacy":
            print(
                f"[SDK] Bulk hand position+speed write failed with ret={ret}; "
                "retrying via per-finger Modbus commands."
            )
        self.hand_modbus_api_mode = "legacy"
        return self._set_hand_positions_with_single_finger_commands(clipped, speed)

    def _refresh_hand_config(self, force: bool = False) -> None:
        if not self.hand_modbus_enabled:
            return

        now = time.monotonic()
        if not force and now - self.last_hand_config_refresh < HAND_CONFIG_REFRESH_SEC:
            return

        unit_ret, unit_mode = self._read_hand_single_holding(HAND_UNIT_MODE_REG)
        if unit_ret == 0:
            self.hand_unit_mode = int(unit_mode)
        else:
            print(f"[SDK] read hand unit mode failed: ret={unit_ret}")

        min_ret, min_deg = self._read_hand_multiple_holding(HAND_MIN_POS_REG_START, HAND_NUM_JOINTS)
        if min_ret == 0:
            self.hand_min_deg = [float(value) for value in min_deg]
        else:
            print(f"[SDK] read hand min limits failed: ret={min_ret}")

        max_ret, max_deg = self._read_hand_multiple_holding(HAND_MAX_POS_REG_START, HAND_NUM_JOINTS)
        if max_ret == 0:
            self.hand_max_deg = [float(value) for value in max_deg]
        else:
            print(f"[SDK] read hand max limits failed: ret={max_ret}")

        self.last_hand_config_refresh = now

    def _normalized_to_degree(self, raw_value: int, joint_idx: int) -> float:
        min_deg = self.hand_min_deg[joint_idx]
        max_deg = self.hand_max_deg[joint_idx]
        return min_deg + (float(raw_value) / 1000.0) * (max_deg - min_deg)

    def read_hand_joint_rad(self, strict: bool = False) -> list[float]:
        if not self.enable_hand_modbus(strict=strict):
            return self.last_hand_joint_rad.copy()

        self._refresh_hand_config(force=False)
        ret, raw_positions = self._read_hand_multiple_input(HAND_ACTUAL_POS_REG_START, HAND_NUM_JOINTS)
        if ret != 0:
            if strict:
                raise RuntimeError(f"failed to read RealMan hand actual positions, ret={ret}")
            now = time.monotonic()
            if now - self.last_hand_read_error_time > 1.0:
                self.last_hand_read_error_time = now
                print(f"[SDK] failed to read RealMan hand actual positions, ret={ret}")
            return self.last_hand_joint_rad.copy()

        hand_joints_rad_native: list[float] = []
        for idx, raw_value in enumerate(raw_positions):
            if self.hand_unit_mode == 1:
                degree = float(raw_value) / 10.0
            else:
                degree = self._normalized_to_degree(int(raw_value), idx)
            hand_joints_rad_native.append(math.radians(degree))

        teleop_order = [hand_joints_rad_native[index] for index in HAND_READBACK_TO_TELEOP_ORDER]
        self.last_hand_joint_rad = teleop_order.copy()
        return teleop_order


def wait_for_joint_state_sdk(client: RealmanSdkClient, timeout_sec: float) -> list[float]:
    deadline = time.monotonic() + timeout_sec
    last_error: Optional[Exception] = None
    while time.monotonic() < deadline:
        try:
            return client.get_joint_deg()
        except Exception as exc:
            last_error = exc
            time.sleep(0.1)
    if last_error is not None:
        raise TimeoutError(f"Timed out waiting for RealMan SDK state: {last_error}") from last_error
    raise TimeoutError("Timed out waiting for RealMan SDK state.")


class RealmanHandFollower:
    """
    Map the existing UDP hand_qpos convention
    [thumb rotate, thumb bend, index, middle, ring, pinky]
    to RealMan dex-hand follow_pos order
    [pinky, ring, middle, index, thumb bend, thumb rotate].
    """

    REALMAN_FOLLOW_ORDER = [5, 4, 3, 2, 1, 0]
    REALMAN_MODBUS_ORDER = HAND_TELEOP_TO_MODBUS_ORDER
    BRAINCO_ECAT_ORDER = [0, 1, 2, 3, 4, 5]

    def __init__(self, config: HandTeleopConfig):
        self.config = config
        self.last_qpos: Optional[np.ndarray] = None
        self.last_processed_qpos: Optional[np.ndarray] = None

    def prepare_qpos(self, qpos: np.ndarray) -> np.ndarray:
        q = np.array(qpos[:6], dtype=np.float32)
        q = np.clip(q, self.config.lower, self.config.upper)
        if self.last_qpos is not None and self.last_qpos.shape == q.shape:
            delta = np.clip(q - self.last_qpos, -self.config.max_delta, self.config.max_delta)
            q = self.last_qpos + delta
        self.last_qpos = q.copy()
        self.last_processed_qpos = q.copy()
        return q

    def _qpos_to_positions(self, qpos: np.ndarray, order: list[int]) -> list[int]:
        q = np.array(qpos[:6], dtype=np.float32)

        span = np.maximum(self.config.upper - self.config.lower, 1e-6)
        normalized = np.clip((q - self.config.lower) / span, 0.0, 1.0)
        follow_pos = np.rint(
            self.config.follow_pos_min
            + normalized * (self.config.follow_pos_max - self.config.follow_pos_min)
        ).astype(np.int32)
        reordered = follow_pos[order]
        return [int(value) for value in reordered.tolist()]

    def process_qpos(self, qpos: np.ndarray) -> list[int]:
        return self._qpos_to_positions(self.prepare_qpos(qpos), self.REALMAN_FOLLOW_ORDER)

    def process_qpos_modbus(self, qpos: np.ndarray) -> list[int]:
        return self._qpos_to_positions(self.prepare_qpos(qpos), self.REALMAN_MODBUS_ORDER)

    def process_qpos_brainco_ethercat(self, qpos: np.ndarray) -> list[int]:
        return self._qpos_to_positions(self.prepare_qpos(qpos), self.BRAINCO_ECAT_ORDER)

    def positions_to_qpos(self, positions: list[int] | np.ndarray, order: list[int]) -> np.ndarray:
        raw = np.array(list(positions[:6]), dtype=np.float32)
        ordered = np.zeros(6, dtype=np.float32)
        for src_idx, teleop_idx in enumerate(order):
            ordered[teleop_idx] = raw[src_idx]
        normalized = np.clip(
            (ordered - self.config.follow_pos_min)
            / max(self.config.follow_pos_max - self.config.follow_pos_min, 1),
            0.0,
            1.0,
        )
        span = np.maximum(self.config.upper - self.config.lower, 1e-6)
        return (self.config.lower + normalized * span).astype(np.float32)

    def brainco_ethercat_positions_to_qpos(self, positions: list[int] | np.ndarray) -> np.ndarray:
        return self.positions_to_qpos(positions, self.BRAINCO_ECAT_ORDER)
