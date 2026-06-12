#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from hybrid_teleop_common import (
    CUBE_SIDE_MOUNT_CONFIG_PATH,
    HybridArmController,
    HybridConfig,
    load_hybrid_config,
    override_config_joint_seed_from_snapshot,
    parse_hand_qpos_packet,
    parse_wrist_packet,
    poll_latest_packet,
)
from brainco_hand_ethercat import BraincoEthercatHandClient, can_import_brainco_sdk
from realman_sdk_common import (
    DEFAULT_ARM_IP,
    DEFAULT_ARM_PORT,
    DEFAULT_HAND_CONFIG_PATH,
    RealmanHandFollower,
    RealmanSdkClient,
    can_import_sdk,
    deg_to_rad,
    load_hand_teleop_config,
    rad_to_deg,
)
from realman_teleop_dataset_recorder import RealmanTeleopDatasetRecorder


DEFAULT_JOINT_SNAPSHOT_PATH = Path("/tmp/rm65_real_sdk_arm_snapshot.json")
DEFAULT_REALMAN_CONFIG_PATH = CUBE_SIDE_MOUNT_CONFIG_PATH


class RealmanSdkHybridTeleop:
    def __init__(
        self,
        config: HybridConfig,
        client: RealmanSdkClient,
        wrist_port: int,
        hand_port: int,
        hand_follower: Optional[RealmanHandFollower],
        brainco_hand_client: Optional[BraincoEthercatHandClient],
        joint_snapshot_path: Optional[Path],
        arm_command_mode: str,
        hand_backend: str,
        hand_speed: int,
        hand_control_dt: float,
        canfd_trajectory_mode: int,
        canfd_radio: int,
        control_dt: float,
        feedback_source: str,
        seed_current_as_nominal: bool,
        dataset_recorder: Optional[RealmanTeleopDatasetRecorder],
    ):
        self.config = config
        if control_dt > 0.0:
            self.config.trajectory_control_dt = float(control_dt)
        self.client = client
        self.wrist_port = wrist_port
        self.hand_port = hand_port
        self.hand_follower = hand_follower
        self.brainco_hand_client = brainco_hand_client
        self.joint_snapshot_path = joint_snapshot_path
        self.arm_command_mode = arm_command_mode
        self.hand_backend = "realman_follow" if hand_backend == "auto" else hand_backend
        self.hand_speed = 600 if hand_speed < 0 else hand_speed
        self.hand_control_dt = max(hand_control_dt, 0.02)
        self.canfd_trajectory_mode = canfd_trajectory_mode
        self.canfd_radio = canfd_radio
        self.control_dt = max(self.config.trajectory_control_dt, 0.005)
        self.feedback_source = feedback_source
        self.seed_current_as_nominal = seed_current_as_nominal
        self.dataset_recorder = dataset_recorder

        self.arm_controller = HybridArmController(config)
        self.latest_wrist: Optional[np.ndarray] = None
        self.latest_landmarks: Optional[np.ndarray] = None
        self.latest_wrist_time: Optional[float] = None
        self.latest_hand_qpos: Optional[np.ndarray] = None
        self.latest_hand_time: Optional[float] = None
        self.last_snapshot_write_time = 0.0
        self.last_log_time = 0.0
        self.last_stale_log_time = 0.0
        self.last_arm_command_ret: Optional[int] = None
        self.last_hand_command_ret: Optional[int] = None
        self.last_hand_command_time = 0.0
        self.last_hand_state_read_time = 0.0
        self.last_hand_backend_log = ""
        self.latest_arm_qpos_rad = config.initial_joint_positions.astype(np.float32).copy()
        self.active_hand_qpos = np.zeros(6, dtype=np.float32)
        self.latest_hand_state_qpos = np.zeros(6, dtype=np.float32)
        if self.dataset_recorder is not None:
            self.hand_state_poll_dt = max(0.02, min(self.dataset_recorder.record_period * 0.5, 0.1))
        else:
            self.hand_state_poll_dt = 0.05

        self.wrist_sock = self.make_udp_socket(self.wrist_port)
        self.hand_sock = (
            self.make_udp_socket(self.hand_port)
            if self.hand_follower is not None and self.hand_port > 0
            else None
        )

    @staticmethod
    def make_udp_socket(port: int) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", port))
        sock.setblocking(False)
        return sock

    def send_arm_command(self, target_q_rad: np.ndarray) -> int:
        target_deg = rad_to_deg(target_q_rad.tolist())
        if self.arm_command_mode == "movej_follow":
            return self.client.movej_follow_deg(target_deg)
        if self.arm_command_mode == "movej_canfd_high":
            return self.client.movej_canfd_deg(
                target_deg,
                follow=True,
                trajectory_mode=self.canfd_trajectory_mode,
                radio=self.canfd_radio,
            )
        if self.arm_command_mode == "movej_canfd_vrteleop":
            return self.client.movej_canfd_deg(target_deg, follow=True)
        if self.arm_command_mode == "movej_canfd_low":
            return self.client.movej_canfd_deg(
                target_deg,
                follow=False,
                trajectory_mode=0,
                radio=0,
            )
        raise ValueError(f"Unsupported arm command mode: {self.arm_command_mode}")

    def maybe_send_hand_command(self, now: float) -> None:
        if self.hand_follower is None:
            return
        if self.latest_hand_qpos is None or self.latest_hand_time is None:
            return
        if now - self.latest_hand_time > 1.0:
            return
        if now - self.last_hand_command_time < self.hand_control_dt:
            return

        if self.hand_backend == "brainco_ethercat":
            hand_pos = self.hand_follower.process_qpos_brainco_ethercat(self.latest_hand_qpos)
            duration_ms = max(1, int(round(self.hand_control_dt * 1000.0)))
            self.last_hand_command_ret = self.brainco_hand_client.set_positions(
                hand_pos,
                duration_ms=duration_ms,
            )
            backend_log = "brainco_ethercat"
        elif self.hand_backend == "realman_modbus":
            hand_pos = self.hand_follower.process_qpos_modbus(self.latest_hand_qpos)
            self.last_hand_command_ret = self.client.set_hand_modbus_positions(
                hand_pos,
                speed=self.hand_speed,
            )
            backend_log = "realman_modbus"
        else:
            hand_pos = self.hand_follower.process_qpos(self.latest_hand_qpos)
            self.last_hand_command_ret = self.client.set_hand_follow_pos(hand_pos, block=False)
            backend_log = "realman_follow"
            if self.last_hand_command_ret not in (None, 0) and self.hand_backend == "realman_follow":
                print(
                    f"[Hand] rm_set_hand_follow_pos failed with ret={self.last_hand_command_ret}; "
                    "switching to RealMan tool-port Modbus backend."
                )
                self.hand_backend = "realman_modbus"
                hand_pos = self.hand_follower.process_qpos_modbus(self.latest_hand_qpos)
                self.last_hand_command_ret = self.client.set_hand_modbus_positions(
                    hand_pos,
                    speed=self.hand_speed,
                )
                backend_log = "realman_modbus"
        self.last_hand_command_time = now
        if self.hand_follower.last_processed_qpos is not None:
            self.active_hand_qpos = self.hand_follower.last_processed_qpos.astype(np.float32)
        else:
            self.active_hand_qpos = self.latest_hand_qpos[:6].astype(np.float32)
        if backend_log != self.last_hand_backend_log:
            print(f"[Hand] active backend -> {backend_log}")
            self.last_hand_backend_log = backend_log

    def maybe_update_hand_state(self, now: float, force: bool = False) -> None:
        if self.hand_follower is None and self.dataset_recorder is None:
            return
        if self.dataset_recorder is None:
            self.latest_hand_state_qpos = self.active_hand_qpos.copy()
            return
        if not force and now - self.last_hand_state_read_time < self.hand_state_poll_dt:
            return
        if self.hand_backend == "brainco_ethercat" and self.brainco_hand_client is not None:
            positions = self.brainco_hand_client.get_positions()
            if len(positions) >= 6 and self.hand_follower is not None:
                self.latest_hand_state_qpos = self.hand_follower.brainco_ethercat_positions_to_qpos(
                    positions
                )
                self.last_hand_state_read_time = now
            return
        hand_state = self.client.read_hand_joint_rad(strict=False)
        if len(hand_state) >= 6:
            self.latest_hand_state_qpos = np.array(hand_state[:6], dtype=np.float32)
            self.last_hand_state_read_time = now

    def write_joint_snapshot(self) -> None:
        if self.joint_snapshot_path is None:
            return
        now_monotonic = time.monotonic()
        if now_monotonic - self.last_snapshot_write_time < 0.2:
            return

        snapshot_path = self.joint_snapshot_path
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": time.time(),
            "monotonic_time": now_monotonic,
            "arm_joint_names": [f"joint{i}" for i in range(1, 7)],
            "arm_qpos": [float(value) for value in self.latest_arm_qpos_rad],
            "hand_qpos": [float(value) for value in self.latest_hand_state_qpos],
            "axis_mapping": self.config.axis_mapping,
            "rotation_control_mode": self.config.rotation_control_mode,
            "source": "realman_sdk_hybrid_teleop",
            "arm_command_mode": self.arm_command_mode,
        }
        temp_path = snapshot_path.with_suffix(snapshot_path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(snapshot_path)
        self.last_snapshot_write_time = now_monotonic

    def log_status(self, now: float, current_qpos: np.ndarray, target_q: Optional[np.ndarray]) -> None:
        if now - self.last_log_time < 1.0:
            return
        self.last_log_time = now
        parts = [
            f"arm_mode={self.arm_command_mode}",
            f"hand_backend={self.hand_backend}",
            f"arm_ret={self.last_arm_command_ret}",
            f"hand_ret={self.last_hand_command_ret}",
            f"q={np.round(current_qpos, 3).tolist()}",
        ]
        if target_q is not None:
            parts.append(f"cmd={np.round(target_q, 3).tolist()}")
        print("RealMan SDK teleop:", *parts)

    def log_stale(self, now: float) -> None:
        if now - self.last_stale_log_time < 1.0:
            return
        self.last_stale_log_time = now
        print(
            "RealMan SDK teleop waiting for fresh wrist UDP:",
            f"port={self.wrist_port}",
            f"timeout={self.config.packet_timeout:.3f}s",
        )

    def sync_initial_arm_state(self) -> None:
        current_deg = self.client.get_joint_deg()
        current_qpos = np.array(deg_to_rad(current_deg), dtype=np.float32)
        self.latest_arm_qpos_rad = current_qpos.copy()
        if self.seed_current_as_nominal:
            self.config.initial_joint_positions = current_qpos.copy()
            self.config.nominal_joint_positions = current_qpos.copy()
            self.arm_controller.trajectory_smoother.reset(
                current_qpos,
                np.zeros(6, dtype=np.float32),
            )
            print(
                "RealMan SDK teleop seeded current real arm pose as v1 initial/nominal:",
                f"q_deg={np.round(current_deg, 3).tolist()}",
            )

    def step(self) -> None:
        wrist = poll_latest_packet(self.wrist_sock, parse_wrist_packet)
        if wrist is not None:
            self.latest_wrist, self.latest_landmarks = wrist
            self.latest_wrist_time = time.monotonic()

        if self.hand_sock is not None:
            hand_qpos = poll_latest_packet(self.hand_sock, parse_hand_qpos_packet)
            if hand_qpos is not None:
                self.latest_hand_qpos = hand_qpos[:6]
                self.latest_hand_time = time.monotonic()

        if self.feedback_source == "sdk":
            current_deg = self.client.get_joint_deg()
            current_qpos = np.array(deg_to_rad(current_deg), dtype=np.float32)
            self.latest_arm_qpos_rad = current_qpos.copy()
        else:
            current_qpos = self.latest_arm_qpos_rad.copy()

        now = time.monotonic()
        target_q = None
        if (
            self.latest_wrist is not None
            and self.latest_wrist_time is not None
            and now - self.latest_wrist_time <= self.config.packet_timeout
        ):
            target_q = self.arm_controller.update(
                self.latest_wrist,
                current_qpos,
                self.latest_landmarks,
            ).astype(np.float32)
            self.last_arm_command_ret = self.send_arm_command(target_q)
            if self.feedback_source == "command":
                self.latest_arm_qpos_rad = target_q.copy()
        else:
            self.log_stale(now)

        self.maybe_send_hand_command(now)
        self.maybe_update_hand_state(now)
        if self.dataset_recorder is not None and target_q is not None:
            self.dataset_recorder.maybe_record_step(
                now_monotonic=now,
                q_now=current_qpos,
                q_cmd=target_q,
                hand_state=self.latest_hand_state_qpos,
                hand_action=self.active_hand_qpos,
            )
        self.write_joint_snapshot()
        self.log_status(now, current_qpos, target_q)

    def run(self) -> None:
        self.sync_initial_arm_state()
        loop_dt = self.control_dt
        next_tick = time.monotonic()
        print(
            f"Listening for wrist_pose UDP on :{self.wrist_port}"
            + (
                f" and hand_qpos UDP on :{self.hand_port}."
                if self.hand_sock is not None
                else "."
            )
        )
        print(
            "RealMan SDK hybrid teleop started:",
            f"axis_mapping={self.config.axis_mapping}",
            f"rotation_mode={self.config.rotation_control_mode}",
            f"smoother={self.config.trajectory_smoother}",
            f"arm_mode={self.arm_command_mode}",
            f"control_dt={loop_dt:.3f}s",
            f"feedback_source={self.feedback_source}",
            f"hand_backend={self.hand_backend}",
        )
        while True:
            self.step()
            next_tick += loop_dt
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                next_tick = time.monotonic()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the same DLS + Ruckig wrist teleop pipeline as MuJoCo, "
            "but send final arm commands directly through the RealMan Python SDK."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_REALMAN_CONFIG_PATH,
        help="Path to a wrist bridge-compatible config yaml. Defaults to the cube-side-mount mainline config.",
    )
    parser.add_argument(
        "--hand-config",
        type=Path,
        default=DEFAULT_HAND_CONFIG_PATH,
        help="Path to hand_qpos_node yaml for clipping/rate-limit settings.",
    )
    parser.add_argument(
        "--wrist-port",
        type=int,
        default=5005,
        help="UDP port for forwarded wrist_pose.",
    )
    parser.add_argument(
        "--hand-port",
        type=int,
        default=5010,
        help="UDP port for forwarded hand_qpos. Set to 0 to leave hand on another backend.",
    )
    parser.add_argument(
        "--joint-snapshot-path",
        type=Path,
        default=DEFAULT_JOINT_SNAPSHOT_PATH,
        help="Optional JSON snapshot of the current real arm pose.",
    )
    parser.add_argument(
        "--startup-snapshot-path",
        type=Path,
        default=None,
        help=(
            "Optional JSON snapshot whose arm_qpos overrides the controller's initial and nominal "
            "joint seed before teleop starts."
        ),
    )
    parser.add_argument(
        "--arm-ip",
        type=str,
        default=DEFAULT_ARM_IP,
        help="RealMan controller IP.",
    )
    parser.add_argument(
        "--arm-port",
        type=int,
        default=DEFAULT_ARM_PORT,
        help="RealMan controller TCP port.",
    )
    parser.add_argument(
        "--arm-command-mode",
        choices=["movej_follow", "movej_canfd_high", "movej_canfd_low", "movej_canfd_vrteleop"],
        default="movej_follow",
        help="SDK follow backend used for continuous arm teleoperation.",
    )
    parser.add_argument(
        "--canfd-trajectory-mode",
        type=int,
        default=2,
        help="CANFD high-follow trajectory_mode when arm-command-mode=movej_canfd_high.",
    )
    parser.add_argument(
        "--canfd-radio",
        type=int,
        default=50,
        help="CANFD smoothing parameter when arm-command-mode=movej_canfd_high.",
    )
    parser.add_argument(
        "--control-dt",
        type=float,
        default=0.0,
        help=(
            "Override the main arm control period in seconds. Use 0.02 to match "
            "vr_teleop RM65 real-control frequency."
        ),
    )
    parser.add_argument(
        "--trajectory-smoother",
        choices=["config", "auto", "ruckig", "accel_limited", "none"],
        default="config",
        help=(
            "Override the v1 joint smoother. 'config' keeps the YAML setting; "
            "vr_teleop itself sends CANFD targets directly, but keeping Ruckig is usually safer "
            "with the v1 DLS controller."
        ),
    )
    parser.add_argument(
        "--feedback-source",
        choices=["sdk", "command"],
        default="sdk",
        help=(
            "'sdk' reads real joints every cycle. 'command' uses the last commanded "
            "joint state as the next IK seed, matching vr_teleop's RM65 loop more closely."
        ),
    )
    parser.add_argument(
        "--seed-current-as-nominal",
        action="store_true",
        help=(
            "At startup, read the real arm once and use that pose as v1 initial/nominal "
            "regularization posture. This avoids an implicit pull toward stale config seeds."
        ),
    )
    parser.add_argument(
        "--hand-control-dt",
        type=float,
        default=0.02,
        help="Minimum interval between SDK hand follow commands. RealMan hand follow is capped at 50Hz.",
    )
    parser.add_argument(
        "--disable-hand-follow",
        action="store_true",
        help="Ignore incoming hand_qpos UDP and drive only the arm.",
    )
    parser.add_argument(
        "--hand-backend",
        choices=["auto", "realman_follow", "realman_modbus", "brainco_ethercat"],
        default="auto",
        help=(
            "Hand command backend. 'auto' tries rm_set_hand_follow_pos first and falls back to "
            "RealMan tool-port Modbus if the controller rejects follow-pos. "
            "'brainco_ethercat' drives the hand through the BrainCo SDK over EtherCAT."
        ),
    )
    parser.add_argument(
        "--brainco-master-pos",
        type=int,
        default=0,
        help="EtherCAT master index for the BrainCo hand backend.",
    )
    parser.add_argument(
        "--brainco-slave-pos",
        type=int,
        default=0,
        help="EtherCAT slave position for the BrainCo hand backend.",
    )
    parser.add_argument(
        "--brainco-cycle-ns",
        type=int,
        default=1_000_000,
        help="PDO loop cycle time in nanoseconds for the BrainCo EtherCAT backend.",
    )
    parser.add_argument(
        "--hand-speed",
        type=int,
        default=-1,
        help=(
            "Hand speed preset. For rm_set_hand_follow_pos it is forwarded to the RealMan SDK "
            "when supported; for the Modbus backend it becomes the per-finger target speed."
        ),
    )
    parser.add_argument(
        "--hand-force",
        type=int,
        default=-1,
        help="Optional SDK hand force preset. Negative values leave it unchanged.",
    )
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=None,
        help=(
            "Optional dataset output directory. When set, recording front/wrist images, "
            "arm q_now, arm q_cmd, actual hand state, hand q_cmd, and task is enabled."
        ),
    )
    parser.add_argument(
        "--task",
        type=str,
        default="",
        help="Language task text stored in each recorded step.",
    )
    parser.add_argument(
        "--record-hz",
        type=float,
        default=10.0,
        help="Recording frequency in Hz when --record-dir is enabled.",
    )
    parser.add_argument(
        "--front-camera-serial",
        type=str,
        default="",
        help="Optional RealSense serial for the front camera.",
    )
    parser.add_argument(
        "--wrist-camera-serial",
        type=str,
        default="",
        help="Optional RealSense serial for the wrist camera.",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=640,
        help="Recorded image width for front and wrist cameras.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=480,
        help="Recorded image height for front and wrist cameras.",
    )
    parser.add_argument(
        "--camera-fps",
        type=int,
        default=30,
        help="Requested RealSense color stream FPS for recording.",
    )
    parser.add_argument(
        "--record-jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality used for saved front/wrist images.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not can_import_sdk():
        print(
            "RealMan SDK is not importable in this Python environment. "
            "Run this script with the SDK conda env or set REALMAN_SDK_PYTHON accordingly.",
            file=sys.stderr,
        )
        return 2

    config = load_hybrid_config(args.config)
    if args.trajectory_smoother != "config":
        config.trajectory_smoother = args.trajectory_smoother
    config = override_config_joint_seed_from_snapshot(
        config,
        args.startup_snapshot_path,
        log_prefix="[RealMan teleop]",
    )
    hand_follower = None
    brainco_hand_client = None
    if not args.disable_hand_follow or args.record_dir is not None:
        hand_follower = RealmanHandFollower(load_hand_teleop_config(args.hand_config))
    dataset_recorder = None
    if args.record_dir is not None:
        if not args.task.strip():
            print("--task is required when --record-dir is enabled.", file=sys.stderr)
            return 3
        dataset_recorder = RealmanTeleopDatasetRecorder(
            record_dir=args.record_dir,
            task=args.task,
            record_hz=args.record_hz,
            front_camera_serial=args.front_camera_serial,
            wrist_camera_serial=args.wrist_camera_serial,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            camera_fps=args.camera_fps,
            jpeg_quality=args.record_jpeg_quality,
        )
        dataset_recorder.start()

    client = RealmanSdkClient(args.arm_ip, args.arm_port)
    client.connect()
    try:
        if not args.disable_hand_follow and args.hand_backend == "brainco_ethercat":
            if not can_import_brainco_sdk():
                print(
                    "BrainCo SDK is not importable in this Python environment. "
                    "Install bc-stark-sdk or switch to another hand backend.",
                    file=sys.stderr,
                )
                return 4
            brainco_hand_client = BraincoEthercatHandClient(
                master_pos=args.brainco_master_pos,
                slave_pos=args.brainco_slave_pos,
                cycle_ns=args.brainco_cycle_ns,
                command_duration_ms=max(1, int(round(args.hand_control_dt * 1000.0))),
            )
            brainco_hand_client.connect()
            print(
                "[BrainCo] EtherCAT hand connected:",
                f"master={args.brainco_master_pos}",
                f"slave={args.brainco_slave_pos}",
                brainco_hand_client.device_info.description,
            )
        requires_hand_state = args.record_dir is not None
        wants_hand_backend = not args.disable_hand_follow
        uses_realman_hand_backend = (
            (wants_hand_backend and args.hand_backend in {"auto", "realman_follow", "realman_modbus"})
            or (requires_hand_state and brainco_hand_client is None)
        )
        if uses_realman_hand_backend:
            hand_modbus_ready = client.enable_hand_modbus(strict=requires_hand_state)
            if hand_modbus_ready:
                initial_hand_state = client.read_hand_joint_rad(strict=requires_hand_state)
            else:
                initial_hand_state = None
        elif brainco_hand_client is not None:
            initial_hand_state = hand_follower.brainco_ethercat_positions_to_qpos(
                brainco_hand_client.get_positions()
            )
        else:
            initial_hand_state = None

        if args.hand_speed >= 0:
            ret = client.set_hand_speed(args.hand_speed)
            print(f"[SDK] rm_set_hand_speed({args.hand_speed}) -> {ret}")
        if args.hand_force >= 0:
            ret = client.set_hand_force(args.hand_force)
            print(f"[SDK] rm_set_hand_force({args.hand_force}) -> {ret}")

        teleop = RealmanSdkHybridTeleop(
            config=config,
            client=client,
            wrist_port=args.wrist_port,
            hand_port=args.hand_port,
            hand_follower=hand_follower,
            brainco_hand_client=brainco_hand_client,
            joint_snapshot_path=args.joint_snapshot_path,
            arm_command_mode=args.arm_command_mode,
            hand_backend=args.hand_backend,
            hand_speed=args.hand_speed,
            hand_control_dt=args.hand_control_dt,
            canfd_trajectory_mode=args.canfd_trajectory_mode,
            canfd_radio=args.canfd_radio,
            control_dt=args.control_dt,
            feedback_source=args.feedback_source,
            seed_current_as_nominal=args.seed_current_as_nominal,
            dataset_recorder=dataset_recorder,
        )
        if initial_hand_state is not None and len(initial_hand_state) >= 6:
            teleop.latest_hand_state_qpos = np.array(initial_hand_state[:6], dtype=np.float32)
            teleop.last_hand_state_read_time = time.monotonic()
        teleop.run()
        return 0
    except KeyboardInterrupt:
        print("RealMan SDK hybrid teleop interrupted.")
        return 0
    finally:
        if dataset_recorder is not None:
            dataset_recorder.close()
        if brainco_hand_client is not None:
            brainco_hand_client.close()
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
