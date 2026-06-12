#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import os
import socket
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
V1_ROOT = SCRIPT_PATH.parent
VR_ROOT = V1_ROOT.parent
RM65_DOCKER_ROOT = VR_ROOT / "rm65_dex_docker"
RM65_SCRIPTS = RM65_DOCKER_ROOT / "scripts"
WS_SRC = RM65_DOCKER_ROOT / "workspace/rm65_dex_ws/src"
if str(RM65_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(RM65_SCRIPTS))

from hybrid_teleop_common import (  # noqa: E402
    ARM_JOINT_NAMES,
    HAND_ACTUATED_JOINT_NAMES,
    HAND_MIMIC_RULES,
    HybridArmController,
    HybridConfig,
    load_hybrid_config,
    override_config_joint_seed_from_snapshot,
    parse_hand_qpos_packet,
    parse_wrist_packet,
    poll_latest_packet,
)


def prepare_mujoco_render_backend() -> None:
    backend = os.environ.get("MUJOCO_GL", "").lower()
    if backend != "osmesa":
        return

    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")
    os.environ.setdefault("GALLIUM_DRIVER", "llvmpipe")

    preload_libs = [
        "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
        "/usr/lib/x86_64-linux-gnu/libOSMesa.so",
        "/usr/lib/x86_64-linux-gnu/libGL.so",
        "/usr/lib/x86_64-linux-gnu/libOpenGL.so.0",
    ]
    for lib_path in preload_libs:
        if os.path.exists(lib_path):
            ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)


prepare_mujoco_render_backend()


try:
    import mujoco  # noqa: E402
    from mujoco import viewer as mj_viewer  # noqa: E402
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "MuJoCo Python package is not installed in the current environment.\n"
        "If you want to run this on the host, first install it in your active env, for example:\n"
        "  /home/xwen/anaconda3/envs/VR/bin/pip install mujoco\n"
        "Then rerun this script."
    ) from exc


ARM_URDF_PATH = WS_SRC / "rm_description/urdf/rm_65.urdf"
DEX_HAND_XACRO_PATH = WS_SRC / "rm65_dex_description/urdf/dex_hand.urdf.xacro"
DEFAULT_CONFIG_PATH = V1_ROOT / "config/wrist_realman_arm_only.yaml"
DEFAULT_INITIAL_SNAPSHOT_PATH = Path("/tmp/rm65_real_sdk_arm_snapshot.json")
DEFAULT_JOINT_SNAPSHOT_PATH = Path("/tmp/v1_mujoco_realman_arm_snapshot.json")


def parse_float_vector_arg(value: str, *, name: str, length: int = 6) -> np.ndarray:
    parts = value.replace(",", " ").split()
    if len(parts) != length:
        raise argparse.ArgumentTypeError(f"{name} expects {length} values, got {len(parts)}: {value}")
    try:
        return np.array([float(part) for part in parts], dtype=np.float32)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} expects float values: {value}") from exc


def parse_optional_float_vector_arg(
    value: str,
    *,
    name: str,
    length: int = 6,
) -> Optional[np.ndarray]:
    if value.strip().lower() in ("none", "off", "disable", "disabled"):
        return None
    return parse_float_vector_arg(value, name=name, length=length)


def apply_cli_config_overrides(config: HybridConfig, args: argparse.Namespace) -> HybridConfig:
    overrides: list[str] = []

    def set_scalar(attr: str, value) -> None:
        if value is None:
            return
        setattr(config, attr, value)
        overrides.append(f"{attr}={value}")

    set_scalar("position_lpf_alpha", args.position_lpf_alpha)
    set_scalar("rotation_lpf_alpha", args.rotation_lpf_alpha)
    set_scalar("pre_ik_position_gain", args.pre_ik_position_gain)
    set_scalar("pre_ik_rotation_gain", args.pre_ik_rotation_gain)
    set_scalar("dls_damping", args.dls_damping)
    set_scalar("dls_gain", args.dls_position_gain)
    set_scalar("dls_iterations", args.dls_iterations)
    set_scalar("max_joint_step", args.max_joint_step)
    set_scalar("orientation_deadband_rad", args.orientation_deadband_rad)
    set_scalar("max_position_offset", args.max_position_offset)
    set_scalar("max_rotation_error", args.max_rotation_error)
    set_scalar("trajectory_control_dt", args.trajectory_control_dt)

    if args.trajectory_smoother is not None and args.trajectory_smoother != "config":
        config.trajectory_smoother = args.trajectory_smoother
        overrides.append(f"trajectory_smoother={args.trajectory_smoother}")
    if args.dls_max_delta_per_joint is not None:
        config.dls_max_delta_per_joint = parse_optional_float_vector_arg(
            args.dls_max_delta_per_joint,
            name="--dls-max-delta-per-joint",
        )
        overrides.append(
            "dls_max_delta_per_joint="
            f"{None if config.dls_max_delta_per_joint is None else np.round(config.dls_max_delta_per_joint, 4).tolist()}"
        )
    if args.max_joint_step_per_joint is not None:
        config.max_joint_step_per_joint = parse_optional_float_vector_arg(
            args.max_joint_step_per_joint,
            name="--max-joint-step-per-joint",
        )
        overrides.append(
            "max_joint_step_per_joint="
            f"{None if config.max_joint_step_per_joint is None else np.round(config.max_joint_step_per_joint, 4).tolist()}"
        )
    if args.ruckig_max_velocity is not None:
        config.ruckig_max_velocity = parse_float_vector_arg(
            args.ruckig_max_velocity,
            name="--ruckig-max-velocity",
        )
        overrides.append(f"ruckig_max_velocity={np.round(config.ruckig_max_velocity, 4).tolist()}")
    if args.ruckig_max_acceleration is not None:
        config.ruckig_max_acceleration = parse_float_vector_arg(
            args.ruckig_max_acceleration,
            name="--ruckig-max-acceleration",
        )
        overrides.append(
            f"ruckig_max_acceleration={np.round(config.ruckig_max_acceleration, 4).tolist()}"
        )
    if args.ruckig_max_jerk is not None:
        config.ruckig_max_jerk = parse_float_vector_arg(
            args.ruckig_max_jerk,
            name="--ruckig-max-jerk",
        )
        overrides.append(f"ruckig_max_jerk={np.round(config.ruckig_max_jerk, 4).tolist()}")

    if overrides:
        print("v1 MuJoCo config overrides:", "; ".join(overrides))
    return config


def extract_dex_hand_body(xacro_path: Path, prefix: str = "hand_") -> str:
    text = xacro_path.read_text(encoding="utf-8")
    start_token = "<xacro:macro"
    start = text.find(start_token)
    if start < 0:
        raise RuntimeError(f"Failed to locate xacro:macro in {xacro_path}")
    body_start = text.find(">", start)
    body_end = text.rfind("</xacro:macro>")
    if body_start < 0 or body_end < 0 or body_end <= body_start:
        raise RuntimeError(f"Failed to extract dex-hand xacro body from {xacro_path}")
    body = text[body_start + 1 : body_end]
    return body.replace("${prefix}", prefix)


def build_arm_only_urdf() -> Path:
    arm_text = ARM_URDF_PATH.read_text(encoding="utf-8")
    hand_body = extract_dex_hand_body(DEX_HAND_XACRO_PATH, prefix="hand_")
    assembly = """
  <link name="scene_world"/>

  <joint name="scene_world_to_base" type="fixed">
    <parent link="scene_world"/>
    <child link="base_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>

  <link name="scene_floor_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="2.0 2.0 0.02"/></geometry>
      <material name="scene_floor_mat"><color rgba="0.74 0.74 0.74 1"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="2.0 2.0 0.02"/></geometry>
    </collision>
  </link>
  <joint name="scene_world_to_floor" type="fixed">
    <parent link="scene_world"/>
    <child link="scene_floor_link"/>
    <origin xyz="0 0 -0.01" rpy="0 0 0"/>
  </joint>

  <link name="rm65_tool0"/>
  <joint name="Link6_to_rm65_tool0" type="fixed">
    <parent link="Link6"/>
    <child link="rm65_tool0"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>

  <joint name="rm65_tool0_to_dex_hand_base" type="fixed">
    <parent link="rm65_tool0"/>
    <child link="hand_base_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>
"""
    combined = arm_text.replace("</robot>", f"{hand_body}\n{assembly}\n</robot>")
    combined = combined.replace(
        "package://rm_description/",
        f"{(WS_SRC / 'rm_description').as_posix()}/",
    )
    combined = combined.replace(
        "package://rm65_dex_description/",
        f"{(WS_SRC / 'rm65_dex_description').as_posix()}/",
    )

    temp_dir = Path(tempfile.gettempdir()) / "v1_rm65_arm_only_mujoco"
    temp_dir.mkdir(parents=True, exist_ok=True)
    urdf_path = temp_dir / "rm65_arm_only.urdf"
    urdf_path.write_text(combined, encoding="utf-8")
    return urdf_path


class MujocoRealmanArmOnlyValidator:
    def __init__(
        self,
        config: HybridConfig,
        wrist_port: int,
        hand_port: int,
        joint_snapshot_path: Optional[Path],
        initial_snapshot_path: Optional[Path],
    ):
        self.config = config
        self.wrist_port = wrist_port
        self.hand_port = hand_port
        self.joint_snapshot_path = joint_snapshot_path
        self.initial_snapshot_path = initial_snapshot_path
        self.snapshot_write_period = max(
            float(os.environ.get("MUJOCO_JOINT_SNAPSHOT_PERIOD", "0.10")),
            0.0,
        )
        self.last_snapshot_write_time = 0.0
        self.last_status_log_time = 0.0
        self.render_backend = os.environ.get("MUJOCO_GL", "").lower()
        self.urdf_path = build_arm_only_urdf()
        self.model = mujoco.MjModel.from_xml_path(str(self.urdf_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.002
        self.model.opt.iterations = max(int(self.model.opt.iterations), 80)
        self.arm_controller = HybridArmController(config)
        self.arm_qpos = config.initial_joint_positions.astype(np.float32).copy()
        self.hand_qpos = np.zeros(6, dtype=np.float32)
        self.try_load_initial_snapshot()
        self.latest_wrist_time: Optional[float] = None
        self.latest_wrist: Optional[np.ndarray] = None
        self.latest_landmarks: Optional[np.ndarray] = None
        self.latest_hand_time: Optional[float] = None
        self.latest_hand_qpos: Optional[np.ndarray] = None
        self.wrist_sock = self.make_udp_socket(wrist_port)
        self.hand_sock = self.make_udp_socket(hand_port)
        self.qpos_addr = self.build_qpos_addr()
        self.controlled_dof_indices = self.build_controlled_dof_indices()
        self.apply_joint_state()
        self.write_joint_snapshot(force=True)
        print(f"Loaded arm-only MuJoCo model from {self.urdf_path}")
        print(f"Render backend: {self.render_backend or 'glfw/viewer'}")
        if self.initial_snapshot_path is not None:
            print(f"Arm-only MuJoCo initial arm snapshot source: {self.initial_snapshot_path}")
        if self.joint_snapshot_path is not None:
            print(f"Writing arm-only MuJoCo joint snapshot to {self.joint_snapshot_path}")

    @staticmethod
    def make_udp_socket(port: int) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", port))
        sock.setblocking(False)
        return sock

    def build_qpos_addr(self) -> dict[str, int]:
        mapping: dict[str, int] = {}
        for name in ARM_JOINT_NAMES + HAND_ACTUATED_JOINT_NAMES + list(HAND_MIMIC_RULES.keys()):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid != -1:
                mapping[name] = int(self.model.jnt_qposadr[jid])
        return mapping

    def build_controlled_dof_indices(self) -> np.ndarray:
        dof_indices: list[int] = []
        controlled_joint_names = ARM_JOINT_NAMES + HAND_ACTUATED_JOINT_NAMES + list(HAND_MIMIC_RULES.keys())
        for joint_name in controlled_joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id == -1:
                continue
            dof_adr = int(self.model.jnt_dofadr[joint_id])
            if dof_adr >= 0:
                dof_indices.append(dof_adr)
        return np.array(sorted(set(dof_indices)), dtype=np.int32)

    def try_load_initial_snapshot(self) -> None:
        if self.initial_snapshot_path is None:
            return
        if not self.initial_snapshot_path.exists():
            print(
                "Arm-only MuJoCo initial snapshot was requested but the file does not exist:"
                f" {self.initial_snapshot_path}"
            )
            return
        try:
            data = json.loads(self.initial_snapshot_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(
                "Arm-only MuJoCo initial snapshot could not be parsed:"
                f" path={self.initial_snapshot_path} error={exc}"
            )
            return

        arm_qpos = data.get("arm_qpos")
        if isinstance(arm_qpos, list) and len(arm_qpos) >= 6:
            self.arm_qpos = np.array(arm_qpos[:6], dtype=np.float32)
            print(
                "Initialized arm-only MuJoCo joint state from snapshot:"
                f" arm_qpos={np.round(self.arm_qpos, 4).tolist()}"
            )
        else:
            print(
                "Arm-only MuJoCo initial snapshot does not contain a usable arm_qpos[6]:"
                f" {self.initial_snapshot_path}"
            )
        hand_qpos = data.get("hand_qpos")
        if isinstance(hand_qpos, list) and len(hand_qpos) >= 6:
            self.hand_qpos = np.array(hand_qpos[:6], dtype=np.float32)

    def stabilize_controlled_dofs(self) -> None:
        if self.controlled_dof_indices.size == 0:
            return
        self.data.qvel[self.controlled_dof_indices] = 0.0
        self.data.qacc_warmstart[self.controlled_dof_indices] = 0.0
        self.data.qfrc_applied[self.controlled_dof_indices] = 0.0

    def set_joint_qpos(self, joint_name: str, value: float) -> None:
        addr = self.qpos_addr.get(joint_name)
        if addr is not None:
            self.data.qpos[addr] = value

    def apply_joint_state(self) -> None:
        for joint_name, value in zip(ARM_JOINT_NAMES, self.arm_qpos):
            self.set_joint_qpos(joint_name, float(value))
        actuated = {
            name: float(value)
            for name, value in zip(HAND_ACTUATED_JOINT_NAMES, self.hand_qpos)
        }
        for name, value in actuated.items():
            self.set_joint_qpos(name, value)
        for mimic_name, (source_name, multiplier, offset) in HAND_MIMIC_RULES.items():
            if source_name in actuated:
                self.set_joint_qpos(mimic_name, actuated[source_name] * multiplier + offset)
        self.stabilize_controlled_dofs()
        mujoco.mj_forward(self.model, self.data)

    def write_joint_snapshot(self, force: bool = False) -> None:
        if self.joint_snapshot_path is None:
            return
        now_monotonic = time.monotonic()
        if (
            not force
            and self.snapshot_write_period > 0.0
            and now_monotonic - self.last_snapshot_write_time < self.snapshot_write_period
        ):
            return
        payload = {
            "timestamp": time.time(),
            "monotonic_time": now_monotonic,
            "arm_joint_names": list(ARM_JOINT_NAMES),
            "arm_qpos": [float(value) for value in self.arm_qpos],
            "hand_qpos": [float(value) for value in self.hand_qpos],
            "axis_mapping": self.config.axis_mapping,
            "position_target_mode": self.config.position_target_mode,
            "rotation_control_mode": self.config.rotation_control_mode,
            "orientation_target_mode": self.config.orientation_target_mode,
            "wrist_to_hand_quat_xyzw": [
                float(value) for value in self.config.wrist_to_hand_quat_xyzw
            ],
            "source": "v1_mujoco_realman_no_table_with_hand_validate",
        }
        self.joint_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.joint_snapshot_path.with_suffix(self.joint_snapshot_path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        temp_path.replace(self.joint_snapshot_path)
        self.last_snapshot_write_time = now_monotonic

    def log_status(self, now: float) -> None:
        if now - self.last_status_log_time < 1.0:
            return
        self.last_status_log_time = now
        if self.latest_wrist_time is None:
            print("v1 MuJoCo: waiting for wrist UDP", f"port={self.wrist_port}")
            return
        print("v1 MuJoCo:", f"arm_q={np.round(self.arm_qpos, 3).tolist()}")

    def step(self) -> None:
        wrist = poll_latest_packet(self.wrist_sock, parse_wrist_packet)
        if wrist is not None:
            self.latest_wrist, self.latest_landmarks = wrist
            self.latest_wrist_time = time.monotonic()
        hand_qpos = poll_latest_packet(self.hand_sock, parse_hand_qpos_packet)
        if hand_qpos is not None:
            self.latest_hand_qpos = hand_qpos[:6]
            self.latest_hand_time = time.monotonic()

        now = time.monotonic()
        if (
            self.latest_wrist is not None
            and self.latest_wrist_time is not None
            and now - self.latest_wrist_time <= self.config.packet_timeout
        ):
            self.arm_qpos = self.arm_controller.update(
                self.latest_wrist,
                self.arm_qpos,
                self.latest_landmarks,
            ).astype(np.float32)

        if (
            self.latest_hand_qpos is not None
            and self.latest_hand_time is not None
            and now - self.latest_hand_time <= 1.0
        ):
            self.hand_qpos = self.latest_hand_qpos.astype(np.float32)

        self.apply_joint_state()
        self.write_joint_snapshot()
        self.log_status(now)
        for _ in range(4):
            self.stabilize_controlled_dofs()
            mujoco.mj_step(self.model, self.data)

    @staticmethod
    def configure_camera(camera) -> None:
        camera.azimuth = 145
        camera.elevation = -22
        camera.distance = 1.55
        camera.lookat[:] = [-0.15, 0.0, 0.36]

    def run_viewer(self) -> None:
        with mj_viewer.launch_passive(self.model, self.data) as viewer:
            self.configure_camera(viewer.cam)
            print(
                f"Listening for wrist_pose UDP on :{self.wrist_port} and hand_qpos UDP on :{self.hand_port}."
            )
            while viewer.is_running():
                self.step()
                viewer.sync()
                time.sleep(0.01)
        if os.environ.get("MUJOCO_GLFW_FAST_EXIT", "1") == "1":
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)

    def run_osmesa(self) -> None:
        import cv2

        camera = mujoco.MjvCamera()
        self.configure_camera(camera)
        render_height = int(os.environ.get("MUJOCO_OSMESA_HEIGHT", "960"))
        render_width = int(os.environ.get("MUJOCO_OSMESA_WIDTH", "1280"))
        renderer = mujoco.Renderer(self.model, height=render_height, width=render_width)
        window_name = "v1 MuJoCo RealMan Arm Only (OSMesa)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(window_name, render_width, render_height)

        print(
            f"Listening for wrist_pose UDP on :{self.wrist_port} and hand_qpos UDP on :{self.hand_port}."
        )
        print(
            "Using Mesa CPU rendering through OSMesa."
            f" Window/render size: {render_width}x{render_height}. Press 'q' in the window to quit."
        )

        while True:
            self.step()
            renderer.update_scene(self.data, camera=camera)
            rgb = renderer.render()
            bgr = rgb[:, :, ::-1]
            cv2.imshow(window_name, bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            time.sleep(0.01)

        cv2.destroyAllWindows()

    def run(self) -> None:
        if self.render_backend == "osmesa":
            self.run_osmesa()
            return
        self.run_viewer()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a v1 MuJoCo no-table scene with the RealMan arm and dex-hand, while "
            "reusing the same DLS + regularization + Ruckig controller logic as the mainline validator."
        )
    )
    default_initial_snapshot = os.environ.get(
        "V1_REALMAN_INITIAL_SNAPSHOT_PATH",
        str(DEFAULT_INITIAL_SNAPSHOT_PATH),
    ).strip()
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the v1 RealMan no-table wrist bridge config yaml.",
    )
    parser.add_argument("--wrist-port", type=int, default=5005, help="UDP port for forwarded wrist_pose.")
    parser.add_argument("--hand-port", type=int, default=5010, help="UDP port for forwarded hand_qpos.")
    parser.add_argument(
        "--joint-snapshot-path",
        type=Path,
        default=Path(
            os.environ.get("V1_MUJOCO_JOINT_SNAPSHOT_PATH", str(DEFAULT_JOINT_SNAPSHOT_PATH))
        ),
        help="JSON file that is periodically updated with the current v1 MuJoCo arm joint positions.",
    )
    parser.add_argument(
        "--initial-snapshot-path",
        type=Path,
        default=Path(default_initial_snapshot) if default_initial_snapshot else None,
        help="Optional JSON snapshot used to initialize the arm qpos before teleop starts.",
    )
    parser.add_argument(
        "--position-lpf-alpha",
        type=float,
        default=None,
        help="Override wrist position EMA gain. 1.0 disables this input smoothing.",
    )
    parser.add_argument(
        "--rotation-lpf-alpha",
        type=float,
        default=None,
        help="Override wrist quaternion EMA gain. 1.0 disables this input smoothing.",
    )
    parser.add_argument(
        "--pre-ik-position-gain",
        type=float,
        default=None,
        help="Override IK target position smoothing gain. 1.0 disables pre-IK position smoothing.",
    )
    parser.add_argument(
        "--pre-ik-rotation-gain",
        type=float,
        default=None,
        help="Override IK target rotation smoothing gain. 1.0 disables pre-IK rotation smoothing.",
    )
    parser.add_argument(
        "--dls-damping",
        type=float,
        default=None,
        help="Override DLS damping.",
    )
    parser.add_argument(
        "--dls-position-gain",
        type=float,
        default=None,
        help="Override DLS position/orientation step gain.",
    )
    parser.add_argument(
        "--dls-iterations",
        type=int,
        default=None,
        help="Override DLS iterations per control step.",
    )
    parser.add_argument(
        "--dls-max-delta-per-joint",
        type=str,
        default=None,
        help="Override DLS per-joint delta limits, comma/space separated 6 floats, or 'none'.",
    )
    parser.add_argument(
        "--max-joint-step",
        type=float,
        default=None,
        help="Override post-IK global joint step limit. 0 disables this limit.",
    )
    parser.add_argument(
        "--max-joint-step-per-joint",
        type=str,
        default=None,
        help="Override post-IK per-joint step limits, comma/space separated 6 floats, or 'none'.",
    )
    parser.add_argument(
        "--trajectory-smoother",
        choices=["config", "auto", "ruckig", "accel_limited", "none"],
        default=None,
        help="Override joint trajectory smoother. 'config' leaves the YAML value unchanged.",
    )
    parser.add_argument(
        "--trajectory-control-dt",
        type=float,
        default=None,
        help="Override trajectory smoother dt.",
    )
    parser.add_argument(
        "--ruckig-max-velocity",
        type=str,
        default=None,
        help="Override Ruckig max velocity, comma/space separated 6 floats.",
    )
    parser.add_argument(
        "--ruckig-max-acceleration",
        type=str,
        default=None,
        help="Override Ruckig max acceleration, comma/space separated 6 floats.",
    )
    parser.add_argument(
        "--ruckig-max-jerk",
        type=str,
        default=None,
        help="Override Ruckig max jerk, comma/space separated 6 floats.",
    )
    parser.add_argument(
        "--orientation-deadband-rad",
        type=float,
        default=None,
        help="Override orientation deadband in radians.",
    )
    parser.add_argument(
        "--max-position-offset",
        type=float,
        default=None,
        help="Override max Cartesian position offset from calibration origin.",
    )
    parser.add_argument(
        "--max-rotation-error",
        type=float,
        default=None,
        help="Override max orientation error from calibration origin.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_hybrid_config(args.config)
    config = override_config_joint_seed_from_snapshot(
        config,
        args.initial_snapshot_path,
        log_prefix="[v1 MuJoCo]",
    )
    config = apply_cli_config_overrides(config, args)
    validator = MujocoRealmanArmOnlyValidator(
        config=config,
        wrist_port=args.wrist_port,
        hand_port=args.hand_port,
        joint_snapshot_path=args.joint_snapshot_path,
        initial_snapshot_path=args.initial_snapshot_path,
    )
    validator.run()


if __name__ == "__main__":
    main()
