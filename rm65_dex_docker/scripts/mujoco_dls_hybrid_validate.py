#!/usr/bin/env python3
import argparse
import ctypes
import json
import os
import re
import socket
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml


SCRIPT_PATH = Path(__file__).resolve()
RM65_DOCKER_ROOT = SCRIPT_PATH.parent.parent
WS_SRC = RM65_DOCKER_ROOT / "workspace/rm65_dex_ws/src"
from hybrid_teleop_common import (  # noqa: E402
    ARM_JOINT_NAMES,
    DEFAULT_CONFIG_PATH,
    HAND_ACTUATED_JOINT_NAMES,
    HAND_MIMIC_RULES,
    HybridArmController,
    HybridConfig,
    load_hybrid_config,
    parse_hand_qpos_packet,
    parse_wrist_packet,
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
def extract_dex_hand_body(xacro_path: Path, prefix: str = "hand_") -> str:
    text = xacro_path.read_text(encoding="utf-8")
    match = re.search(r"<xacro:macro[^>]*>(.*)</xacro:macro>", text, re.S)
    if not match:
        raise RuntimeError(f"Failed to extract xacro:macro body from {xacro_path}")
    body = match.group(1)
    return body.replace("${prefix}", prefix)


def build_combined_urdf() -> Path:
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
      <geometry><box size="2.4 2.4 0.02"/></geometry>
      <material name="scene_floor_mat"><color rgba="0.72 0.72 0.72 1"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="2.4 2.4 0.02"/></geometry>
    </collision>
  </link>
  <joint name="scene_world_to_floor" type="fixed">
    <parent link="scene_world"/>
    <child link="scene_floor_link"/>
    <origin xyz="0 0 -0.01" rpy="0 0 0"/>
  </joint>

  <link name="task_table_top_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.55 0.45 0.04"/></geometry>
      <material name="table_top_mat"><color rgba="0.62 0.47 0.31 1"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.55 0.45 0.04"/></geometry>
    </collision>
  </link>
  <joint name="scene_world_to_table_top" type="fixed">
    <parent link="scene_world"/>
    <child link="task_table_top_link"/>
    <origin xyz="-0.39 0.00 0.25" rpy="0 0 0"/>
  </joint>

  <link name="task_table_leg_fl_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.04 0.04 0.23"/></geometry>
      <material name="table_leg_mat"><color rgba="0.39 0.27 0.16 1"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.04 0.04 0.23"/></geometry>
    </collision>
  </link>
  <joint name="scene_world_to_table_leg_fl" type="fixed">
    <parent link="scene_world"/>
    <child link="task_table_leg_fl_link"/>
    <origin xyz="-0.62 0.18 0.115" rpy="0 0 0"/>
  </joint>

  <link name="task_table_leg_fr_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.04 0.04 0.23"/></geometry>
      <material name="table_leg_mat_fr"><color rgba="0.39 0.27 0.16 1"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.04 0.04 0.23"/></geometry>
    </collision>
  </link>
  <joint name="scene_world_to_table_leg_fr" type="fixed">
    <parent link="scene_world"/>
    <child link="task_table_leg_fr_link"/>
    <origin xyz="-0.62 -0.18 0.115" rpy="0 0 0"/>
  </joint>

  <link name="task_table_leg_rl_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.04 0.04 0.23"/></geometry>
      <material name="table_leg_mat_rl"><color rgba="0.39 0.27 0.16 1"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.04 0.04 0.23"/></geometry>
    </collision>
  </link>
  <joint name="scene_world_to_table_leg_rl" type="fixed">
    <parent link="scene_world"/>
    <child link="task_table_leg_rl_link"/>
    <origin xyz="-0.16 0.18 0.115" rpy="0 0 0"/>
  </joint>

  <link name="task_table_leg_rr_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.04 0.04 0.23"/></geometry>
      <material name="table_leg_mat_rr"><color rgba="0.39 0.27 0.16 1"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.04 0.04 0.23"/></geometry>
    </collision>
  </link>
  <joint name="scene_world_to_table_leg_rr" type="fixed">
    <parent link="scene_world"/>
    <child link="task_table_leg_rr_link"/>
    <origin xyz="-0.16 -0.18 0.115" rpy="0 0 0"/>
  </joint>

  <link name="task_cube_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.06"/>
      <inertia ixx="0.000025" ixy="0.0" ixz="0.0" iyy="0.000025" iyz="0.0" izz="0.000025"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.03 0.03 0.03"/></geometry>
      <material name="task_cube_mat"><color rgba="0.84 0.25 0.22 1"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.05 0.05 0.05"/></geometry>
    </collision>
  </link>
  <joint name="scene_world_to_task_cube" type="floating">
    <parent link="scene_world"/>
    <child link="task_cube_link"/>
    <origin xyz="-0.40 -0.03 0.296" rpy="0 0 0"/>
  </joint>

  <link name="task_bin_base_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.18 0.18 0.02"/></geometry>
      <material name="task_bin_mat"><color rgba="0.20 0.48 0.76 0.95"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.18 0.18 0.02"/></geometry>
    </collision>
  </link>
  <joint name="scene_world_to_task_bin_base" type="fixed">
    <parent link="scene_world"/>
    <child link="task_bin_base_link"/>
    <origin xyz="-0.30 0.17 0.28" rpy="0 0 0"/>
  </joint>

  <link name="task_bin_wall_front_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.18 0.01 0.08"/></geometry>
      <material name="task_bin_wall_front_mat"><color rgba="0.18 0.44 0.70 0.95"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.18 0.01 0.08"/></geometry>
    </collision>
  </link>
  <joint name="scene_world_to_task_bin_wall_front" type="fixed">
    <parent link="scene_world"/>
    <child link="task_bin_wall_front_link"/>
    <origin xyz="-0.30 0.255 0.32" rpy="0 0 0"/>
  </joint>

  <link name="task_bin_wall_back_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.18 0.01 0.08"/></geometry>
      <material name="task_bin_wall_back_mat"><color rgba="0.18 0.44 0.70 0.95"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.18 0.01 0.08"/></geometry>
    </collision>
  </link>
  <joint name="scene_world_to_task_bin_wall_back" type="fixed">
    <parent link="scene_world"/>
    <child link="task_bin_wall_back_link"/>
    <origin xyz="-0.30 0.085 0.32" rpy="0 0 0"/>
  </joint>

  <link name="task_bin_wall_left_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.01 0.18 0.08"/></geometry>
      <material name="task_bin_wall_left_mat"><color rgba="0.18 0.44 0.70 0.95"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.01 0.18 0.08"/></geometry>
    </collision>
  </link>
  <joint name="scene_world_to_task_bin_wall_left" type="fixed">
    <parent link="scene_world"/>
    <child link="task_bin_wall_left_link"/>
    <origin xyz="-0.385 0.17 0.32" rpy="0 0 0"/>
  </joint>

  <link name="task_bin_wall_right_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.01 0.18 0.08"/></geometry>
      <material name="task_bin_wall_right_mat"><color rgba="0.18 0.44 0.70 0.95"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.01 0.18 0.08"/></geometry>
    </collision>
  </link>
  <joint name="scene_world_to_task_bin_wall_right" type="fixed">
    <parent link="scene_world"/>
    <child link="task_bin_wall_right_link"/>
    <origin xyz="-0.215 0.17 0.32" rpy="0 0 0"/>
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
    replacements = {
        "package://rm_description/": f"{(WS_SRC / 'rm_description').as_posix()}/",
        "package://rm65_dex_description/": f"{(WS_SRC / 'rm65_dex_description').as_posix()}/",
    }
    for src, dst in replacements.items():
        combined = combined.replace(src, dst)

    temp_dir = Path(tempfile.gettempdir()) / "rm65_revo2_mujoco"
    temp_dir.mkdir(parents=True, exist_ok=True)
    urdf_path = temp_dir / "rm65_revo2_combined.urdf"
    urdf_path.write_text(combined, encoding="utf-8")
    return urdf_path


class MujocoHybridValidator:
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
        self.render_backend = os.environ.get("MUJOCO_GL", "").lower()
        self.urdf_path = build_combined_urdf()
        self.model = mujoco.MjModel.from_xml_path(str(self.urdf_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.002
        self.model.opt.iterations = max(int(self.model.opt.iterations), 80)
        self.arm_controller = HybridArmController(config)
        self.arm_qpos = config.initial_joint_positions.astype(np.float32).copy()
        self.hand_qpos = np.zeros(6, dtype=np.float32)
        self.try_load_initial_snapshot()
        self.latest_wrist_time: Optional[float] = None
        self.latest_hand_time: Optional[float] = None
        self.latest_wrist: Optional[np.ndarray] = None
        self.latest_landmarks: Optional[np.ndarray] = None
        self.latest_hand_qpos: Optional[np.ndarray] = None
        self.wrist_sock = self.make_udp_socket(wrist_port)
        self.hand_sock = self.make_udp_socket(hand_port)
        self.qpos_addr = self.build_qpos_addr()
        self.controlled_dof_indices = self.build_controlled_dof_indices()
        self.configure_task_contacts()
        self.apply_joint_state()
        self.write_joint_snapshot(force=True)
        print(f"Loaded MuJoCo model from {self.urdf_path}")
        print(f"Render backend: {self.render_backend or 'glfw/viewer'}")
        if self.joint_snapshot_path is not None:
            print(f"Writing MuJoCo arm joint snapshot to {self.joint_snapshot_path}")
        if self.initial_snapshot_path is not None:
            print(f"MuJoCo initial arm snapshot source: {self.initial_snapshot_path}")

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
                "MuJoCo initial snapshot was requested but the file does not exist:"
                f" {self.initial_snapshot_path}"
            )
            return
        try:
            data = json.loads(self.initial_snapshot_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(
                "MuJoCo initial snapshot could not be parsed:"
                f" path={self.initial_snapshot_path} error={exc}"
            )
            return

        arm_qpos = data.get("arm_qpos")
        if isinstance(arm_qpos, list) and len(arm_qpos) >= 6:
            self.arm_qpos = np.array(arm_qpos[:6], dtype=np.float32)
        else:
            print(
                "MuJoCo initial snapshot does not contain a usable arm_qpos[6]:"
                f" {self.initial_snapshot_path}"
            )

        hand_qpos = data.get("hand_qpos")
        if isinstance(hand_qpos, list) and len(hand_qpos) >= 6:
            self.hand_qpos = np.array(hand_qpos[:6], dtype=np.float32)
        print(
            "Initialized MuJoCo joint state from snapshot:"
            f" arm_qpos={np.round(self.arm_qpos, 4).tolist()}"
        )

    def set_body_geom_properties(
        self,
        body_name: str,
        friction: np.ndarray,
        condim: int = 4,
        margin: float = 0.002,
    ) -> None:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            return
        geom_adr = int(self.model.body_geomadr[body_id])
        geom_num = int(self.model.body_geomnum[body_id])
        for geom_id in range(geom_adr, geom_adr + geom_num):
            self.model.geom_friction[geom_id] = friction
            self.model.geom_condim[geom_id] = condim
            self.model.geom_margin[geom_id] = margin

    def configure_task_contacts(self) -> None:
        self.set_body_geom_properties(
            "task_table_top_link",
            np.array([0.85, 0.03, 0.01], dtype=np.float32),
            condim=4,
            margin=0.002,
        )
        self.set_body_geom_properties(
            "task_cube_link",
            np.array([1.10, 0.05, 0.02], dtype=np.float32),
            condim=4,
            margin=0.003,
        )
        self.set_body_geom_properties(
            "task_bin_base_link",
            np.array([0.75, 0.02, 0.01], dtype=np.float32),
            condim=4,
            margin=0.002,
        )
        for wall_name in (
            "task_bin_wall_front_link",
            "task_bin_wall_back_link",
            "task_bin_wall_left_link",
            "task_bin_wall_right_link",
        ):
            self.set_body_geom_properties(
                wall_name,
                np.array([0.70, 0.02, 0.01], dtype=np.float32),
                condim=4,
                margin=0.002,
            )
        for body_id in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name and body_name.startswith("hand_"):
                self.set_body_geom_properties(
                    body_name,
                    np.array([1.45, 0.08, 0.02], dtype=np.float32),
                    condim=4,
                    margin=0.003,
                )
        cube_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "scene_world_to_task_cube"
        )
        if cube_joint_id != -1:
            dof_adr = int(self.model.jnt_dofadr[cube_joint_id])
            if dof_adr >= 0:
                self.model.dof_damping[dof_adr : dof_adr + 3] = 0.15
                self.model.dof_damping[dof_adr + 3 : dof_adr + 6] = 0.01

    def stabilize_controlled_dofs(self) -> None:
        if self.controlled_dof_indices.size == 0:
            return
        self.data.qvel[self.controlled_dof_indices] = 0.0
        self.data.qacc_warmstart[self.controlled_dof_indices] = 0.0
        self.data.qfrc_applied[self.controlled_dof_indices] = 0.0

    def set_joint_qpos(self, joint_name: str, value: float):
        addr = self.qpos_addr.get(joint_name)
        if addr is not None:
            self.data.qpos[addr] = value

    def apply_joint_state(self):
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

    def write_joint_snapshot(self, force: bool = False):
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
            "rotation_control_mode": self.config.rotation_control_mode,
            "source": "mujoco_dls_hybrid_validate",
        }
        snapshot_path = self.joint_snapshot_path
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = Path(f"{snapshot_path}.tmp")
        temp_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        temp_path.replace(snapshot_path)
        self.last_snapshot_write_time = now_monotonic

    def poll_socket(self, sock: socket.socket, parser):
        latest = None
        try:
            while True:
                packet, _ = sock.recvfrom(65535)
                parsed = parser(packet)
                if parsed is not None:
                    latest = parsed
        except BlockingIOError:
            pass
        return latest

    def step(self):
        wrist = self.poll_socket(self.wrist_sock, parse_wrist_packet)
        if wrist is not None:
            self.latest_wrist, self.latest_landmarks = wrist
            self.latest_wrist_time = time.monotonic()
        hand_qpos = self.poll_socket(self.hand_sock, parse_hand_qpos_packet)
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
        for _ in range(4):
            self.stabilize_controlled_dofs()
            mujoco.mj_step(self.model, self.data)

    @staticmethod
    def configure_camera(camera):
        camera.azimuth = 138
        camera.elevation = -24
        camera.distance = 1.85
        camera.lookat[:] = [-0.34, 0.03, 0.34]

    def run_viewer(self):
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

    def run_osmesa(self):
        import cv2  # lazy import so GLFW path does not depend on OpenCV UI

        camera = mujoco.MjvCamera()
        self.configure_camera(camera)
        render_height = int(os.environ.get("MUJOCO_OSMESA_HEIGHT", "960"))
        render_width = int(os.environ.get("MUJOCO_OSMESA_WIDTH", "1280"))
        renderer = mujoco.Renderer(self.model, height=render_height, width=render_width)
        window_name = "MuJoCo Hybrid Validator (OSMesa)"
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

    def run(self):
        if self.render_backend == "osmesa":
            self.run_osmesa()
            return
        self.run_viewer()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate the current RM65+Revo2 hybrid Quest control in MuJoCo."
    )
    default_initial_snapshot = os.environ.get("MUJOCO_INITIAL_SNAPSHOT_PATH", "").strip()
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to a wrist bridge config yaml. The current hybrid parameters are read from here.",
    )
    parser.add_argument("--wrist-port", type=int, default=5005, help="UDP port for forwarded wrist_pose.")
    parser.add_argument("--hand-port", type=int, default=5010, help="UDP port for forwarded hand_qpos.")
    parser.add_argument(
        "--joint-snapshot-path",
        type=Path,
        default=Path(
            os.environ.get("MUJOCO_JOINT_SNAPSHOT_PATH", "/tmp/rm65_mujoco_arm_snapshot.json")
        ),
        help="JSON file that is periodically updated with the current MuJoCo arm joint positions.",
    )
    parser.add_argument(
        "--initial-snapshot-path",
        type=Path,
        default=Path(default_initial_snapshot) if default_initial_snapshot else None,
        help=(
            "Optional JSON snapshot used to initialize MuJoCo arm/hand qpos before teleop starts. "
            "Defaults to $MUJOCO_INITIAL_SNAPSHOT_PATH when set."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_hybrid_config(args.config)
    validator = MujocoHybridValidator(
        config,
        wrist_port=args.wrist_port,
        hand_port=args.hand_port,
        joint_snapshot_path=args.joint_snapshot_path,
        initial_snapshot_path=args.initial_snapshot_path,
    )
    validator.run()


if __name__ == "__main__":
    main()
