#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
import ctypes
from pathlib import Path
from typing import Optional

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
RM65_DOCKER_ROOT = SCRIPT_PATH.parent.parent
WS_SRC = RM65_DOCKER_ROOT / "workspace/rm65_dex_ws/src"
QUEST_BRIDGE_SRC = WS_SRC / "quest_bridge"
if str(QUEST_BRIDGE_SRC) not in sys.path:
    sys.path.insert(0, str(QUEST_BRIDGE_SRC))


def _try_load_system_libstdcpp() -> None:
    system_lib = Path("/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
    if not system_lib.is_file():
        return
    try:
        ctypes.CDLL(str(system_lib), mode=ctypes.RTLD_GLOBAL)
    except Exception:
        pass


_try_load_system_libstdcpp()

from quest_bridge.local_dls_ik import solve_local_dls_6d, solve_local_dls_position  # noqa: E402
from quest_bridge.realman_ik import RealManIK  # noqa: E402
from quest_bridge.trajectory_smoothing import HAS_RUCKIG, JointTrajectorySmoother  # noqa: E402


# Re-export the split modules through this legacy entry point so existing
# v1/MuJoCo/RealMan scripts keep their current import paths.
from teleop_core.config import (  # noqa: E402
    HybridConfig,
    load_arm_qpos_from_snapshot,
    load_hybrid_config,
    override_config_joint_seed_from_snapshot,
)
from teleop_core.constants import (  # noqa: E402
    ARM_JOINT_NAMES,
    CUBE_SIDE_MOUNT_CONFIG_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_WRIST_REGULARIZATION_JOINTS,
    HAND_ACTUATED_JOINT_NAMES,
    HAND_MIMIC_RULES,
    RM65_JOINT_LIMIT_MAX,
    RM65_JOINT_LIMIT_MIN,
)
from teleop_core.mapping import (  # noqa: E402
    is_vr_teleop_rotation_mapping,
    make_quest_to_robot_matrix,
    maybe_negate_relative_rot_xy,
    mirror_left_wrist_quat_to_right_like,
    quest_rotation_to_robot,
    vr_teleop_left_realman_rotation_to_robot,
    vr_teleop_left_sim_rotation_to_robot,
    vr_teleop_realman_rotation_to_robot,
    vr_teleop_sim_rotation_to_robot,
)
from teleop_core.math_utils import (  # noqa: E402
    limit_vector,
    low_pass_quat_xyzw,
    low_pass_vector,
    matrix_to_quat_xyzw,
    normalized_quat_xyzw,
    quat_xyzw_to_matrix,
    rotation_matrix_to_rotvec,
    rotvec_to_matrix,
    scale_rotation_matrix,
    smooth_rotation_matrix,
)
from teleop_core.udp_packets import (  # noqa: E402
    parse_hand_qpos_packet,
    parse_landmark_array,
    parse_wrist_packet,
    poll_latest_packet,
)


class HybridArmController:
    def __init__(self, config: HybridConfig):
        self.config = config
        self.quest_to_robot = make_quest_to_robot_matrix(config.axis_mapping)
        self.rotation_quest_to_robot = (
            self.quest_to_robot
            if is_vr_teleop_rotation_mapping(config.rotation_axis_mapping)
            else make_quest_to_robot_matrix(config.rotation_axis_mapping)
        )
        self.wrist_to_hand_rotation = quat_xyzw_to_matrix(config.wrist_to_hand_quat_xyzw)
        self.tool_to_hand_rotation = quat_xyzw_to_matrix(config.tool_to_hand_quat_xyzw)
        self.fk = RealManIK(
            lib_path=config.realman_lib_path,
            robot_model=config.robot_model,
            traversal_mode=False,
        )
        self.prev_wrist_position: Optional[np.ndarray] = None
        self.prev_wrist_rotation: Optional[np.ndarray] = None
        self.filtered_wrist_position: Optional[np.ndarray] = None
        self.filtered_wrist_quat_xyzw: Optional[np.ndarray] = None
        self.wrist_origin_position: Optional[np.ndarray] = None
        self.wrist_rotation_origin: Optional[np.ndarray] = None
        self.target_position: Optional[np.ndarray] = None
        self.target_rotation: Optional[np.ndarray] = None
        self.ee_origin: Optional[np.ndarray] = None
        self.ee_rotation_origin: Optional[np.ndarray] = None
        self.hand_origin_position: Optional[np.ndarray] = None
        self.hand_rotation_origin: Optional[np.ndarray] = None
        self.joint6_origin_position: Optional[float] = None
        self.target_joint6_position: Optional[float] = None
        self.command_target_position: Optional[np.ndarray] = None
        self.command_target_rotation: Optional[np.ndarray] = None
        self.last_command_positions: Optional[np.ndarray] = None
        self.last_command_velocities: Optional[np.ndarray] = None
        self.last_log_time = 0.0
        self.last_orientation_debug_time = 0.0
        self.trajectory_smoother = JointTrajectorySmoother(
            dof=6,
            dt=config.trajectory_control_dt,
            mode=config.trajectory_smoother,
            max_velocity=config.ruckig_max_velocity,
            max_acceleration=config.ruckig_max_acceleration,
            max_jerk=config.ruckig_max_jerk,
        )
        print(
            "Hybrid DLS smoother:"
            f" mode={self.trajectory_smoother.mode},"
            f" ruckig_available={HAS_RUCKIG},"
            f" dls_max_delta={None if config.dls_max_delta_per_joint is None else np.round(config.dls_max_delta_per_joint, 3).tolist()},"
            f" dls_reg=(cur={config.current_q_regularization_weight:.3f},"
            f" posture={config.posture_regularization_weight:.3f},"
            f" limit={config.joint_limit_weight:.3f},"
            f" wrist={config.wrist_regularization_weight:.3f})"
        )

    def should_use_landmark_frame(self, landmarks: Optional[np.ndarray]) -> bool:
        if landmarks is None:
            return False
        if self.config.orientation_input_source == "landmark_frame":
            return True
        if self.config.orientation_input_source == "wrist_quaternion":
            return False
        return True

    def map_quest_landmarks_to_robot(self, landmarks: np.ndarray) -> np.ndarray:
        return landmarks @ self.quest_to_robot.T

    @staticmethod
    def normalize_vector(vector: np.ndarray) -> Optional[np.ndarray]:
        norm = float(np.linalg.norm(vector))
        if norm < 1e-8:
            return None
        return vector / norm

    def landmark_frame_to_rotation(self, landmarks_robot: np.ndarray) -> Optional[np.ndarray]:
        if landmarks_robot.shape != (21, 3):
            return None
        wrist = landmarks_robot[0]
        index_mcp = landmarks_robot[5]
        middle_mcp = landmarks_robot[9]
        pinky_mcp = landmarks_robot[17]

        z_axis = self.normalize_vector(middle_mcp - wrist)
        if z_axis is None:
            return None

        x_raw = index_mcp - pinky_mcp
        x_axis = x_raw - float(np.dot(x_raw, z_axis)) * z_axis
        x_axis = self.normalize_vector(x_axis)
        if x_axis is None:
            return None

        y_axis = self.normalize_vector(np.cross(z_axis, x_axis))
        if y_axis is None:
            return None

        x_axis = self.normalize_vector(np.cross(y_axis, z_axis))
        if x_axis is None:
            return None

        return np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32)

    def resolve_wrist_tracking_inputs(
        self,
        wrist: np.ndarray,
        landmarks: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        wrist_position_raw = self.quest_to_robot @ wrist[:3]
        wrist_rotation_raw = quest_rotation_to_robot(
            wrist[3:7],
            self.rotation_quest_to_robot,
            self.config.rotation_axis_mapping,
        )
        wrist_quat_xyzw_raw = self.align_wrist_rotation_to_hand(wrist_rotation_raw)
        if not self.should_use_landmark_frame(landmarks):
            return wrist_position_raw, wrist_quat_xyzw_raw

        landmarks_robot = self.map_quest_landmarks_to_robot(landmarks)
        landmark_rotation = self.landmark_frame_to_rotation(landmarks_robot)
        if landmark_rotation is None:
            return wrist_position_raw, wrist_quat_xyzw_raw
        return wrist_position_raw, self.align_wrist_rotation_to_hand(landmark_rotation)

    def align_wrist_rotation_to_hand(self, wrist_rotation: np.ndarray) -> np.ndarray:
        return matrix_to_quat_xyzw(wrist_rotation @ self.wrist_to_hand_rotation)

    def tool_pose_to_hand_pose(self, tool_position: np.ndarray, tool_rotation: np.ndarray):
        hand_rotation = tool_rotation @ self.tool_to_hand_rotation
        hand_position = tool_position + tool_rotation @ self.config.tool_to_hand_translation
        return hand_position, hand_rotation

    def hand_pose_to_tool_pose(self, hand_position: np.ndarray, hand_rotation: np.ndarray):
        tool_rotation = hand_rotation @ self.tool_to_hand_rotation.T
        tool_position = hand_position - tool_rotation @ self.config.tool_to_hand_translation
        return tool_position, tool_rotation

    def joint6_axis_index(self) -> int:
        return {"x": 0, "y": 1, "z": 2}[self.config.joint6_axis]

    def apply_rotvec_deadband(self, rotvec: np.ndarray) -> np.ndarray:
        if self.config.orientation_deadband_rad <= 0.0:
            return rotvec.copy()
        abs_rotvec = np.abs(rotvec)
        reduced = np.maximum(abs_rotvec - self.config.orientation_deadband_rad, 0.0)
        return np.sign(rotvec) * reduced

    def relative_wrist_rotvec_local(self, wrist_rotation: np.ndarray) -> np.ndarray:
        if self.wrist_rotation_origin is None or self.hand_rotation_origin is None:
            return np.zeros(3, dtype=np.float32)
        relative_rotation_world = wrist_rotation @ self.wrist_rotation_origin.T
        relative_rotvec_world = rotation_matrix_to_rotvec(relative_rotation_world)
        relative_rotvec_local = self.hand_rotation_origin.T @ relative_rotvec_world
        return self.apply_rotvec_deadband(relative_rotvec_local)

    def twist_angle_about_axis(self, rotation_local: np.ndarray, axis_index: int) -> float:
        quat = matrix_to_quat_xyzw(rotation_local)
        axis = np.zeros(3, dtype=np.float32)
        axis[axis_index] = 1.0
        projected = axis * float(np.dot(quat[:3], axis))
        twist = np.array([projected[0], projected[1], projected[2], quat[3]], dtype=np.float32)
        norm = float(np.linalg.norm(twist))
        if norm < 1e-8:
            return 0.0
        twist = twist / norm
        if twist[3] < 0.0:
            twist = -twist
        sin_half = float(np.dot(twist[:3], axis))
        cos_half = float(twist[3])
        return float(2.0 * np.arctan2(sin_half, cos_half))

    def hybrid_target_hand_rotation(self, wrist_rotation: np.ndarray) -> np.ndarray:
        if self.hand_rotation_origin is None:
            return np.eye(3, dtype=np.float32)
        pose_rotvec_local = self.relative_wrist_rotvec_local(wrist_rotation)
        pose_rotvec_local[self.joint6_axis_index()] = 0.0
        pose_rotvec_local *= self.config.hybrid_pose_rotation_scale
        return self.hand_rotation_origin @ rotvec_to_matrix(pose_rotvec_local)

    def compute_joint6_target(self, wrist_rotation: np.ndarray, current_qpos: np.ndarray) -> float:
        if (
            self.wrist_rotation_origin is None
            or self.hand_rotation_origin is None
            or self.joint6_origin_position is None
        ):
            if self.target_joint6_position is not None:
                return float(self.target_joint6_position)
            return float(current_qpos[5])
        relative_rotation_world = wrist_rotation @ self.wrist_rotation_origin.T
        relative_rotation_local = (
            self.hand_rotation_origin.T @ relative_rotation_world @ self.hand_rotation_origin
        )
        axis_value = self.twist_angle_about_axis(
            relative_rotation_local,
            self.joint6_axis_index(),
        )
        if self.config.joint6_deadband_rad > 0.0:
            abs_axis = abs(axis_value)
            if abs_axis <= self.config.joint6_deadband_rad:
                axis_value = 0.0
            else:
                axis_value = np.sign(axis_value) * (abs_axis - self.config.joint6_deadband_rad)
        desired_joint6 = float(
            self.joint6_origin_position + axis_value * self.config.joint6_scale
        )
        desired_joint6 = float(np.clip(desired_joint6, self.config.joint6_min, self.config.joint6_max))
        if self.target_joint6_position is None:
            self.target_joint6_position = float(current_qpos[5])
        if self.config.max_joint6_step > 0.0:
            delta = desired_joint6 - self.target_joint6_position
            delta = float(np.clip(delta, -self.config.max_joint6_step, self.config.max_joint6_step))
            self.target_joint6_position = float(self.target_joint6_position + delta)
        else:
            self.target_joint6_position = desired_joint6
        return float(self.target_joint6_position)

    def limit_hand_target(self, target_hand_position: np.ndarray, target_hand_rotation: np.ndarray):
        if self.hand_origin_position is None or self.hand_rotation_origin is None:
            return target_hand_position, target_hand_rotation, False
        position_clipped = False
        if self.config.max_position_offset > 0.0:
            target_offset = target_hand_position - self.hand_origin_position
            limited_offset = limit_vector(target_offset, self.config.max_position_offset)
            position_clipped = np.linalg.norm(limited_offset - target_offset) > 1e-6
            target_hand_position = self.hand_origin_position + limited_offset
        if self.config.max_rotation_error > 0.0 and self.config.rotation_control_mode in ("full_pose", "hybrid"):
            target_rotvec = rotation_matrix_to_rotvec(
                target_hand_rotation @ self.hand_rotation_origin.T
            )
            limited_rotvec = limit_vector(target_rotvec, self.config.max_rotation_error)
            target_hand_rotation = rotvec_to_matrix(limited_rotvec) @ self.hand_rotation_origin
        return target_hand_position, target_hand_rotation, position_clipped

    def limit_joint_step(self, reference: np.ndarray, target: np.ndarray) -> np.ndarray:
        if self.config.max_joint_step_per_joint is not None:
            max_step = self.config.max_joint_step_per_joint
        elif self.config.max_joint_step > 0.0:
            max_step = np.full(6, self.config.max_joint_step, dtype=np.float32)
        else:
            return target.copy()
        delta = target - reference
        return reference + np.clip(delta, -max_step, max_step)

    def check_ik_continuity(self, reference: np.ndarray, target: np.ndarray):
        delta = target - reference
        delta_norm = float(np.linalg.norm(delta))
        delta_max = float(np.max(np.abs(delta)))
        if self.config.max_ik_jump_norm > 0.0 and delta_norm > self.config.max_ik_jump_norm:
            return True, delta_norm, delta_max
        if self.config.max_ik_jump_per_joint is not None and np.any(
            np.abs(delta) > self.config.max_ik_jump_per_joint
        ):
            return True, delta_norm, delta_max
        return False, delta_norm, delta_max

    def update(
        self,
        wrist: np.ndarray,
        current_qpos: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        ee_position, ee_quat_xyzw = self.fk.forward(current_qpos)
        ee_rotation = quat_xyzw_to_matrix(ee_quat_xyzw)
        hand_position, hand_rotation = self.tool_pose_to_hand_pose(ee_position, ee_rotation)

        wrist_position_raw, wrist_quat_xyzw_raw = self.resolve_wrist_tracking_inputs(
            wrist,
            landmarks,
        )
        wrist_position = low_pass_vector(
            self.filtered_wrist_position,
            wrist_position_raw,
            self.config.position_lpf_alpha,
        )
        wrist_quat_xyzw = low_pass_quat_xyzw(
            self.filtered_wrist_quat_xyzw,
            wrist_quat_xyzw_raw,
            self.config.rotation_lpf_alpha,
        )
        self.filtered_wrist_position = wrist_position.copy()
        self.filtered_wrist_quat_xyzw = wrist_quat_xyzw.copy()
        wrist_rotation = quat_xyzw_to_matrix(wrist_quat_xyzw)

        if (
            self.prev_wrist_position is None
            or self.prev_wrist_rotation is None
            or self.wrist_origin_position is None
            or self.wrist_rotation_origin is None
            or self.target_position is None
            or self.target_rotation is None
            or self.command_target_position is None
            or self.command_target_rotation is None
        ):
            self.prev_wrist_position = wrist_position.copy()
            self.prev_wrist_rotation = wrist_rotation.copy()
            self.wrist_origin_position = wrist_position.copy()
            self.wrist_rotation_origin = wrist_rotation.copy()
            self.target_position = ee_position.copy()
            self.target_rotation = ee_rotation.copy()
            self.command_target_position = ee_position.copy()
            self.command_target_rotation = ee_rotation.copy()
            self.ee_origin = ee_position.copy()
            self.ee_rotation_origin = ee_rotation.copy()
            self.hand_origin_position = hand_position.copy()
            self.hand_rotation_origin = hand_rotation.copy()
            self.joint6_origin_position = float(current_qpos[5])
            self.target_joint6_position = float(current_qpos[5])
            self.last_command_positions = current_qpos.copy()
            self.last_command_velocities = np.zeros(6, dtype=np.float32)
            self.trajectory_smoother.reset(current_qpos, self.last_command_velocities)
            print("Calibrated hybrid wrist origin.")
            return current_qpos.copy()

        delta_position_world = (wrist_position - self.prev_wrist_position) * self.config.position_scale
        delta_rotation_world = rotation_matrix_to_rotvec(
            wrist_rotation @ self.prev_wrist_rotation.T
        ) * self.config.rotation_scale
        self.prev_wrist_position = wrist_position.copy()
        self.prev_wrist_rotation = wrist_rotation.copy()

        target_position_before = self.target_position.copy()
        target_rotation_before = self.target_rotation.copy()
        command_target_position_before = self.command_target_position.copy()
        command_target_rotation_before = self.command_target_rotation.copy()
        target_hand_position_before, target_hand_rotation_before = self.tool_pose_to_hand_pose(
            target_position_before,
            target_rotation_before,
        )

        if self.config.position_target_mode == "absolute":
            wrist_offset_world = (wrist_position - self.wrist_origin_position) * self.config.position_scale
            target_hand_position = self.hand_origin_position + wrist_offset_world
        else:
            delta_position_local_current = hand_rotation.T @ delta_position_world
            target_to_current = target_hand_rotation_before.T @ hand_rotation
            delta_position_local_target = target_to_current @ delta_position_local_current
            target_hand_position = (
                target_hand_position_before
                + target_hand_rotation_before @ delta_position_local_target
            )

        if self.config.rotation_control_mode == "full_pose":
            if self.config.orientation_target_mode == "absolute_wrist":
                target_hand_rotation = wrist_rotation.copy()
            elif self.config.orientation_target_mode == "relative":
                relative_rotation_world = wrist_rotation @ self.wrist_rotation_origin.T
                relative_rotation_world = maybe_negate_relative_rot_xy(
                    relative_rotation_world,
                    self.config.negate_rot_xy,
                )
                scaled_relative_rotation_world = scale_rotation_matrix(
                    relative_rotation_world,
                    self.config.rotation_scale,
                )
                target_hand_rotation = scaled_relative_rotation_world @ self.hand_rotation_origin
            elif self.config.orientation_target_mode == "relative_local":
                relative_rotation_local = (
                    self.wrist_rotation_origin.T @ wrist_rotation
                )
                relative_rotation_local = maybe_negate_relative_rot_xy(
                    relative_rotation_local,
                    self.config.negate_rot_xy,
                )
                scaled_relative_rotation_local = scale_rotation_matrix(
                    relative_rotation_local,
                    self.config.rotation_scale,
                )
                target_hand_rotation = self.hand_rotation_origin @ scaled_relative_rotation_local
            else:
                delta_rotation_local_current = hand_rotation.T @ delta_rotation_world
                target_to_current = target_hand_rotation_before.T @ hand_rotation
                delta_rotation_local_target = target_to_current @ delta_rotation_local_current
                target_hand_rotation = target_hand_rotation_before @ rotvec_to_matrix(
                    delta_rotation_local_target
                )
        elif self.config.rotation_control_mode == "hybrid":
            target_hand_rotation = self.hybrid_target_hand_rotation(wrist_rotation)
        else:
                target_hand_rotation = self.hand_rotation_origin.copy()

        debug_enabled = str(
            getattr(self.config, "orientation_debug", False)
        ).lower() in ("1", "true", "yes", "on")
        if debug_enabled:
            now = time.monotonic()
            if now - self.last_orientation_debug_time > 1.0:
                self.last_orientation_debug_time = now
                wrist_rel = rotation_matrix_to_rotvec(
                    wrist_rotation @ self.wrist_rotation_origin.T
                )
                target_rel = rotation_matrix_to_rotvec(
                    target_hand_rotation @ self.hand_rotation_origin.T
                )
                print(
                    "orientation debug:",
                    f"wrist_rel_xyz={np.round(wrist_rel, 3).tolist()}",
                    f"target_rel_xyz={np.round(target_rel, 3).tolist()}",
                    f"rotation_axis_mapping={self.config.rotation_axis_mapping}",
                    f"negate_rot_xy={self.config.negate_rot_xy}",
                    f"mode={self.config.orientation_target_mode}",
                )

        target_hand_position, target_hand_rotation, _ = self.limit_hand_target(
            target_hand_position,
            target_hand_rotation,
        )
        self.target_position, self.target_rotation = self.hand_pose_to_tool_pose(
            target_hand_position,
            target_hand_rotation,
        )
        self.command_target_position = low_pass_vector(
            self.command_target_position,
            self.target_position,
            self.config.pre_ik_position_gain,
        )
        self.command_target_rotation = smooth_rotation_matrix(
            self.command_target_rotation,
            self.target_rotation,
            self.config.pre_ik_rotation_gain,
        )
        target_quat_xyzw = matrix_to_quat_xyzw(self.command_target_rotation)

        seed_q = current_qpos.copy()
        active_mask = np.ones(6, dtype=bool)
        if self.config.rotation_control_mode in ("joint6_only", "hybrid"):
            active_mask[5] = False

        if self.config.rotation_control_mode == "full_pose":
            dls_result = solve_local_dls_6d(
                forward_fn=lambda q: self.fk.forward(q),
                q_seed=seed_q,
                target_position=self.command_target_position,
                target_quat=target_quat_xyzw,
                damping=self.config.dls_damping,
                gain=self.config.dls_gain,
                iterations=self.config.dls_iterations,
                epsilon=self.config.dls_fk_epsilon,
                position_tolerance=self.config.dls_position_tolerance,
                orientation_tolerance=self.config.dls_orientation_tolerance,
                active_mask=active_mask,
                max_delta_per_joint=self.config.dls_max_delta_per_joint,
                current_q=current_qpos,
                nominal_q=self.config.nominal_joint_positions,
                current_q_weight=self.config.current_q_regularization_weight,
                posture_weight=self.config.posture_regularization_weight,
                centering_gain=self.config.centering_gain,
                joint_limit_min=self.config.joint_limit_min,
                joint_limit_max=self.config.joint_limit_max,
                joint_limit_weight=self.config.joint_limit_weight,
                wrist_joint_indices=self.config.wrist_regularization_joints,
                wrist_weight=self.config.wrist_regularization_weight,
                singularity_threshold=self.config.singularity_threshold,
            )
        else:
            dls_result = solve_local_dls_position(
                forward_fn=lambda q: self.fk.forward(q),
                q_seed=seed_q,
                target_position=self.command_target_position,
                damping=self.config.dls_damping,
                gain=self.config.dls_gain,
                iterations=self.config.dls_iterations,
                epsilon=self.config.dls_fk_epsilon,
                tolerance=self.config.dls_position_tolerance,
                active_mask=active_mask,
                max_delta_per_joint=self.config.dls_max_delta_per_joint,
                current_q=current_qpos,
                nominal_q=self.config.nominal_joint_positions,
                current_q_weight=self.config.current_q_regularization_weight,
                posture_weight=self.config.posture_regularization_weight,
                centering_gain=self.config.centering_gain,
                joint_limit_min=self.config.joint_limit_min,
                joint_limit_max=self.config.joint_limit_max,
                joint_limit_weight=self.config.joint_limit_weight,
                wrist_joint_indices=self.config.wrist_regularization_joints,
                wrist_weight=self.config.wrist_regularization_weight,
            )
        ik_q = dls_result.q_next

        continuity_blocked, continuity_norm, continuity_max = self.check_ik_continuity(
            seed_q,
            ik_q,
        )
        if continuity_blocked:
            self.target_position = target_position_before
            self.target_rotation = target_rotation_before
            self.command_target_position = command_target_position_before
            self.command_target_rotation = command_target_rotation_before
            now = time.monotonic()
            if now - self.last_log_time > 1.0:
                self.last_log_time = now
                print(
                    "Hybrid DLS blocked:",
                    f"jump_norm={continuity_norm:.4f}",
                    f"jump_max={continuity_max:.4f}",
                    f"solver={'6d' if self.config.rotation_control_mode == 'full_pose' else 'position'}",
                )
            if self.last_command_positions is not None:
                return self.last_command_positions.copy()
            return current_qpos.copy()

        command_q = self.limit_joint_step(current_qpos, ik_q)
        if self.config.rotation_control_mode in ("joint6_only", "hybrid"):
            command_q[5] = self.compute_joint6_target(wrist_rotation, current_qpos)
        current_dq = (
            np.zeros(6, dtype=np.float32)
            if self.last_command_velocities is None
            else self.last_command_velocities.copy()
        )
        smoothing_result = self.trajectory_smoother.step(
            current_q=current_qpos,
            current_dq=current_dq,
            target_q=command_q,
        )
        command_q = smoothing_result.q_cmd
        self.last_command_velocities = smoothing_result.dq_cmd.copy()
        if self.last_command_positions is not None and self.config.joint_lpf_alpha < 1.0:
            command_q = low_pass_vector(self.last_command_positions, command_q, self.config.joint_lpf_alpha)
        self.last_command_positions = command_q.copy()
        return command_q
