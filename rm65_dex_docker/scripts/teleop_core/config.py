from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from .constants import (
    DEFAULT_WRIST_REGULARIZATION_JOINTS,
    RM65_JOINT_LIMIT_MAX,
    RM65_JOINT_LIMIT_MIN,
)
from .math_utils import normalized_quat_xyzw


@dataclass
class HybridConfig:
    scene_variant: str
    axis_mapping: str
    rotation_axis_mapping: str
    position_scale: float
    rotation_scale: float
    position_target_mode: str
    position_lpf_alpha: float
    rotation_lpf_alpha: float
    pre_ik_position_gain: float
    pre_ik_rotation_gain: float
    joint_lpf_alpha: float
    packet_timeout: float
    max_position_offset: float
    max_rotation_error: float
    max_joint_step: float
    max_joint_step_per_joint: Optional[np.ndarray]
    realman_lib_path: str
    robot_model: str
    use_orientation: bool
    rotation_control_mode: str
    orientation_input_source: str
    hybrid_pose_rotation_scale: float
    orientation_deadband_rad: float
    orientation_target_mode: str
    negate_rot_xy: bool
    orientation_debug: bool
    orientation_fallback_to_position_only: bool
    block_singularity: bool
    max_ik_jump_norm: float
    max_ik_jump_per_joint: Optional[np.ndarray]
    dls_damping: float
    dls_gain: float
    dls_iterations: int
    dls_fk_epsilon: float
    dls_position_tolerance: float
    dls_orientation_tolerance: float
    dls_max_delta_per_joint: Optional[np.ndarray]
    centering_gain: float
    current_q_regularization_weight: float
    posture_regularization_weight: float
    joint_limit_weight: float
    wrist_regularization_weight: float
    wrist_regularization_joints: np.ndarray
    joint_limit_min: np.ndarray
    joint_limit_max: np.ndarray
    singularity_threshold: float
    singularity_decelerate: bool
    nominal_joint_positions: np.ndarray
    initial_joint_positions: np.ndarray
    joint6_axis: str
    joint6_scale: float
    joint6_deadband_rad: float
    joint6_min: float
    joint6_max: float
    max_joint6_step: float
    trajectory_smoother: str
    trajectory_control_dt: float
    ruckig_max_velocity: np.ndarray
    ruckig_max_acceleration: np.ndarray
    ruckig_max_jerk: np.ndarray
    wrist_to_hand_quat_xyzw: np.ndarray
    tool_to_hand_translation: np.ndarray
    tool_to_hand_quat_xyzw: np.ndarray


def _optional_positive_vector(params: dict, key: str) -> Optional[np.ndarray]:
    values = np.array(params.get(key, [0.0] * 6), dtype=np.float32)
    return values if np.any(values > 0.0) else None


def load_hybrid_config(config_path: Path) -> HybridConfig:
    config_path = Path(config_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    params = data["wrist_ik_bridge"]["ros__parameters"]
    max_joint_step_per_joint = _optional_positive_vector(params, "max_joint_step_per_joint")
    max_ik_jump_per_joint = _optional_positive_vector(params, "max_ik_jump_per_joint")
    dls_max_delta_per_joint = _optional_positive_vector(params, "dls_max_delta_per_joint")
    if dls_max_delta_per_joint is None:
        dls_max_delta_per_joint = max_joint_step_per_joint
    return HybridConfig(
        scene_variant=str(params.get("scene_variant", "table_bin")),
        axis_mapping=str(params.get("axis_mapping", "quest3_teleop_flip_forward")),
        rotation_axis_mapping=str(
            params.get(
                "rotation_axis_mapping",
                params.get("axis_mapping", "quest3_teleop_flip_forward"),
            )
        ),
        position_scale=float(params.get("position_scale", 1.0)),
        rotation_scale=float(params.get("rotation_scale", 1.0)),
        position_target_mode=str(params.get("position_target_mode", "absolute")),
        position_lpf_alpha=float(params.get("position_lpf_alpha", 1.0)),
        rotation_lpf_alpha=float(params.get("rotation_lpf_alpha", 1.0)),
        pre_ik_position_gain=float(params.get("pre_ik_position_gain", 1.0)),
        pre_ik_rotation_gain=float(params.get("pre_ik_rotation_gain", 1.0)),
        joint_lpf_alpha=float(params.get("joint_lpf_alpha", 1.0)),
        packet_timeout=float(params.get("packet_timeout", 0.25)),
        max_position_offset=float(params.get("max_position_offset", 0.45)),
        max_rotation_error=float(params.get("max_rotation_error", 0.8)),
        max_joint_step=float(params.get("max_joint_step", 0.0)),
        max_joint_step_per_joint=max_joint_step_per_joint,
        realman_lib_path=str(params.get("realman_lib_path", "")),
        robot_model=str(params.get("robot_model", "rm65")),
        use_orientation=bool(params.get("use_orientation", True)),
        rotation_control_mode=str(params.get("rotation_control_mode", "joint6_only")),
        orientation_input_source=str(params.get("orientation_input_source", "auto")),
        hybrid_pose_rotation_scale=float(params.get("hybrid_pose_rotation_scale", 0.25)),
        orientation_deadband_rad=float(params.get("orientation_deadband_rad", 0.0)),
        orientation_target_mode=str(params.get("orientation_target_mode", "relative")),
        negate_rot_xy=bool(params.get("negate_rot_xy", False)),
        orientation_debug=bool(params.get("orientation_debug", False)),
        orientation_fallback_to_position_only=bool(
            params.get("orientation_fallback_to_position_only", True)
        ),
        block_singularity=bool(params.get("block_singularity", False)),
        max_ik_jump_norm=float(params.get("max_ik_jump_norm", 0.0)),
        max_ik_jump_per_joint=max_ik_jump_per_joint,
        initial_joint_positions=np.array(
            params.get("initial_joint_positions", [0.0, -0.7, 1.2, 0.0, 0.8, 0.0]),
            dtype=np.float32,
        ),
        dls_damping=float(params.get("dls_damping", 0.05)),
        dls_gain=float(params.get("dls_position_gain", params.get("dls_gain", 0.5))),
        dls_iterations=int(params.get("dls_iterations", 3)),
        dls_fk_epsilon=float(params.get("dls_fk_epsilon", 1e-4)),
        dls_position_tolerance=float(params.get("dls_position_tolerance", 2e-3)),
        dls_orientation_tolerance=float(params.get("dls_orientation_tolerance", 5e-2)),
        dls_max_delta_per_joint=dls_max_delta_per_joint,
        centering_gain=float(params.get("dls_centering_gain", params.get("centering_gain", 0.01))),
        current_q_regularization_weight=float(params.get("current_q_regularization_weight", 0.0)),
        posture_regularization_weight=float(params.get("posture_regularization_weight", 0.0)),
        joint_limit_weight=float(params.get("joint_limit_weight", 0.0)),
        wrist_regularization_weight=float(params.get("wrist_regularization_weight", 0.0)),
        wrist_regularization_joints=np.array(
            params.get("wrist_regularization_joints", DEFAULT_WRIST_REGULARIZATION_JOINTS),
            dtype=np.int32,
        ),
        joint_limit_min=np.array(
            params.get("joint_limit_min", RM65_JOINT_LIMIT_MIN),
            dtype=np.float32,
        ),
        joint_limit_max=np.array(
            params.get("joint_limit_max", RM65_JOINT_LIMIT_MAX),
            dtype=np.float32,
        ),
        singularity_threshold=float(
            params.get("dls_singularity_threshold", params.get("singularity_threshold", 0.1))
        ),
        singularity_decelerate=bool(params.get("singularity_decelerate", True)),
        nominal_joint_positions=np.array(
            params.get("nominal_joint_positions", [0.0, -0.7, 1.2, 0.0, 0.8, 0.0]),
            dtype=np.float32,
        ),
        joint6_axis=str(params.get("joint6_axis", "z")),
        joint6_scale=float(params.get("joint6_scale", 1.0)),
        joint6_deadband_rad=float(params.get("joint6_deadband_rad", 0.0)),
        joint6_min=float(params.get("joint6_min", -6.28)),
        joint6_max=float(params.get("joint6_max", 6.28)),
        max_joint6_step=float(params.get("max_joint6_step", 0.06)),
        trajectory_smoother=str(params.get("trajectory_smoother", "auto")),
        trajectory_control_dt=float(params.get("trajectory_control_dt", 0.01)),
        ruckig_max_velocity=np.array(
            params.get("ruckig_max_velocity", [0.20, 0.20, 0.20, 0.40, 0.40, 0.40]),
            dtype=np.float32,
        ),
        ruckig_max_acceleration=np.array(
            params.get("ruckig_max_acceleration", [0.40, 0.40, 0.40, 0.80, 0.80, 0.80]),
            dtype=np.float32,
        ),
        ruckig_max_jerk=np.array(
            params.get("ruckig_max_jerk", [1.50, 1.50, 1.50, 3.00, 3.00, 3.00]),
            dtype=np.float32,
        ),
        wrist_to_hand_quat_xyzw=normalized_quat_xyzw(
            np.array(
                params.get("wrist_to_hand_quat_xyzw", [0.0, 0.0, 0.0, 1.0]),
                dtype=np.float32,
            )
        ),
        tool_to_hand_translation=np.array(
            params.get("tool_to_hand_translation", [0.0, 0.0, 0.0]),
            dtype=np.float32,
        ),
        tool_to_hand_quat_xyzw=normalized_quat_xyzw(
            np.array(
                params.get("tool_to_hand_quat_xyzw", [0.0, 0.0, 0.0, 1.0]),
                dtype=np.float32,
            )
        ),
    )


def load_arm_qpos_from_snapshot(snapshot_path: Optional[Path]) -> Optional[np.ndarray]:
    if snapshot_path is None:
        return None
    snapshot_path = Path(snapshot_path)
    if not snapshot_path.exists():
        return None
    try:
        data = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    arm_qpos = data.get("arm_qpos")
    if not isinstance(arm_qpos, list) or len(arm_qpos) < 6:
        return None
    try:
        return np.array(arm_qpos[:6], dtype=np.float32)
    except (TypeError, ValueError):
        return None


def override_config_joint_seed_from_snapshot(
    config: HybridConfig,
    snapshot_path: Optional[Path],
    *,
    override_initial: bool = True,
    override_nominal: bool = True,
    log_prefix: str = "",
) -> HybridConfig:
    arm_qpos = load_arm_qpos_from_snapshot(snapshot_path)
    if arm_qpos is None:
        return config
    updates = {}
    if override_initial:
        updates["initial_joint_positions"] = arm_qpos.copy()
    if override_nominal:
        updates["nominal_joint_positions"] = arm_qpos.copy()
    if not updates:
        return config
    prefix = f"{log_prefix} " if log_prefix else ""
    print(
        f"{prefix}Using startup snapshot as joint seed:"
        f" path={snapshot_path}"
        f" arm_qpos={np.round(arm_qpos, 4).tolist()}"
    )
    return replace(config, **updates)
