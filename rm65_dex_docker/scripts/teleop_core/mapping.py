from __future__ import annotations

import numpy as np

from .math_utils import (
    matrix_to_quat_xyzw,
    normalized_quat_xyzw,
    quat_xyzw_to_matrix,
)


def make_quest_to_robot_matrix(axis_mapping: str) -> np.ndarray:
    if axis_mapping == "cube_forward":
        return np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    if axis_mapping == "quest3_teleop":
        return np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    if axis_mapping == "quest3_teleop_flip_forward":
        return np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    if axis_mapping == "quest3_teleop_flip_forward_up_y_left_z":
        return np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
    if axis_mapping == "quest3_front_x_left_y_up_z":
        return np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    if axis_mapping == "base_y_left":
        return np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    if axis_mapping == "base_y_left_flip_forward":
        return np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    if axis_mapping == "base_z_right":
        return np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    if axis_mapping == "base_z_left":
        return np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    if axis_mapping == "wrist_left_to_base_x_right_y_forward_z_up":
        # Wrist visual axes: +X red right, +Y green up, +Z blue forward.
        # Robot base axes: +X red right, +Y green forward, +Z blue up.
        # This maps wrist +X -> base +X, wrist +Y -> base +Z,
        # wrist +Z -> base +Y. The matrix is an improper orthogonal
        # coordinate transform because the wrist frame is left-handed here.
        return np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    if axis_mapping == "wrist_left_rotation_xz_unswap":
        # Rotation-only correction for the left-wrist frame above. It is the
        # position mapping composed with an X/Z pre-swap, giving a proper
        # rotation matrix so wrist X/Z rotation axes are not interchanged.
        return np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    if axis_mapping == "wrist_left_rotation_xz_unswap_flip_blue":
        # Same axis pairing as wrist_left_rotation_xz_unswap, but flips the
        # wrist blue-axis rotation direction. Green flips with it so the
        # matrix remains a proper rotation (det=+1) for quaternion mapping.
        return np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float32)
    if axis_mapping == "identity":
        return np.eye(3, dtype=np.float32)
    raise ValueError(f"Unsupported axis_mapping: {axis_mapping}")


def is_vr_teleop_rotation_mapping(axis_mapping: str) -> bool:
    return axis_mapping in (
        "vr_teleop_sim",
        "vr_teleop_realman",
        "vr_teleop_quest3_raw",
        "vr_teleop_left",
        "vr_teleop_left_sim",
        "vr_teleop_left_realman",
    )


def mirror_left_wrist_quat_to_right_like(quat_xyzw: np.ndarray) -> np.ndarray:
    # Quest left-wrist +X points anatomically left; mirror X so it follows
    # the right-wrist convention expected by vr_teleop's RM65 examples.
    qx, qy, qz, qw = normalized_quat_xyzw(quat_xyzw)
    return np.array([qx, -qy, -qz, qw], dtype=np.float32)


def vr_teleop_sim_rotation_to_robot(quat_xyzw: np.ndarray) -> np.ndarray:
    wrist_rotation = quat_xyzw_to_matrix(quat_xyzw)
    rotation_transform = np.array(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=np.float32,
    )
    return rotation_transform @ wrist_rotation @ rotation_transform.T


def vr_teleop_realman_rotation_to_robot(quat_xyzw: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = normalized_quat_xyzw(quat_xyzw)
    wrist_quat = np.array([qz, qy, -qx, qw], dtype=np.float32)
    wrist_rotation = quat_xyzw_to_matrix(wrist_quat)
    rotation_transform = np.array(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=np.float32,
    )
    return rotation_transform @ wrist_rotation @ rotation_transform.T


def vr_teleop_left_sim_rotation_to_robot(quat_xyzw: np.ndarray) -> np.ndarray:
    return vr_teleop_sim_rotation_to_robot(
        mirror_left_wrist_quat_to_right_like(quat_xyzw)
    )


def vr_teleop_left_realman_rotation_to_robot(quat_xyzw: np.ndarray) -> np.ndarray:
    return vr_teleop_realman_rotation_to_robot(
        mirror_left_wrist_quat_to_right_like(quat_xyzw)
    )


def quest_rotation_to_robot(
    quat_xyzw: np.ndarray,
    quest_to_robot: np.ndarray,
    rotation_axis_mapping: str = "",
) -> np.ndarray:
    if rotation_axis_mapping in ("vr_teleop_left", "vr_teleop_left_sim"):
        return vr_teleop_left_sim_rotation_to_robot(quat_xyzw)
    if rotation_axis_mapping == "vr_teleop_left_realman":
        return vr_teleop_left_realman_rotation_to_robot(quat_xyzw)
    if rotation_axis_mapping == "vr_teleop_sim":
        return vr_teleop_sim_rotation_to_robot(quat_xyzw)
    if is_vr_teleop_rotation_mapping(rotation_axis_mapping):
        return vr_teleop_realman_rotation_to_robot(quat_xyzw)
    quest_rotation = quat_xyzw_to_matrix(quat_xyzw)
    return quest_to_robot @ quest_rotation @ quest_to_robot.T


def maybe_negate_relative_rot_xy(rotation: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return rotation
    quat = matrix_to_quat_xyzw(rotation)
    quat[0] = -quat[0]
    quat[1] = -quat[1]
    return quat_xyzw_to_matrix(quat)

