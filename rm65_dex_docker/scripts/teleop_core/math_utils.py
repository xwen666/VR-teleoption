from __future__ import annotations

from typing import Optional

import numpy as np


def normalized_quat_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = np.array(quat_xyzw, dtype=np.float32)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return quat / norm


def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = normalized_quat_xyzw(quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def matrix_to_quat_xyzw(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            w = (rotation[2, 1] - rotation[1, 2]) / s
            x = 0.25 * s
            y = (rotation[0, 1] + rotation[1, 0]) / s
            z = (rotation[0, 2] + rotation[2, 0]) / s
        elif index == 1:
            s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            w = (rotation[0, 2] - rotation[2, 0]) / s
            x = (rotation[0, 1] + rotation[1, 0]) / s
            y = 0.25 * s
            z = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            w = (rotation[1, 0] - rotation[0, 1]) / s
            x = (rotation[0, 2] + rotation[2, 0]) / s
            y = (rotation[1, 2] + rotation[2, 1]) / s
            z = 0.25 * s
    return normalized_quat_xyzw(np.array([x, y, z, w], dtype=np.float32))


def rotation_matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    cos_angle = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cos_angle))
    if angle < 1e-6:
        return np.zeros(3, dtype=np.float32)
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float32,
    )
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-9:
        return np.zeros(3, dtype=np.float32)
    axis /= axis_norm
    return axis * angle


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-9:
        return np.eye(3, dtype=np.float32)
    axis = rotvec / angle
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float32,
    )


def scale_rotation_matrix(rotation: np.ndarray, scale: float) -> np.ndarray:
    return rotvec_to_matrix(rotation_matrix_to_rotvec(rotation) * scale)


def low_pass_vector(previous: Optional[np.ndarray], current: np.ndarray, alpha: float) -> np.ndarray:
    if previous is None or alpha >= 1.0:
        return current.copy()
    if alpha <= 0.0:
        return previous.copy()
    return previous + alpha * (current - previous)


def low_pass_quat_xyzw(previous: Optional[np.ndarray], current: np.ndarray, alpha: float) -> np.ndarray:
    current_norm = normalized_quat_xyzw(current)
    if previous is None or alpha >= 1.0:
        return current_norm
    if alpha <= 0.0:
        return normalized_quat_xyzw(previous)
    prev_norm = normalized_quat_xyzw(previous)
    if float(np.dot(prev_norm, current_norm)) < 0.0:
        current_norm = -current_norm
    blended = prev_norm + alpha * (current_norm - prev_norm)
    return normalized_quat_xyzw(blended)


def smooth_rotation_matrix(
    previous: Optional[np.ndarray],
    target: np.ndarray,
    gain: float,
) -> np.ndarray:
    if previous is None or gain >= 1.0:
        return target.copy()
    if gain <= 0.0:
        return previous.copy()
    relative_rotation = target @ previous.T
    return rotvec_to_matrix(rotation_matrix_to_rotvec(relative_rotation) * gain) @ previous


def limit_vector(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= max_norm or norm < 1e-9:
        return vector.copy()
    return vector * (max_norm / norm)

