from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np


ForwardFn = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]
ForwardFn6D = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]


@dataclass
class LocalDLSStepResult:
    q_next: np.ndarray
    error_before: float
    error_after: float
    iterations_used: int
    current_distance_after: float
    posture_distance_after: float
    joint_limit_cost_after: float
    wrist_distance_after: float


@dataclass
class LocalDLS6DResult:
    q_next: np.ndarray
    position_error_before: float
    position_error_after: float
    orientation_error_before: float
    orientation_error_after: float
    singularity_distance: float
    iterations_used: int
    current_distance_after: float
    posture_distance_after: float
    joint_limit_cost_after: float
    wrist_distance_after: float


def _as_optional_array(values: Optional[Sequence[float] | np.ndarray], dof: int) -> Optional[np.ndarray]:
    if values is None:
        return None
    array = np.array(values, dtype=np.float32).reshape(-1)
    if array.shape[0] < dof:
        raise ValueError(f"Expected at least {dof} values, got {array.shape[0]}")
    return array[:dof].copy()


def _wrist_index_array(
    wrist_joint_indices: Optional[Sequence[int] | np.ndarray],
    dof: int,
) -> Optional[np.ndarray]:
    if wrist_joint_indices is None:
        return None
    indices = np.array(wrist_joint_indices, dtype=np.int32).reshape(-1)
    if indices.size == 0:
        return None
    indices = indices[(indices >= 0) & (indices < dof)]
    return None if indices.size == 0 else indices


def joint_limit_cost(
    q: np.ndarray,
    joint_limit_min: Optional[Sequence[float] | np.ndarray],
    joint_limit_max: Optional[Sequence[float] | np.ndarray],
) -> float:
    q_min = _as_optional_array(joint_limit_min, q.shape[0])
    q_max = _as_optional_array(joint_limit_max, q.shape[0])
    if q_min is None or q_max is None:
        return 0.0
    q_mid = 0.5 * (q_min + q_max)
    q_range = np.maximum(0.5 * (q_max - q_min), 1e-3)
    normalized = (q - q_mid) / q_range
    return float(np.sum(normalized**4))


def joint_limit_avoidance_direction(
    q: np.ndarray,
    joint_limit_min: Optional[Sequence[float] | np.ndarray],
    joint_limit_max: Optional[Sequence[float] | np.ndarray],
) -> np.ndarray:
    q_min = _as_optional_array(joint_limit_min, q.shape[0])
    q_max = _as_optional_array(joint_limit_max, q.shape[0])
    if q_min is None or q_max is None:
        return np.zeros_like(q, dtype=np.float32)
    q_mid = 0.5 * (q_min + q_max)
    q_range = np.maximum(0.5 * (q_max - q_min), 1e-3)
    normalized = (q - q_mid) / q_range
    # Negative cost gradient nudges joints back toward the center of their range.
    return (-4.0 * (normalized**3) / q_range).astype(np.float32)


def wrist_distance_to_nominal(
    q: np.ndarray,
    nominal_q: Optional[Sequence[float] | np.ndarray],
    wrist_joint_indices: Optional[Sequence[int] | np.ndarray],
) -> float:
    nominal = _as_optional_array(nominal_q, q.shape[0])
    wrist_indices = _wrist_index_array(wrist_joint_indices, q.shape[0])
    if nominal is None or wrist_indices is None:
        return 0.0
    return float(np.linalg.norm(q[wrist_indices] - nominal[wrist_indices]))


def regularization_direction(
    q: np.ndarray,
    current_q: Optional[Sequence[float] | np.ndarray] = None,
    current_q_weight: float = 0.0,
    nominal_q: Optional[Sequence[float] | np.ndarray] = None,
    posture_weight: float = 0.0,
    joint_limit_min: Optional[Sequence[float] | np.ndarray] = None,
    joint_limit_max: Optional[Sequence[float] | np.ndarray] = None,
    joint_limit_weight: float = 0.0,
    wrist_joint_indices: Optional[Sequence[int] | np.ndarray] = None,
    wrist_weight: float = 0.0,
) -> np.ndarray:
    direction = np.zeros_like(q, dtype=np.float32)

    q_now = _as_optional_array(current_q, q.shape[0])
    if q_now is not None and current_q_weight > 0.0:
        direction += float(current_q_weight) * (q_now - q)

    nominal = _as_optional_array(nominal_q, q.shape[0])
    if nominal is not None and posture_weight > 0.0:
        direction += float(posture_weight) * (nominal - q)

    wrist_indices = _wrist_index_array(wrist_joint_indices, q.shape[0])
    if nominal is not None and wrist_indices is not None and wrist_weight > 0.0:
        direction[wrist_indices] += float(wrist_weight) * (nominal[wrist_indices] - q[wrist_indices])

    if joint_limit_weight > 0.0:
        direction += float(joint_limit_weight) * joint_limit_avoidance_direction(
            q,
            joint_limit_min,
            joint_limit_max,
        )

    return direction.astype(np.float32)


def numerical_position_jacobian(
    forward_fn: ForwardFn,
    q: np.ndarray,
    epsilon: float,
    active_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    q = np.array(q, dtype=np.float32)
    pos0, _ = forward_fn(q)
    dof = q.shape[0]
    jacobian = np.zeros((3, dof), dtype=np.float32)

    for index in range(dof):
        if active_mask is not None and not bool(active_mask[index]):
            continue
        q_perturbed = q.copy()
        q_perturbed[index] += epsilon
        pos_eps, _ = forward_fn(q_perturbed)
        jacobian[:, index] = (pos_eps - pos0) / epsilon

    return jacobian


def solve_local_dls_position(
    forward_fn: ForwardFn,
    q_seed: np.ndarray,
    target_position: np.ndarray,
    damping: float = 0.1,
    gain: float = 0.8,
    iterations: int = 2,
    epsilon: float = 1e-3,
    tolerance: float = 2e-3,
    active_mask: Optional[np.ndarray] = None,
    max_delta_per_joint: Optional[np.ndarray] = None,
    current_q: Optional[np.ndarray] = None,
    current_q_weight: float = 0.0,
    nominal_q: Optional[np.ndarray] = None,
    posture_weight: float = 0.0,
    centering_gain: float = 0.0,
    joint_limit_min: Optional[np.ndarray] = None,
    joint_limit_max: Optional[np.ndarray] = None,
    joint_limit_weight: float = 0.0,
    wrist_joint_indices: Optional[Sequence[int] | np.ndarray] = None,
    wrist_weight: float = 0.0,
) -> LocalDLSStepResult:
    q = np.array(q_seed, dtype=np.float32).copy()
    target_position = np.array(target_position, dtype=np.float32)

    pos, _ = forward_fn(q)
    error = target_position - pos
    error_before = float(np.linalg.norm(error))
    iterations_used = 0

    if active_mask is None:
        active_mask = np.ones_like(q, dtype=bool)

    for iteration in range(max(int(iterations), 1)):
        pos, _ = forward_fn(q)
        error = target_position - pos
        error_norm = float(np.linalg.norm(error))
        iterations_used = iteration + 1
        if error_norm <= tolerance:
            break

        jacobian = numerical_position_jacobian(
            forward_fn,
            q,
            epsilon=epsilon,
            active_mask=active_mask,
        )
        damping_matrix = (float(damping) ** 2) * np.eye(3, dtype=np.float32)
        system_matrix = jacobian @ jacobian.T + damping_matrix
        try:
            task_delta = np.linalg.solve(system_matrix, error)
        except np.linalg.LinAlgError:
            task_delta = np.linalg.pinv(system_matrix) @ error

        dq_task = jacobian.T @ task_delta
        dq_task = dq_task * float(gain)

        try:
            system_inverse = np.linalg.solve(
                system_matrix,
                np.eye(jacobian.shape[0], dtype=np.float32),
            )
        except np.linalg.LinAlgError:
            system_inverse = np.linalg.pinv(system_matrix)
        jacobian_pinv = jacobian.T @ system_inverse
        nullspace_projector = np.eye(q.shape[0], dtype=np.float32) - jacobian_pinv @ jacobian

        dq_regularization = regularization_direction(
            q,
            current_q=current_q,
            current_q_weight=current_q_weight,
            nominal_q=nominal_q,
            posture_weight=posture_weight + centering_gain,
            joint_limit_min=joint_limit_min,
            joint_limit_max=joint_limit_max,
            joint_limit_weight=joint_limit_weight,
            wrist_joint_indices=wrist_joint_indices,
            wrist_weight=wrist_weight,
        )

        dq = dq_task + nullspace_projector @ dq_regularization

        dq = dq * active_mask.astype(np.float32)

        if max_delta_per_joint is not None:
            dq = np.clip(dq, -max_delta_per_joint, max_delta_per_joint)

        q = q + dq.astype(np.float32)

    pos_final, _ = forward_fn(q)
    error_after = float(np.linalg.norm(target_position - pos_final))
    return LocalDLSStepResult(
        q_next=q.astype(np.float32),
        error_before=error_before,
        error_after=error_after,
        iterations_used=iterations_used,
        current_distance_after=(
            0.0
            if current_q is None
            else float(np.linalg.norm(q - _as_optional_array(current_q, q.shape[0])))
        ),
        posture_distance_after=(
            0.0
            if nominal_q is None
            else float(np.linalg.norm(q - _as_optional_array(nominal_q, q.shape[0])))
        ),
        joint_limit_cost_after=joint_limit_cost(q, joint_limit_min, joint_limit_max),
        wrist_distance_after=wrist_distance_to_nominal(q, nominal_q, wrist_joint_indices),
    )


def numerical_jacobian_6d(
    forward_fn: ForwardFn6D,
    q: np.ndarray,
    epsilon: float = 1e-4,
    active_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute numerical Jacobian for 6D pose (position + orientation).
    
    Returns a 6xDOF Jacobian matrix where:
    - Rows 0-2: position derivatives
    - Rows 3-5: orientation derivatives (rotation vector)
    
    The forward_fn should return (position, quaternion_xyzw).
    """
    q = np.array(q, dtype=np.float32)
    pos0, quat0 = forward_fn(q)
    dof = min(q.shape[0], 6)
    jacobian = np.zeros((6, dof), dtype=np.float32)
    
    if active_mask is None:
        active_mask = np.ones(dof, dtype=bool)
    
    for index in range(dof):
        if not bool(active_mask[index]):
            continue
        q_perturbed = q.copy()
        q_perturbed[index] += epsilon
        pos_eps, quat_eps = forward_fn(q_perturbed)
        
        jacobian[0:3, index] = (pos_eps - pos0) / epsilon
        
        delta_quat = _quat_multiply(quat_eps, _quat_inverse(quat0))
        delta_rotvec = _quat_to_rotvec(delta_quat)
        jacobian[3:6, index] = delta_rotvec / epsilon
    
    return jacobian


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions [x, y, z, w]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], dtype=np.float32)


def _quat_inverse(q: np.ndarray) -> np.ndarray:
    """Inverse quaternion [x, y, z, w]."""
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)


def _quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to rotation vector."""
    w = q[3]
    xyz = q[0:3]
    norm_xyz = float(np.linalg.norm(xyz))
    
    if norm_xyz < 1e-6:
        return np.zeros(3, dtype=np.float32)
    
    angle = 2.0 * np.arctan2(norm_xyz, abs(w))
    if abs(angle) < 1e-6:
        return np.zeros(3, dtype=np.float32)
    
    axis = xyz / norm_xyz
    return axis * angle


def solve_local_dls_6d(
    forward_fn: ForwardFn6D,
    q_seed: np.ndarray,
    target_position: np.ndarray,
    target_quat: np.ndarray,
    damping: float = 0.05,
    gain: float = 0.5,
    iterations: int = 3,
    epsilon: float = 1e-4,
    position_tolerance: float = 2e-3,
    orientation_tolerance: float = 5e-2,
    active_mask: Optional[np.ndarray] = None,
    max_delta_per_joint: Optional[np.ndarray] = None,
    current_q: Optional[np.ndarray] = None,
    nominal_q: Optional[np.ndarray] = None,
    current_q_weight: float = 0.0,
    posture_weight: float = 0.0,
    centering_gain: float = 0.01,
    joint_limit_min: Optional[np.ndarray] = None,
    joint_limit_max: Optional[np.ndarray] = None,
    joint_limit_weight: float = 0.0,
    wrist_joint_indices: Optional[Sequence[int] | np.ndarray] = None,
    wrist_weight: float = 0.0,
    singularity_threshold: float = 0.1,
) -> LocalDLS6DResult:
    """
    Solve 6D pose IK using Damped Least Squares method.
    
    This method is more robust near singularities compared to exact IK.
    It uses adaptive damping that increases near singularities to prevent
    large joint jumps, and supports joint centering to keep the arm in
    comfortable configurations.
    
    Args:
        forward_fn: Function that takes q and returns (position, quat_xyzw)
        q_seed: Initial joint configuration
        target_position: Target end-effector position
        target_quat: Target end-effector quaternion [x, y, z, w]
        damping: Base damping factor (increased near singularities)
        gain: Step gain (0 < gain <= 1)
        iterations: Maximum iterations per step
        epsilon: Perturbation for numerical Jacobian
        position_tolerance: Position error tolerance
        orientation_tolerance: Orientation error tolerance (rad)
        active_mask: Which joints to use (None = all)
        max_delta_per_joint: Maximum joint change per iteration
        nominal_q: Preferred joint configuration for centering
        centering_gain: Strength of joint centering preference
        singularity_threshold: Manipulability threshold for adaptive damping
    
    Returns:
        LocalDLS6DResult with next joint config and error metrics
    """
    q = np.array(q_seed, dtype=np.float32).copy()
    target_position = np.array(target_position, dtype=np.float32)
    target_quat = np.array(target_quat, dtype=np.float32)
    
    pos, quat = forward_fn(q)
    position_error_before = float(np.linalg.norm(target_position - pos))
    
    delta_quat = _quat_multiply(target_quat, _quat_inverse(quat))
    orientation_error_before = float(np.linalg.norm(_quat_to_rotvec(delta_quat)))
    
    iterations_used = 0
    singularity_distance = 1.0
    
    if active_mask is None:
        active_mask = np.ones(min(q.shape[0], 6), dtype=bool)
    
    for iteration in range(max(int(iterations), 1)):
        pos, quat = forward_fn(q)
        
        pos_error = target_position - pos
        pos_error_norm = float(np.linalg.norm(pos_error))
        
        delta_quat = _quat_multiply(target_quat, _quat_inverse(quat))
        rotvec_error = _quat_to_rotvec(delta_quat)
        rot_error_norm = float(np.linalg.norm(rotvec_error))
        
        iterations_used = iteration + 1
        
        if pos_error_norm <= position_tolerance and rot_error_norm <= orientation_tolerance:
            break
        
        jacobian = numerical_jacobian_6d(
            forward_fn,
            q,
            epsilon=epsilon,
            active_mask=active_mask,
        )
        
        manipulability = float(np.sqrt(max(np.linalg.det(jacobian @ jacobian.T), 0.0)))
        singularity_distance = max(manipulability, 0.001)
        
        adaptive_damping = float(damping)
        if singularity_distance < singularity_threshold:
            adaptive_damping = float(damping) * (singularity_threshold / singularity_distance) ** 2
        
        task_error = np.concatenate([pos_error, rotvec_error])
        damping_matrix = (adaptive_damping ** 2) * np.eye(6, dtype=np.float32)
        
        system_matrix = jacobian @ jacobian.T + damping_matrix
        
        try:
            task_delta = np.linalg.solve(system_matrix, task_error)
        except np.linalg.LinAlgError:
            task_delta = np.linalg.lstsq(system_matrix, task_error, rcond=None)[0]
        
        dq_task = jacobian.T @ task_delta
        dq_task = dq_task * float(gain)

        if singularity_distance < singularity_threshold:
            slowdown_factor = singularity_distance / singularity_threshold
            dq_task = dq_task * float(slowdown_factor)

        try:
            system_inverse = np.linalg.solve(
                system_matrix,
                np.eye(jacobian.shape[0], dtype=np.float32),
            )
        except np.linalg.LinAlgError:
            system_inverse = np.linalg.pinv(system_matrix)

        jacobian_pinv = jacobian.T @ system_inverse
        nullspace_projector = np.eye(q.shape[0], dtype=np.float32) - jacobian_pinv @ jacobian

        dq_regularization = regularization_direction(
            q,
            current_q=current_q,
            current_q_weight=current_q_weight,
            nominal_q=nominal_q,
            posture_weight=posture_weight + centering_gain,
            joint_limit_min=joint_limit_min,
            joint_limit_max=joint_limit_max,
            joint_limit_weight=joint_limit_weight,
            wrist_joint_indices=wrist_joint_indices,
            wrist_weight=wrist_weight,
        )

        dq = dq_task + nullspace_projector @ dq_regularization

        dq = dq * active_mask[:dq.shape[0]].astype(np.float32)
        
        if max_delta_per_joint is not None:
            dq = np.clip(dq, -max_delta_per_joint[:dq.shape[0]], max_delta_per_joint[:dq.shape[0]])
        
        q[:dq.shape[0]] = q[:dq.shape[0]] + dq.astype(np.float32)
    
    pos_final, quat_final = forward_fn(q)
    position_error_after = float(np.linalg.norm(target_position - pos_final))
    
    delta_quat_final = _quat_multiply(target_quat, _quat_inverse(quat_final))
    orientation_error_after = float(np.linalg.norm(_quat_to_rotvec(delta_quat_final)))
    
    return LocalDLS6DResult(
        q_next=q.astype(np.float32),
        position_error_before=position_error_before,
        position_error_after=position_error_after,
        orientation_error_before=orientation_error_before,
        orientation_error_after=orientation_error_after,
        iterations_used=iterations_used,
        singularity_distance=singularity_distance,
        current_distance_after=(
            0.0
            if current_q is None
            else float(np.linalg.norm(q - _as_optional_array(current_q, q.shape[0])))
        ),
        posture_distance_after=(
            0.0
            if nominal_q is None
            else float(np.linalg.norm(q - _as_optional_array(nominal_q, q.shape[0])))
        ),
        joint_limit_cost_after=joint_limit_cost(q, joint_limit_min, joint_limit_max),
        wrist_distance_after=wrist_distance_to_nominal(q, nominal_q, wrist_joint_indices),
    )
