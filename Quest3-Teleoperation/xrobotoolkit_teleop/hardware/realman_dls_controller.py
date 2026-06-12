"""
RealMan DLS IK Controller - Local Damped Least Squares IK for smooth teleoperation.

This controller uses Jacobian-based servo with damping instead of exact IK solving,
which avoids getting stuck near singularities and provides smoother motion.

Key features:
- Local DLS IK (damped least squares) instead of exact IK
- Joint centering / minimal displacement preference
- Singularity-aware deceleration (not hard stop)
- 6D pose control (position + orientation)
"""

import os
import time
import threading
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.hardware.interface.realman_robots import (
    CONTROLLER_DEADZONE,
    GRIPPER_FORCE,
    GRIPPER_SPEED,
    RIGHT_INITIAL_JOINT_DEG,
    MAX_VELOCITY,
    MAX_ACCELERATION,
    SERVO_TIME,
    LOOKAHEAD_TIME,
    SERVO_GAIN,
    RealManController,
)
from xrobotoolkit_teleop.utils.geometry import (
    R_HEADSET_TO_WORLD,
    quat_diff_as_angle_axis,
)
from xrobotoolkit_teleop.utils.parallel_gripper_utils import calc_parallel_gripper_position
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH


@dataclass
class DLSIKResult:
    q_next: np.ndarray
    position_error: float
    orientation_error: float
    singularity_distance: float
    iterations_used: int


def numerical_jacobian_6d(
    forward_fn,
    q: np.ndarray,
    epsilon: float = 1e-4,
) -> np.ndarray:
    """
    Compute numerical Jacobian for 6D pose (position + orientation).
    
    Returns a 6x6 Jacobian matrix where:
    - Rows 0-2: position derivatives
    - Rows 3-5: orientation derivatives (rotation vector)
    """
    q = np.array(q, dtype=np.float32)
    pos0, quat0 = forward_fn(q)
    
    dof = min(q.shape[0], 6)
    jacobian = np.zeros((6, 6), dtype=np.float32)
    
    for index in range(dof):
        q_perturbed = q.copy()
        q_perturbed[index] += epsilon
        pos_eps, quat_eps = forward_fn(q_perturbed)
        
        jacobian[0:3, index] = (pos_eps - pos0) / epsilon
        
        delta_quat = quat_multiply(quat_eps, quat_inverse(quat0))
        delta_rotvec = quat_to_rotvec(delta_quat)
        jacobian[3:6, index] = delta_rotvec / epsilon
    
    return jacobian


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions [x, y, z, w]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], dtype=np.float32)


def quat_inverse(q: np.ndarray) -> np.ndarray:
    """Inverse quaternion [x, y, z, w]."""
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to rotation vector."""
    w = q[3]
    xyz = q[0:3]
    norm_xyz = np.linalg.norm(xyz)
    
    if norm_xyz < 1e-6:
        return np.zeros(3, dtype=np.float32)
    
    angle = 2.0 * np.arctan2(norm_xyz, abs(w))
    axis = xyz / norm_xyz
    return axis * angle


def rotvec_to_quat(rotvec: np.ndarray) -> np.ndarray:
    """Convert rotation vector to quaternion [x, y, z, w]."""
    angle = np.linalg.norm(rotvec)
    if angle < 1e-6:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    
    axis = rotvec / angle
    half_angle = angle / 2.0
    return np.array([
        axis[0] * np.sin(half_angle),
        axis[1] * np.sin(half_angle),
        axis[2] * np.sin(half_angle),
        np.cos(half_angle),
    ], dtype=np.float32)


def solve_dls_6d(
    forward_fn,
    q_seed: np.ndarray,
    target_position: np.ndarray,
    target_quat: np.ndarray,
    damping: float = 0.05,
    gain: float = 0.5,
    iterations: int = 3,
    epsilon: float = 1e-4,
    position_tolerance: float = 1e-3,
    orientation_tolerance: float = 1e-2,
    max_delta_per_joint: Optional[np.ndarray] = None,
    nominal_q: Optional[np.ndarray] = None,
    centering_gain: float = 0.01,
    singularity_threshold: float = 0.1,
) -> DLSIKResult:
    """
    Solve 6D pose IK using Damped Least Squares method.
    
    This method is more robust near singularities compared to exact IK.
    """
    q = np.array(q_seed, dtype=np.float32).copy()
    target_position = np.array(target_position, dtype=np.float32)
    target_quat = np.array(target_quat, dtype=np.float32)
    
    pos, quat = forward_fn(q)
    position_error = float(np.linalg.norm(target_position - pos))
    
    delta_quat = quat_multiply(target_quat, quat_inverse(quat))
    orientation_error = float(np.linalg.norm(quat_to_rotvec(delta_quat)))
    
    iterations_used = 0
    singularity_distance = 1.0
    
    for iteration in range(max(int(iterations), 1)):
        pos, quat = forward_fn(q)
        
        pos_error = target_position - pos
        pos_error_norm = float(np.linalg.norm(pos_error))
        
        delta_quat = quat_multiply(target_quat, quat_inverse(quat))
        rotvec_error = quat_to_rotvec(delta_quat)
        rot_error_norm = float(np.linalg.norm(rotvec_error))
        
        iterations_used = iteration + 1
        
        if pos_error_norm <= position_tolerance and rot_error_norm <= orientation_tolerance:
            break
        
        jacobian = numerical_jacobian_6d(forward_fn, q, epsilon)
        
        manipulability = float(np.sqrt(np.linalg.det(jacobian @ jacobian.T)))
        singularity_distance = max(manipulability, 0.001)
        
        adaptive_damping = damping
        if singularity_distance < singularity_threshold:
            adaptive_damping = damping * (singularity_threshold / singularity_distance) ** 2
        
        task_error = np.concatenate([pos_error, rotvec_error])
        damping_matrix = (adaptive_damping ** 2) * np.eye(6, dtype=np.float32)
        
        system_matrix = jacobian @ jacobian.T + damping_matrix
        
        try:
            task_delta = np.linalg.solve(system_matrix, task_error)
        except np.linalg.LinAlgError:
            task_delta = np.linalg.pinv(system_matrix) @ task_error
        
        dq = jacobian.T @ task_delta
        dq = dq * float(gain)
        
        if singularity_distance < singularity_threshold:
            slowdown_factor = singularity_distance / singularity_threshold
            dq = dq * slowdown_factor
        
        if nominal_q is not None and centering_gain > 0.0:
            centering_term = centering_gain * (np.array(nominal_q, dtype=np.float32) - q)
            dq = dq + centering_term
        
        if max_delta_per_joint is not None:
            dq = np.clip(dq[:6], -max_delta_per_joint, max_delta_per_joint)
        
        q[:6] = q[:6] + dq[:6].astype(np.float32)
    
    pos_final, quat_final = forward_fn(q)
    position_error = float(np.linalg.norm(target_position - pos_final))
    
    delta_quat_final = quat_multiply(target_quat, quat_inverse(quat_final))
    orientation_error = float(np.linalg.norm(quat_to_rotvec(delta_quat_final)))
    
    return DLSIKResult(
        q_next=q.astype(np.float32),
        position_error=position_error,
        orientation_error=orientation_error,
        singularity_distance=singularity_distance,
        iterations_used=iterations_used,
    )


class ArmRealManDLSController:
    """
    RealMan controller using Local DLS IK for smooth teleoperation.
    
    Key differences from exact IK controller:
    - Uses damped least squares instead of exact IK solving
    - Never gets stuck near singularities (decelerates instead)
    - Joint centering preference keeps arm in comfortable configuration
    - Smoother motion, less jerky
    """
    
    def __init__(
        self,
        xr_client: XrClient,
        right_initial_joint_deg: np.ndarray = RIGHT_INITIAL_JOINT_DEG,
        max_velocity: float = MAX_VELOCITY,
        max_acceleration: float = MAX_ACCELERATION,
        servo_time: float = SERVO_TIME,
        lookahead_time: float = LOOKAHEAD_TIME,
        servo_gain: float = SERVO_GAIN,
        gripper_force: float = GRIPPER_FORCE,
        gripper_speed: float = GRIPPER_SPEED,
        R_headset_world: np.ndarray = R_HEADSET_TO_WORLD,
        scale_factor: float = 1.2,
        dls_damping: float = 0.05,
        dls_gain: float = 0.5,
        centering_gain: float = 0.01,
        singularity_threshold: float = 0.1,
        max_joint_delta_deg: float = 2.0,
        visualize: bool = False,
    ):
        self.xr_client = xr_client
        self.R_headset_world = R_headset_world
        self.scale_factor = scale_factor
        self.visualize = visualize
        
        self.dls_damping = dls_damping
        self.dls_gain = dls_gain
        self.centering_gain = centering_gain
        self.singularity_threshold = singularity_threshold
        self.max_joint_delta_deg = max_joint_delta_deg
        
        self.right_controller = RealManController(
            initial_joint_positions=right_initial_joint_deg,
            max_velocity=max_velocity,
            max_acceleration=max_acceleration,
            servo_time=servo_time,
            lookahead_time=lookahead_time,
            servo_gain=servo_gain,
            gripper_force=gripper_force,
            gripper_speed=gripper_speed,
            wifi=False,
        )
        
        self.gripper_active = False
        
        self.init_controller_xyz = None
        self.init_controller_quat = None
        
        self.ee_target_position = None
        self.ee_target_quat = None
        
        right_q_init = self.right_controller.get_current_joint_positions()
        self.right_gripper_pos = self.right_controller.get_gripper_open_position()
        
        self.target_right_q = right_q_init.copy()
        self.last_sent_q = right_q_init.copy()
        
        self.nominal_q = right_q_init.copy()
        
        self.max_delta_per_joint = np.full(6, max_joint_delta_deg * np.pi / 180.0, dtype=np.float32)
        
        self.manipulator_config = {
            "right_arm": {
                "link_name": "Link_6",
                "pose_source": "right_controller",
                "control_trigger": "right_grip",
                "gripper_trigger": "right_trigger",
            },
        }
        
        self.mode = "idle"
        self.right_home_q_deg = np.asarray(right_initial_joint_deg, dtype=float).reshape(-1)
        self.home_tol_deg = 1.0
        self._homing_started = False
        
        print(f"[DLS Controller] Initialized with damping={dls_damping}, gain={dls_gain}")
        print(f"[DLS Controller] Centering gain={centering_gain}, singularity threshold={singularity_threshold}")
    
    def forward_kinematics(self, q_rad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward kinematics using RealMan SDK.
        
        Returns (position, quaternion [x, y, z, w]).
        """
        q_deg = q_rad * 180.0 / np.pi
        pose = self.right_controller.get_current_tcp_pose()
        
        position = np.array(pose[:3], dtype=np.float32)
        
        rx, ry, rz = pose[3:6]
        from scipy.spatial.transform import Rotation as R
        rot = R.from_euler('xyz', [rx, ry, rz], degrees=True)
        quat_xyzw = rot.as_quat()
        
        return position, quat_xyzw.astype(np.float32)
    
    def forward_fn_wrapper(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Wrapper for DLS IK forward function."""
        return self.forward_kinematics(q)
    
    def _process_xr_pose(self, xr_pose, R_world_ee: np.ndarray):
        """Process XR controller pose to get local frame delta."""
        controller_xyz = np.array([xr_pose[0], xr_pose[1], xr_pose[2]])
        controller_quat = np.array([xr_pose[6], xr_pose[3], xr_pose[4], xr_pose[5]])
        
        controller_xyz = self.R_headset_world @ controller_xyz
        
        import meshcat.transformations as tf
        R_transform = np.eye(4)
        R_transform[:3, :3] = self.R_headset_world
        R_quat = tf.quaternion_from_matrix(R_transform)
        controller_quat = tf.quaternion_multiply(
            tf.quaternion_multiply(R_quat, controller_quat),
            tf.quaternion_conjugate(R_quat),
        )
        
        if self.init_controller_xyz is None:
            self.init_controller_xyz = controller_xyz.copy()
            self.init_controller_quat = controller_quat.copy()
            return np.zeros(3), np.zeros(3)
        
        delta_xyz_world = (controller_xyz - self.init_controller_xyz) * self.scale_factor
        delta_rot_world = quat_diff_as_angle_axis(self.init_controller_quat, controller_quat)
        
        self.init_controller_xyz = controller_xyz.copy()
        self.init_controller_quat = controller_quat.copy()
        
        R_ee_world = R_world_ee.T
        delta_xyz_local = R_ee_world @ delta_xyz_world
        delta_rot_local = R_ee_world @ delta_rot_world
        
        return delta_xyz_local, delta_rot_local
    
    def calc_target_joint_position(self):
        """
        Calculate target joint positions using DLS IK.
        """
        if self.mode == "idle":
            return
        
        if self.mode == "homing":
            if not self._homing_started:
                print("[DLS] Homing: resetting arm to initial joints...")
                self.right_controller.reset()
                self._homing_started = True
            
            try:
                curr = np.asarray(self.right_controller.get_current_joint_degrees(), dtype=float).reshape(-1)
                err = float(np.max(np.abs(curr - self.right_home_q_deg)))
            except Exception:
                err = self.home_tol_deg + 1.0
            
            if err > self.home_tol_deg:
                return
            
            self.ee_target_position = None
            self.ee_target_quat = None
            self.init_controller_xyz = None
            self.init_controller_quat = None
            self.mode = "teleop"
            print("[DLS] Homing complete. Teleop enabled.")
            return
        
        current_q = self.right_controller.get_current_joint_positions()
        
        config = self.manipulator_config["right_arm"]
        xr_grip_val = self.xr_client.get_key_value_by_name(config["control_trigger"])
        active = xr_grip_val > (1.0 - CONTROLLER_DEADZONE)
        self.gripper_active = active
        
        current_pos, current_quat = self.forward_kinematics(current_q)
        
        from scipy.spatial.transform import Rotation as R
        current_rot = R.from_quat(current_quat)
        R_world_ee = current_rot.as_matrix()
        
        if active:
            if self.ee_target_position is None:
                self.ee_target_position = current_pos.copy()
                self.ee_target_quat = current_quat.copy()
                self.init_controller_xyz = None
                self.init_controller_quat = None
                print(f"[DLS] Activated. Current EE: pos={current_pos}, quat={current_quat}")
            
            xr_pose = self.xr_client.get_pose_by_name(config["pose_source"])
            delta_xyz_local, delta_rot_local = self._process_xr_pose(xr_pose, R_world_ee)
            
            self.ee_target_position = self.ee_target_position + R_world_ee @ delta_xyz_local
            
            delta_rot_quat = rotvec_to_quat(delta_rot_local)
            target_rot = R.from_quat(self.ee_target_quat)
            delta_rot = R.from_quat(delta_rot_quat)
            new_rot = target_rot * delta_rot
            self.ee_target_quat = new_rot.as_quat().astype(np.float32)
            
            result = solve_dls_6d(
                forward_fn=self.forward_fn_wrapper,
                q_seed=current_q,
                target_position=self.ee_target_position,
                target_quat=self.ee_target_quat,
                damping=self.dls_damping,
                gain=self.dls_gain,
                iterations=3,
                epsilon=1e-4,
                max_delta_per_joint=self.max_delta_per_joint,
                nominal_q=self.nominal_q,
                centering_gain=self.centering_gain,
                singularity_threshold=self.singularity_threshold,
            )
            
            if result.singularity_distance < self.singularity_threshold:
                print(f"[DLS] Near singularity (distance={result.singularity_distance:.4f}), decelerating")
            
            self.target_right_q = result.q_next
            
            delta_q = self.target_right_q - self.last_sent_q
            delta_q = np.clip(delta_q, -self.max_delta_per_joint, self.max_delta_per_joint)
            self.target_right_q = self.last_sent_q + delta_q
            
            self.right_controller.servo_joints(self.target_right_q)
            self.last_sent_q = self.target_right_q.copy()
            
        else:
            if self.ee_target_position is not None:
                print("[DLS] Deactivated.")
                self.ee_target_position = None
                self.ee_target_quat = None
                self.init_controller_xyz = None
                self.init_controller_quat = None
    
    def control_gripper(self):
        if self.mode != "teleop":
            return
        
        config = self.manipulator_config["right_arm"]
        trigger_val = self.xr_client.get_key_value_by_name(config["gripper_trigger"])
        
        self.right_gripper_pos = int(
            calc_parallel_gripper_position(
                self.right_controller.get_gripper_open_position(),
                self.right_controller.get_gripper_close_position(),
                trigger_val,
            )
        )
        
        self.right_controller.set_gripper_position(self.right_gripper_pos)
    
    def run_ik_and_control_thread(self, stop_event):
        print("[DLS] Starting IK and control thread...")
        while not stop_event.is_set():
            try:
                self.calc_target_joint_position()
                time.sleep(0.01)
            except Exception as e:
                print(f"[DLS] Error in IK: {e}")
        
        self.right_controller.close()
    
    def run_gripper_control_thread(self, stop_event):
        print("[DLS] Starting gripper control thread...")
        while not stop_event.is_set():
            self.control_gripper()
            time.sleep(0.01)
    
    def reset(self):
        self.right_controller.reset()
    
    def close(self):
        try:
            self.right_controller.close()
        except Exception:
            pass
    
    def run(self, stop_event=threading.Event()):
        try:
            self.reset()
            self.calc_target_joint_position()
        except Exception as e:
            print(f"[DLS] Error during initialization: {e}")
            self.close()
            return
        
        print("[DLS] Starting control loop...")
        
        try:
            ik_thread = threading.Thread(
                target=self.run_ik_and_control_thread,
                args=(stop_event,),
            )
            gripper_thread = threading.Thread(
                target=self.run_gripper_control_thread,
                args=(stop_event,),
            )
            
            ik_thread.start()
            gripper_thread.start()
            
            while not stop_event.is_set():
                try:
                    time.sleep(0.1)
                except KeyboardInterrupt:
                    print("[DLS] Keyboard interrupt. Stopping.")
                    stop_event.set()
            
            ik_thread.join()
            gripper_thread.join()
        
        except Exception as e:
            print(f"[DLS] Exception: {e}")
        
        self.close()
    
    def __del__(self):
        self.close()