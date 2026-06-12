import json
import socket
import time
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from quest_bridge.local_dls_ik import solve_local_dls_6d, solve_local_dls_position
from quest_bridge.realman_ik import RealManIK
from quest_bridge.trajectory_smoothing import HAS_RUCKIG, JointTrajectorySmoother


RM65_JOINT_LIMIT_MIN = [-3.1, -2.268, -2.355, -3.1, -2.233, -6.28]
RM65_JOINT_LIMIT_MAX = [3.1, 2.268, 2.355, 3.1, 2.233, 6.28]
DEFAULT_WRIST_REGULARIZATION_JOINTS = [3, 4, 5]


class WristIKBridge(Node):
    def __init__(self):
        super().__init__("wrist_ik_bridge")

        self.declare_parameter("port", 5005)
        self.declare_parameter("axis_mapping", "quest3_teleop_flip_forward")
        self.declare_parameter("position_scale", 1.0)
        self.declare_parameter("rotation_scale", 0.8)
        self.declare_parameter("position_target_mode", "absolute")
        self.declare_parameter("position_lpf_alpha", 1.0)
        self.declare_parameter("rotation_lpf_alpha", 1.0)
        self.declare_parameter("joint_lpf_alpha", 1.0)
        self.declare_parameter("packet_timeout", 0.25)
        self.declare_parameter("max_position_offset", 0.45)
        self.declare_parameter("max_rotation_error", 0.8)
        self.declare_parameter("max_joint_step", 0.05)
        self.declare_parameter("max_joint_step_per_joint", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter("trajectory_time", 0.05)
        self.declare_parameter("debug_log_period", 1.0)
        self.declare_parameter("debug_log_file", "")
        self.declare_parameter("controller_topic", "/arm_controller/joint_trajectory")
        self.declare_parameter("command_out_type", "trajectory_msgs/JointTrajectory")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("realman_lib_path", "")
        self.declare_parameter("robot_model", "rm65")
        self.declare_parameter("use_orientation", True)
        self.declare_parameter("arm_solver_mode", "local_dls")
        self.declare_parameter("rotation_control_mode", "joint6_only")
        self.declare_parameter("orientation_input_source", "landmark_frame")
        self.declare_parameter("hybrid_pose_rotation_scale", 0.25)
        self.declare_parameter("orientation_deadband_rad", 0.0)
        self.declare_parameter("orientation_target_mode", "relative")
        self.declare_parameter("orientation_fallback_to_position_only", True)
        self.declare_parameter("block_singularity", False)
        self.declare_parameter("max_ik_jump_norm", 0.0)
        self.declare_parameter("max_ik_jump_per_joint", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter("singularity_log_period", 1.0)
        self.declare_parameter("reset_to_initial_pose", True)
        self.declare_parameter("initial_joint_positions", [0.0, -0.7, 1.2, 0.0, 0.8, 0.0])
        self.declare_parameter("startup_reset_duration", 1.0)
        self.declare_parameter("startup_reset_tolerance", 0.015)
        self.declare_parameter("joint6_axis", "z")
        self.declare_parameter("joint6_scale", 1.0)
        self.declare_parameter("joint6_deadband_rad", 0.0)
        self.declare_parameter("joint6_min", -6.28)
        self.declare_parameter("joint6_max", 6.28)
        self.declare_parameter("max_joint6_step", 0.06)
        self.declare_parameter("tool_to_hand_translation", [0.0, 0.0, 0.04])
        self.declare_parameter("tool_to_hand_quat_xyzw", [0.0, 0.0, 0.0, 1.0])
        self.declare_parameter("dls_damping", 0.12)
        self.declare_parameter("dls_position_gain", 0.8)
        self.declare_parameter("dls_iterations", 2)
        self.declare_parameter("dls_fk_epsilon", 1e-3)
        self.declare_parameter("dls_position_tolerance", 2e-3)
        self.declare_parameter("dls_orientation_tolerance", 5e-2)
        self.declare_parameter("dls_centering_gain", 0.02)
        self.declare_parameter("dls_max_delta_per_joint", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter("dls_singularity_threshold", 0.1)
        self.declare_parameter("nominal_joint_positions", [0.0, -0.7, 1.2, 0.0, 0.8, 0.0])
        self.declare_parameter("current_q_regularization_weight", 0.0)
        self.declare_parameter("posture_regularization_weight", 0.0)
        self.declare_parameter("joint_limit_weight", 0.0)
        self.declare_parameter("wrist_regularization_weight", 0.0)
        self.declare_parameter("wrist_regularization_joints", DEFAULT_WRIST_REGULARIZATION_JOINTS)
        self.declare_parameter("joint_limit_min", RM65_JOINT_LIMIT_MIN)
        self.declare_parameter("joint_limit_max", RM65_JOINT_LIMIT_MAX)
        self.declare_parameter("trajectory_smoother", "auto")
        self.declare_parameter("trajectory_control_dt", 0.01)
        self.declare_parameter("ruckig_max_velocity", [0.20, 0.20, 0.20, 0.40, 0.40, 0.40])
        self.declare_parameter("ruckig_max_acceleration", [0.40, 0.40, 0.40, 0.80, 0.80, 0.80])
        self.declare_parameter("ruckig_max_jerk", [1.50, 1.50, 1.50, 3.00, 3.00, 3.00])

        self.port = int(self.get_parameter("port").value)
        self.axis_mapping = str(self.get_parameter("axis_mapping").value)
        self.position_scale = float(self.get_parameter("position_scale").value)
        self.rotation_scale = float(self.get_parameter("rotation_scale").value)
        self.position_target_mode = str(self.get_parameter("position_target_mode").value)
        if self.position_target_mode not in ("absolute", "incremental"):
            raise ValueError("position_target_mode must be 'absolute' or 'incremental'")
        self.position_lpf_alpha = float(self.get_parameter("position_lpf_alpha").value)
        self.rotation_lpf_alpha = float(self.get_parameter("rotation_lpf_alpha").value)
        self.joint_lpf_alpha = float(self.get_parameter("joint_lpf_alpha").value)
        self.packet_timeout = float(self.get_parameter("packet_timeout").value)
        self.max_position_offset = float(self.get_parameter("max_position_offset").value)
        self.max_rotation_error = float(self.get_parameter("max_rotation_error").value)
        self.max_joint_step = float(self.get_parameter("max_joint_step").value)
        max_joint_step_per_joint = np.array(
            self.get_parameter("max_joint_step_per_joint").value,
            dtype=np.float32,
        )
        if max_joint_step_per_joint.shape[0] != 6:
            raise ValueError("max_joint_step_per_joint must contain 6 values")
        self.max_joint_step_per_joint = (
            max_joint_step_per_joint if np.any(max_joint_step_per_joint > 0.0) else None
        )
        self.trajectory_time = float(self.get_parameter("trajectory_time").value)
        self.debug_log_period = float(self.get_parameter("debug_log_period").value)
        self.debug_log_file = str(self.get_parameter("debug_log_file").value)
        self.controller_topic = str(self.get_parameter("controller_topic").value)
        self.command_out_type = str(self.get_parameter("command_out_type").value)
        self.joint_state_topic = str(self.get_parameter("joint_state_topic").value)
        self.realman_lib_path = str(self.get_parameter("realman_lib_path").value)
        self.robot_model = str(self.get_parameter("robot_model").value)
        self.use_orientation = bool(self.get_parameter("use_orientation").value)
        self.arm_solver_mode = str(self.get_parameter("arm_solver_mode").value)
        self.rotation_control_mode = str(self.get_parameter("rotation_control_mode").value)
        self.orientation_input_source = str(
            self.get_parameter("orientation_input_source").value
        )
        self.hybrid_pose_rotation_scale = float(
            self.get_parameter("hybrid_pose_rotation_scale").value
        )
        self.orientation_deadband_rad = float(
            self.get_parameter("orientation_deadband_rad").value
        )
        self.orientation_target_mode = str(self.get_parameter("orientation_target_mode").value)
        if self.orientation_target_mode not in ("relative", "incremental"):
            raise ValueError("orientation_target_mode must be 'relative' or 'incremental'")
        if self.rotation_control_mode == "auto":
            self.rotation_control_mode = "full_pose" if self.use_orientation else "disabled"
        if self.rotation_control_mode not in ("disabled", "full_pose", "joint6_only", "hybrid"):
            raise ValueError(
                "rotation_control_mode must be 'auto', 'disabled', 'full_pose', "
                "'joint6_only', or 'hybrid'"
            )
        if self.arm_solver_mode not in ("official_ik", "local_dls"):
            raise ValueError("arm_solver_mode must be 'official_ik' or 'local_dls'")
        if self.orientation_input_source not in ("auto", "wrist_quaternion", "landmark_frame"):
            raise ValueError(
                "orientation_input_source must be 'auto', 'wrist_quaternion', or "
                "'landmark_frame'"
            )
        self.orientation_fallback_to_position_only = bool(
            self.get_parameter("orientation_fallback_to_position_only").value
        )
        self.block_singularity = bool(self.get_parameter("block_singularity").value)
        self.max_ik_jump_norm = float(self.get_parameter("max_ik_jump_norm").value)
        max_ik_jump_per_joint = np.array(
            self.get_parameter("max_ik_jump_per_joint").value,
            dtype=np.float32,
        )
        if max_ik_jump_per_joint.shape[0] != 6:
            raise ValueError("max_ik_jump_per_joint must contain 6 values")
        self.max_ik_jump_per_joint = (
            max_ik_jump_per_joint if np.any(max_ik_jump_per_joint > 0.0) else None
        )
        self.singularity_log_period = float(self.get_parameter("singularity_log_period").value)
        self.reset_to_initial_pose = bool(self.get_parameter("reset_to_initial_pose").value)
        self.initial_joint_positions = np.array(
            self.get_parameter("initial_joint_positions").value,
            dtype=np.float32,
        )
        if self.initial_joint_positions.shape[0] != 6:
            raise ValueError("initial_joint_positions must contain 6 arm joint values")
        self.startup_reset_duration = float(self.get_parameter("startup_reset_duration").value)
        self.startup_reset_tolerance = float(self.get_parameter("startup_reset_tolerance").value)
        self.joint6_axis = str(self.get_parameter("joint6_axis").value)
        if self.joint6_axis not in ("x", "y", "z"):
            raise ValueError("joint6_axis must be 'x', 'y', or 'z'")
        self.joint6_scale = float(self.get_parameter("joint6_scale").value)
        self.joint6_deadband_rad = float(self.get_parameter("joint6_deadband_rad").value)
        self.joint6_min = float(self.get_parameter("joint6_min").value)
        self.joint6_max = float(self.get_parameter("joint6_max").value)
        self.max_joint6_step = float(self.get_parameter("max_joint6_step").value)
        self.tool_to_hand_translation = np.array(
            self.get_parameter("tool_to_hand_translation").value,
            dtype=np.float32,
        )
        if self.tool_to_hand_translation.shape[0] != 3:
            raise ValueError("tool_to_hand_translation must contain 3 values")
        self.tool_to_hand_quat_xyzw = self.normalized_quat_xyzw(
            np.array(
                self.get_parameter("tool_to_hand_quat_xyzw").value,
                dtype=np.float32,
            )
        )
        if self.tool_to_hand_quat_xyzw.shape[0] != 4:
            raise ValueError("tool_to_hand_quat_xyzw must contain 4 values")
        self.tool_to_hand_rotation = self.quat_xyzw_to_matrix(self.tool_to_hand_quat_xyzw)
        self.dls_damping = float(self.get_parameter("dls_damping").value)
        self.dls_position_gain = float(self.get_parameter("dls_position_gain").value)
        self.dls_iterations = int(self.get_parameter("dls_iterations").value)
        self.dls_fk_epsilon = float(self.get_parameter("dls_fk_epsilon").value)
        self.dls_position_tolerance = float(self.get_parameter("dls_position_tolerance").value)
        self.dls_orientation_tolerance = float(self.get_parameter("dls_orientation_tolerance").value)
        self.dls_centering_gain = float(self.get_parameter("dls_centering_gain").value)
        dls_max_delta_per_joint = np.array(
            self.get_parameter("dls_max_delta_per_joint").value,
            dtype=np.float32,
        )
        if dls_max_delta_per_joint.shape[0] != 6:
            raise ValueError("dls_max_delta_per_joint must contain 6 values")
        self.dls_max_delta_per_joint = (
            dls_max_delta_per_joint
            if np.any(dls_max_delta_per_joint > 0.0)
            else self.max_joint_step_per_joint
        )
        self.dls_singularity_threshold = float(self.get_parameter("dls_singularity_threshold").value)
        self.nominal_joint_positions = np.array(
            self.get_parameter("nominal_joint_positions").value,
            dtype=np.float32,
        )
        if self.nominal_joint_positions.shape[0] != 6:
            raise ValueError("nominal_joint_positions must contain 6 values")
        self.current_q_regularization_weight = float(
            self.get_parameter("current_q_regularization_weight").value
        )
        self.posture_regularization_weight = float(
            self.get_parameter("posture_regularization_weight").value
        )
        self.joint_limit_weight = float(self.get_parameter("joint_limit_weight").value)
        self.wrist_regularization_weight = float(
            self.get_parameter("wrist_regularization_weight").value
        )
        self.wrist_regularization_joints = np.array(
            self.get_parameter("wrist_regularization_joints").value,
            dtype=np.int32,
        )
        self.joint_limit_min = np.array(
            self.get_parameter("joint_limit_min").value,
            dtype=np.float32,
        )
        self.joint_limit_max = np.array(
            self.get_parameter("joint_limit_max").value,
            dtype=np.float32,
        )
        for name, values in (
            ("joint_limit_min", self.joint_limit_min),
            ("joint_limit_max", self.joint_limit_max),
        ):
            if values.shape[0] != 6:
                raise ValueError(f"{name} must contain 6 values")
        self.trajectory_smoother_mode = str(self.get_parameter("trajectory_smoother").value)
        if self.trajectory_smoother_mode not in ("auto", "ruckig", "accel_limited", "none"):
            raise ValueError(
                "trajectory_smoother must be 'auto', 'ruckig', 'accel_limited', or 'none'"
            )
        self.trajectory_control_dt = float(self.get_parameter("trajectory_control_dt").value)
        self.ruckig_max_velocity = np.array(
            self.get_parameter("ruckig_max_velocity").value,
            dtype=np.float32,
        )
        self.ruckig_max_acceleration = np.array(
            self.get_parameter("ruckig_max_acceleration").value,
            dtype=np.float32,
        )
        self.ruckig_max_jerk = np.array(
            self.get_parameter("ruckig_max_jerk").value,
            dtype=np.float32,
        )
        for name, values in (
            ("ruckig_max_velocity", self.ruckig_max_velocity),
            ("ruckig_max_acceleration", self.ruckig_max_acceleration),
            ("ruckig_max_jerk", self.ruckig_max_jerk),
        ):
            if values.shape[0] != 6:
                raise ValueError(f"{name} must contain 6 values")

        self.quest_to_robot = self.make_quest_to_robot_matrix(self.axis_mapping)
        self.ik = RealManIK(
            lib_path=self.realman_lib_path,
            robot_model=self.robot_model,
            traversal_mode=False,
        )

        self.arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.current_joint_positions: Optional[np.ndarray] = None
        self.current_joint_velocities: Optional[np.ndarray] = None
        self.last_joint_state_positions: Optional[np.ndarray] = None
        self.last_joint_state_time: Optional[float] = None
        self.last_command_positions: Optional[np.ndarray] = None
        self.last_command_velocities: Optional[np.ndarray] = None

        self.latest_wrist: Optional[np.ndarray] = None
        self.latest_landmarks: Optional[np.ndarray] = None
        self.latest_wrist_time: Optional[float] = None
        self.last_udp_source = None
        self.packet_count = 0
        self.ik_count = 0
        self.ik_fail_count = 0
        self.command_count = 0

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
        self.last_debug_log_time = 0.0
        self.last_singularity_log_time = 0.0
        self.startup_reset_start_time: Optional[float] = None
        self.startup_reset_done = not self.reset_to_initial_pose

        if self.command_out_type == "std_msgs/Float64MultiArray":
            self.command_pub = self.create_publisher(Float64MultiArray, self.controller_topic, 10)
        elif self.command_out_type == "trajectory_msgs/JointTrajectory":
            self.command_pub = self.create_publisher(JointTrajectory, self.controller_topic, 10)
        else:
            raise ValueError(
                "command_out_type must be 'std_msgs/Float64MultiArray' or "
                "'trajectory_msgs/JointTrajectory'"
            )
        self.joint_state_sub = self.create_subscription(
            JointState,
            self.joint_state_topic,
            self.on_joint_state,
            10,
        )

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", self.port))
        self.sock.setblocking(False)

        self.timer = self.create_timer(0.01, self.step)
        self.trajectory_smoother = JointTrajectorySmoother(
            dof=6,
            dt=self.trajectory_control_dt,
            mode=self.trajectory_smoother_mode,
            max_velocity=self.ruckig_max_velocity,
            max_acceleration=self.ruckig_max_acceleration,
            max_jerk=self.ruckig_max_jerk,
        )

        self.get_logger().info(
            f"Listening for Quest wrist UDP on 0.0.0.0:{self.port}; "
            f"RealMan IK version={self.ik.version()}, lib={self.ik.lib_path}, "
            f"axis_mapping={self.axis_mapping}, output={self.controller_topic}, "
            f"command_out_type={self.command_out_type}, "
            f"arm_solver_mode={self.arm_solver_mode}, "
            f"position_target_mode={self.position_target_mode}, "
            f"rotation_control_mode={self.rotation_control_mode}, "
            f"orientation_input_source={self.orientation_input_source}, "
            f"hybrid_pose_rotation_scale={self.hybrid_pose_rotation_scale}, "
            f"orientation_target_mode={self.orientation_target_mode}, "
            f"block_singularity={self.block_singularity}, "
            f"max_ik_jump_norm={self.max_ik_jump_norm}, "
            f"max_ik_jump_per_joint="
            f"{None if self.max_ik_jump_per_joint is None else np.round(self.max_ik_jump_per_joint, 3).tolist()}, "
            f"dls_max_delta_per_joint="
            f"{None if self.dls_max_delta_per_joint is None else np.round(self.dls_max_delta_per_joint, 3).tolist()}, "
            f"dls_reg=(cur={self.current_q_regularization_weight:.3f}, "
            f"posture={self.posture_regularization_weight:.3f}, "
            f"limit={self.joint_limit_weight:.3f}, wrist={self.wrist_regularization_weight:.3f}), "
            f"trajectory_smoother={self.trajectory_smoother.mode} (ruckig_available={HAS_RUCKIG}), "
            f"tool_to_hand_translation={np.round(self.tool_to_hand_translation, 4).tolist()}, "
            f"tool_to_hand_quat_xyzw={np.round(self.tool_to_hand_quat_xyzw, 4).tolist()}, "
            f"reset_to_initial_pose={self.reset_to_initial_pose}, "
            f"initial_joints={np.round(self.initial_joint_positions, 3).tolist()}"
        )
        self.write_debug_file(
            f"start wrist_ik_bridge listen=0.0.0.0:{self.port}, "
            f"realman_ik={self.ik.version()}, lib={self.ik.lib_path}, "
            f"axis_mapping={self.axis_mapping}, output={self.controller_topic}, "
            f"command_out_type={self.command_out_type}, "
            f"arm_solver_mode={self.arm_solver_mode}, "
            f"position_target_mode={self.position_target_mode}, "
            f"rotation_control_mode={self.rotation_control_mode}, "
            f"hybrid_pose_rotation_scale={self.hybrid_pose_rotation_scale}, "
            f"orientation_target_mode={self.orientation_target_mode}, "
            f"block_singularity={self.block_singularity}, "
            f"max_ik_jump_norm={self.max_ik_jump_norm}, "
            f"max_ik_jump_per_joint="
            f"{None if self.max_ik_jump_per_joint is None else np.round(self.max_ik_jump_per_joint, 3).tolist()}, "
            f"dls_max_delta_per_joint="
            f"{None if self.dls_max_delta_per_joint is None else np.round(self.dls_max_delta_per_joint, 3).tolist()}, "
            f"dls_reg=(cur={self.current_q_regularization_weight:.3f}, "
            f"posture={self.posture_regularization_weight:.3f}, "
            f"limit={self.joint_limit_weight:.3f}, wrist={self.wrist_regularization_weight:.3f}), "
            f"trajectory_smoother={self.trajectory_smoother.mode} (ruckig_available={HAS_RUCKIG}), "
            f"tool_to_hand_translation={np.round(self.tool_to_hand_translation, 4).tolist()}, "
            f"tool_to_hand_quat_xyzw={np.round(self.tool_to_hand_quat_xyzw, 4).tolist()}, "
            f"reset_to_initial_pose={self.reset_to_initial_pose}, "
            f"initial_joints={np.round(self.initial_joint_positions, 3).tolist()}"
        )

    def on_joint_state(self, msg: JointState):
        positions = []
        for joint_name in self.arm_joint_names:
            try:
                index = msg.name.index(joint_name)
            except ValueError:
                return
            positions.append(msg.position[index])
        current_positions = np.array(positions, dtype=np.float32)
        now = time.time()
        if len(msg.velocity) >= len(self.arm_joint_names):
            velocities = []
            for joint_name in self.arm_joint_names:
                index = msg.name.index(joint_name)
                velocities.append(msg.velocity[index])
            self.current_joint_velocities = np.array(velocities, dtype=np.float32)
        elif self.last_joint_state_positions is not None and self.last_joint_state_time is not None:
            dt = max(now - self.last_joint_state_time, 1e-4)
            self.current_joint_velocities = (
                (current_positions - self.last_joint_state_positions) / dt
            ).astype(np.float32)
        else:
            self.current_joint_velocities = np.zeros(6, dtype=np.float32)
        self.current_joint_positions = current_positions
        self.last_joint_state_positions = current_positions.copy()
        self.last_joint_state_time = now

    def step(self):
        latest_data = None
        try:
            while True:
                latest_data, self.last_udp_source = self.sock.recvfrom(65535)
        except BlockingIOError:
            pass

        if latest_data is not None:
            packet = self.parse_wrist_packet(latest_data)
            if packet is not None:
                wrist, landmarks = packet
                self.latest_wrist = wrist
                self.latest_landmarks = landmarks
                self.latest_wrist_time = time.time()
                self.packet_count += 1

        if self.current_joint_positions is None:
            self.log_debug("waiting for /joint_states")
            return
        if not self.startup_reset_done:
            self.publish_startup_reset()
            return
        if self.latest_wrist is None:
            self.log_debug("waiting for wrist_pose")
            return
        if (
            self.latest_wrist_time is not None
            and time.time() - self.latest_wrist_time > self.packet_timeout
        ):
            self.hold_last_command()
            self.reset_tracking(preserve_last_command=True)
            self.log_debug("wrist_pose stale, holding output")
            return

        self.publish_ik_follow(self.latest_wrist, self.latest_landmarks)

    def publish_startup_reset(self):
        now = time.time()
        if self.startup_reset_start_time is None:
            self.startup_reset_start_time = now
            self.get_logger().info(
                "Resetting arm to fixed initial pose before wrist calibration: "
                f"{np.round(self.initial_joint_positions, 3).tolist()}"
            )
            self.write_debug_file(
                "reset arm to fixed initial pose before wrist calibration: "
                f"{np.round(self.initial_joint_positions, 3).tolist()}"
            )

        self.publish_arm_command(self.initial_joint_positions)
        self.last_command_positions = self.initial_joint_positions.copy()

        joint_error = float(np.linalg.norm(self.current_joint_positions - self.initial_joint_positions))
        elapsed = now - self.startup_reset_start_time
        if joint_error <= self.startup_reset_tolerance:
            self.startup_reset_done = True
            self.reset_tracking()
            self.last_command_positions = self.initial_joint_positions.copy()
            self.last_command_velocities = np.zeros(6, dtype=np.float32)
            self.trajectory_smoother.reset(
                self.initial_joint_positions,
                self.last_command_velocities,
            )
            self.get_logger().info(
                "Fixed initial pose ready. The next fresh wrist frame will be used as zero reference."
            )
            self.write_debug_file(
                "fixed initial pose ready; next fresh wrist frame becomes zero reference"
            )
            return

        self.log_debug(
            "resetting to fixed initial pose before wrist calibration, "
            f"joint_error={joint_error:.4f}, elapsed={elapsed:.2f}s"
        )

    def reset_tracking(self, preserve_last_command: bool = False):
        self.prev_wrist_position = None
        self.prev_wrist_rotation = None
        self.filtered_wrist_position = None
        self.filtered_wrist_quat_xyzw = None
        self.wrist_origin_position = None
        self.wrist_rotation_origin = None
        self.target_position = None
        self.target_rotation = None
        self.ee_origin = None
        self.ee_rotation_origin = None
        self.hand_origin_position = None
        self.hand_rotation_origin = None
        self.joint6_origin_position = None
        self.target_joint6_position = None
        if not preserve_last_command:
            self.last_command_positions = None

    def publish_ik_follow(self, wrist: np.ndarray, landmarks: Optional[np.ndarray] = None):
        if wrist.shape[0] < 7:
            self.get_logger().warning("IK follow needs wrist_pose [x, y, z, qx, qy, qz, qw]")
            return

        current_joints = self.current_joint_positions.copy()
        ee_position, ee_quat_xyzw = self.ik.forward(current_joints)
        ee_rotation = self.quat_xyzw_to_matrix(ee_quat_xyzw)
        hand_position, hand_rotation = self.tool_pose_to_hand_pose(ee_position, ee_rotation)
        wrist_position_raw, wrist_quat_xyzw_raw = self.resolve_wrist_tracking_inputs(
            wrist,
            landmarks,
        )
        wrist_position = self.low_pass_vector(
            self.filtered_wrist_position,
            wrist_position_raw,
            self.position_lpf_alpha,
        )
        wrist_quat_xyzw = self.low_pass_quat_xyzw(
            self.filtered_wrist_quat_xyzw,
            wrist_quat_xyzw_raw,
            self.rotation_lpf_alpha,
        )
        self.filtered_wrist_position = wrist_position.copy()
        self.filtered_wrist_quat_xyzw = wrist_quat_xyzw.copy()
        wrist_rotation = self.quat_xyzw_to_matrix(wrist_quat_xyzw)

        if (
            self.prev_wrist_position is None
            or self.prev_wrist_rotation is None
            or self.wrist_origin_position is None
            or self.wrist_rotation_origin is None
            or self.target_position is None
            or self.target_rotation is None
        ):
            self.prev_wrist_position = wrist_position.copy()
            self.prev_wrist_rotation = wrist_rotation.copy()
            self.wrist_origin_position = wrist_position.copy()
            self.wrist_rotation_origin = wrist_rotation.copy()
            self.target_position = ee_position.copy()
            self.target_rotation = ee_rotation.copy()
            self.ee_origin = ee_position.copy()
            self.ee_rotation_origin = ee_rotation.copy()
            self.hand_origin_position = hand_position.copy()
            self.hand_rotation_origin = hand_rotation.copy()
            self.joint6_origin_position = float(current_joints[5])
            self.target_joint6_position = float(current_joints[5])
            self.last_command_positions = current_joints.copy()
            self.last_command_velocities = (
                np.zeros(6, dtype=np.float32)
                if self.current_joint_velocities is None
                else self.current_joint_velocities.copy()
            )
            self.trajectory_smoother.reset(current_joints, self.last_command_velocities)
            self.get_logger().info(
                "Calibrated wrist IK follow origin. Current wrist deltas will drive "
                f"{'local DLS' if self.arm_solver_mode == 'local_dls' else 'RealMan official IK'} "
                "and publish arm joint trajectories."
            )
            self.log_debug("calibrated IK origin, no command sent yet")
            return

        delta_position_world = (wrist_position - self.prev_wrist_position) * self.position_scale
        delta_rotation_world = self.rotation_matrix_to_rotvec(
            wrist_rotation @ self.prev_wrist_rotation.T
        ) * self.rotation_scale

        self.prev_wrist_position = wrist_position.copy()
        self.prev_wrist_rotation = wrist_rotation.copy()

        target_position_before = self.target_position.copy()
        target_rotation_before = self.target_rotation.copy()
        target_hand_position_before, target_hand_rotation_before = self.tool_pose_to_hand_pose(
            target_position_before,
            target_rotation_before,
        )
        if self.position_target_mode == "absolute":
            wrist_offset_world = (
                wrist_position - self.wrist_origin_position
            ) * self.position_scale
            target_hand_position = self.hand_origin_position + wrist_offset_world
        else:
            delta_position_local_current = hand_rotation.T @ delta_position_world
            target_to_current = target_hand_rotation_before.T @ hand_rotation
            delta_position_local_target = target_to_current @ delta_position_local_current
            target_hand_position = (
                target_hand_position_before
                + target_hand_rotation_before @ delta_position_local_target
            )
        if self.rotation_control_mode == "full_pose":
            if self.orientation_target_mode == "relative":
                relative_rotation_world = wrist_rotation @ self.wrist_rotation_origin.T
                scaled_relative_rotation_world = self.scale_rotation_matrix(
                    relative_rotation_world,
                    self.rotation_scale,
                )
                target_hand_rotation = (
                    scaled_relative_rotation_world @ self.hand_rotation_origin
                )
            else:
                delta_rotation_local_current = hand_rotation.T @ delta_rotation_world
                target_to_current = target_hand_rotation_before.T @ hand_rotation
                delta_rotation_local_target = target_to_current @ delta_rotation_local_current
                target_hand_rotation = target_hand_rotation_before @ self.rotvec_to_matrix(
                    delta_rotation_local_target
                )
        elif self.rotation_control_mode == "joint6_only":
            # In joint6-only mode, IK should solve position against a fixed wrist/hand
            # orientation. The human wrist rotation is mapped onto joint6 separately below.
            target_hand_rotation = self.hand_rotation_origin.copy()
        elif self.rotation_control_mode == "hybrid":
            # Keep the main wrist twist for joint6, but feed a light-weight orientation
            # target into IK on the remaining axes so the workspace is less cramped.
            target_hand_rotation = self.hybrid_target_hand_rotation(wrist_rotation)
        else:
            target_hand_rotation = self.hand_rotation_origin.copy()

        target_hand_position, target_hand_rotation, position_clipped = self.limit_hand_target(
            target_hand_position,
            target_hand_rotation,
        )
        self.target_position, self.target_rotation = self.hand_pose_to_tool_pose(
            target_hand_position,
            target_hand_rotation,
        )
        target_quat_xyzw = self.matrix_to_quat_xyzw(self.target_rotation)

        seed_joints = (
            current_joints.copy()
            if self.arm_solver_mode == "local_dls"
            else (
                self.last_command_positions
                if self.last_command_positions is not None
                else current_joints
            )
        )
        ret = 0
        local_dls_singularity_distance: Optional[float] = None
        if self.arm_solver_mode == "local_dls":
            active_mask = np.ones(6, dtype=bool)
            if self.rotation_control_mode in ("joint6_only", "hybrid"):
                active_mask[5] = False
            if self.rotation_control_mode == "full_pose":
                dls_result = solve_local_dls_6d(
                    forward_fn=lambda q: self.ik.forward(q),
                    q_seed=seed_joints,
                    target_position=self.target_position,
                    target_quat=target_quat_xyzw,
                    damping=self.dls_damping,
                    gain=self.dls_position_gain,
                    iterations=self.dls_iterations,
                    epsilon=self.dls_fk_epsilon,
                    position_tolerance=self.dls_position_tolerance,
                    orientation_tolerance=self.dls_orientation_tolerance,
                    active_mask=active_mask,
                    max_delta_per_joint=self.dls_max_delta_per_joint,
                    current_q=current_joints,
                    nominal_q=self.nominal_joint_positions,
                    current_q_weight=self.current_q_regularization_weight,
                    posture_weight=self.posture_regularization_weight,
                    centering_gain=self.dls_centering_gain,
                    joint_limit_min=self.joint_limit_min,
                    joint_limit_max=self.joint_limit_max,
                    joint_limit_weight=self.joint_limit_weight,
                    wrist_joint_indices=self.wrist_regularization_joints,
                    wrist_weight=self.wrist_regularization_weight,
                    singularity_threshold=self.dls_singularity_threshold,
                )
                ik_joints = dls_result.q_next
                local_dls_singularity_distance = dls_result.singularity_distance
            else:
                dls_result = solve_local_dls_position(
                    forward_fn=lambda q: self.ik.forward(q),
                    q_seed=seed_joints,
                    target_position=self.target_position,
                    damping=self.dls_damping,
                    gain=self.dls_position_gain,
                    iterations=self.dls_iterations,
                    epsilon=self.dls_fk_epsilon,
                    tolerance=self.dls_position_tolerance,
                    active_mask=active_mask,
                    max_delta_per_joint=self.dls_max_delta_per_joint,
                    current_q=current_joints,
                    nominal_q=self.nominal_joint_positions,
                    current_q_weight=self.current_q_regularization_weight,
                    posture_weight=self.posture_regularization_weight,
                    centering_gain=self.dls_centering_gain,
                    joint_limit_min=self.joint_limit_min,
                    joint_limit_max=self.joint_limit_max,
                    joint_limit_weight=self.joint_limit_weight,
                    wrist_joint_indices=self.wrist_regularization_joints,
                    wrist_weight=self.wrist_regularization_weight,
                )
                ik_joints = dls_result.q_next
            self.ik_count += 1
        else:
            ret, ik_joints = self.ik.inverse(seed_joints, self.target_position, target_quat_xyzw)
            self.ik_count += 1
        used_position_only_fallback = False
        used_backoff_factor: Optional[float] = None

        if (
            self.arm_solver_mode != "local_dls"
            and
            ret != 0
            and self.rotation_control_mode in ("full_pose", "hybrid")
            and self.use_orientation
            and self.orientation_fallback_to_position_only
        ):
            fallback_ret, fallback_joints = self.ik.inverse(
                seed_joints,
                self.target_position,
                ee_quat_xyzw,
            )
            self.ik_count += 1
            if fallback_ret == 0 and fallback_joints is not None:
                ik_joints = fallback_joints
                target_quat_xyzw = ee_quat_xyzw
                self.target_rotation = ee_rotation.copy()
                ret = fallback_ret
                used_position_only_fallback = True

        position_limit_ret = None
        singularity_ret = None
        singularity_distance = local_dls_singularity_distance
        continuity_blocked = False
        continuity_norm = 0.0
        continuity_max = 0.0
        used_joint_step_rescue = False

        if ret != 0:
            self.target_position = target_position_before
            self.target_rotation = target_rotation_before
            self.ik_fail_count += 1
            self.hold_last_command()
            self.log_ik_debug(
                "IK failed, holding last command",
                wrist_position,
                ee_position,
                delta_position_world,
                delta_rotation_world,
                target_quat_xyzw,
                current_joints,
                None,
                None,
                position_clipped,
                ik_ret=ret,
                singularity_ret=None,
            )
            return

        def evaluate_candidate(candidate_joints: np.ndarray) -> tuple[bool, Optional[int], Optional[int], Optional[float], bool, float, float]:
            limit_ret = self.ik.check_position_limit(candidate_joints)
            if limit_ret != 0:
                return False, limit_ret, None, None, False, 0.0, 0.0
            cand_singularity_ret, cand_singularity_distance = self.ik.singularity(candidate_joints)
            if cand_singularity_ret != 0 and self.block_singularity:
                return False, limit_ret, cand_singularity_ret, cand_singularity_distance, False, 0.0, 0.0
            cand_continuity_blocked, cand_continuity_norm, cand_continuity_max = self.check_ik_continuity(
                seed_joints, candidate_joints
            )
            if cand_continuity_blocked:
                return (
                    False,
                    limit_ret,
                    cand_singularity_ret,
                    cand_singularity_distance,
                    True,
                    cand_continuity_norm,
                    cand_continuity_max,
                )
            return (
                True,
                limit_ret,
                cand_singularity_ret,
                cand_singularity_distance,
                False,
                cand_continuity_norm,
                cand_continuity_max,
            )

        accepted, position_limit_ret, singularity_ret, singularity_distance, continuity_blocked, continuity_norm, continuity_max = evaluate_candidate(
            ik_joints
        )

        if not accepted and self.arm_solver_mode != "local_dls":
            backoff_factors = [0.75, 0.5, 0.25, 0.1]
            for factor in backoff_factors:
                candidate_position = ee_position + (self.target_position - ee_position) * factor
                candidate_rotation = self.interpolate_rotation(
                    ee_rotation,
                    self.target_rotation,
                    factor,
                )
                candidate_quat_xyzw = self.matrix_to_quat_xyzw(candidate_rotation)
                backoff_ret, backoff_joints = self.ik.inverse(
                    seed_joints,
                    candidate_position,
                    candidate_quat_xyzw,
                )
                self.ik_count += 1
                if backoff_ret != 0:
                    continue
                (
                    accepted,
                    position_limit_ret,
                    singularity_ret,
                    singularity_distance,
                    continuity_blocked,
                    continuity_norm,
                    continuity_max,
                ) = evaluate_candidate(backoff_joints)
                if accepted:
                    self.target_position = candidate_position
                    self.target_rotation = candidate_rotation
                    target_quat_xyzw = candidate_quat_xyzw
                    ik_joints = backoff_joints
                    ret = backoff_ret
                    used_backoff_factor = factor
                    break

        if not accepted and self.arm_solver_mode != "local_dls":
            stepped_candidate = self.limit_joint_step(seed_joints, ik_joints)
            (
                stepped_accepted,
                stepped_position_limit_ret,
                stepped_singularity_ret,
                stepped_singularity_distance,
                stepped_continuity_blocked,
                stepped_continuity_norm,
                stepped_continuity_max,
            ) = evaluate_candidate(stepped_candidate)
            if stepped_accepted:
                accepted = True
                ik_joints = stepped_candidate
                position_limit_ret = stepped_position_limit_ret
                singularity_ret = stepped_singularity_ret
                singularity_distance = stepped_singularity_distance
                continuity_blocked = stepped_continuity_blocked
                continuity_norm = stepped_continuity_norm
                continuity_max = stepped_continuity_max
                used_joint_step_rescue = True

        if not accepted:
            self.target_position = target_position_before
            self.target_rotation = target_rotation_before
            self.ik_fail_count += 1
            self.hold_last_command()
            if position_limit_ret is not None and position_limit_ret != 0:
                status = "IK joint position limit blocked, holding last command"
            elif singularity_ret is not None and singularity_ret != 0 and self.block_singularity:
                status = "IK singularity blocked, holding last command"
            elif continuity_blocked:
                status = (
                    "IK continuity blocked, holding last command "
                    f"(jump_norm={continuity_norm:.3f}, jump_max={continuity_max:.3f})"
                )
            else:
                status = "IK blocked by guard, holding last command"
            self.log_ik_debug(
                status,
                wrist_position,
                ee_position,
                delta_position_world,
                delta_rotation_world,
                target_quat_xyzw,
                current_joints,
                ik_joints,
                None,
                position_clipped,
                ik_ret=ret,
                singularity_ret=singularity_ret,
                singularity_distance=singularity_distance,
                limit_ret=position_limit_ret,
            )
            return

        command_joints = self.limit_joint_step(seed_joints, ik_joints)
        if self.rotation_control_mode in ("joint6_only", "hybrid"):
            command_joints[5] = self.compute_joint6_target_from_wrist_rotation(wrist_rotation)
        current_dq = (
            np.zeros(6, dtype=np.float32)
            if self.current_joint_velocities is None
            else self.current_joint_velocities.copy()
        )
        smoothing_result = self.trajectory_smoother.step(
            current_q=current_joints,
            current_dq=current_dq,
            target_q=command_joints,
        )
        command_joints = smoothing_result.q_cmd
        self.last_command_velocities = smoothing_result.dq_cmd.copy()
        if self.last_command_positions is not None and self.joint_lpf_alpha < 1.0:
            command_joints = self.low_pass_vector(
                self.last_command_positions,
                command_joints,
                self.joint_lpf_alpha,
            )
        self.publish_arm_command(command_joints)
        self.last_command_positions = command_joints.copy()
        self.log_ik_debug(
            (
                "IK follow (joint-step rescue)"
                if used_joint_step_rescue
                else
                f"IK follow (continuity backoff x{used_backoff_factor:.2f})"
                if used_backoff_factor is not None
                else ("IK follow (position-only fallback)" if used_position_only_fallback else "IK follow")
            ),
            wrist_position,
            ee_position,
            delta_position_world,
            delta_rotation_world,
            target_quat_xyzw,
            current_joints,
            ik_joints,
            command_joints,
            position_clipped,
            ik_ret=ret,
            singularity_ret=singularity_ret,
            singularity_distance=singularity_distance,
        )

    def limit_hand_target(
        self,
        target_hand_position: np.ndarray,
        target_hand_rotation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        if self.hand_origin_position is None or self.hand_rotation_origin is None:
            return target_hand_position, target_hand_rotation, False

        position_clipped = False
        if self.max_position_offset > 0.0:
            target_offset = target_hand_position - self.hand_origin_position
            limited_offset = self.limit_vector(target_offset, self.max_position_offset)
            position_clipped = np.linalg.norm(limited_offset - target_offset) > 1e-6
            target_hand_position = self.hand_origin_position + limited_offset

        if self.max_rotation_error > 0.0 and self.rotation_control_mode in ("full_pose", "hybrid"):
            target_rotvec = self.rotation_matrix_to_rotvec(
                target_hand_rotation @ self.hand_rotation_origin.T
            )
            limited_rotvec = self.limit_vector(target_rotvec, self.max_rotation_error)
            target_hand_rotation = (
                self.rotvec_to_matrix(limited_rotvec) @ self.hand_rotation_origin
            )

        return target_hand_position, target_hand_rotation, position_clipped

    def limit_joint_step(self, reference: np.ndarray, target: np.ndarray) -> np.ndarray:
        if self.max_joint_step_per_joint is not None:
            max_step = self.max_joint_step_per_joint
        elif self.max_joint_step > 0.0:
            max_step = np.full(6, self.max_joint_step, dtype=np.float32)
        else:
            return target.copy()
        delta = target - reference
        limited_delta = np.clip(delta, -max_step, max_step)
        return reference + limited_delta

    def check_ik_continuity(
        self, reference: np.ndarray, target: np.ndarray
    ) -> tuple[bool, float, float]:
        delta = target - reference
        delta_norm = float(np.linalg.norm(delta))
        delta_max = float(np.max(np.abs(delta)))
        if self.max_ik_jump_norm > 0.0 and delta_norm > self.max_ik_jump_norm:
            return True, delta_norm, delta_max
        if self.max_ik_jump_per_joint is not None and np.any(
            np.abs(delta) > self.max_ik_jump_per_joint
        ):
            return True, delta_norm, delta_max
        return False, delta_norm, delta_max

    def interpolate_rotation(
        self, start_rotation: np.ndarray, target_rotation: np.ndarray, factor: float
    ) -> np.ndarray:
        if factor <= 0.0:
            return start_rotation.copy()
        if factor >= 1.0:
            return target_rotation.copy()
        relative_rotation = target_rotation @ start_rotation.T
        return self.scale_rotation_matrix(relative_rotation, factor) @ start_rotation

    def publish_arm_command(self, positions: np.ndarray):
        if self.command_out_type == "std_msgs/Float64MultiArray":
            msg = Float64MultiArray()
            msg.data = [float(value) for value in positions]
            self.command_pub.publish(msg)
            self.command_count += 1
            return

        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = list(self.arm_joint_names)

        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in positions]
        point.time_from_start = DurationMsg(
            sec=int(self.trajectory_time),
            nanosec=int((self.trajectory_time % 1.0) * 1e9),
        )
        msg.points.append(point)
        self.command_pub.publish(msg)
        self.command_count += 1

    def hold_last_command(self):
        if self.last_command_positions is None:
            return
        self.publish_arm_command(self.last_command_positions)

    def compute_joint6_target_from_wrist_rotation(self, wrist_rotation: np.ndarray) -> float:
        if (
            self.wrist_rotation_origin is None
            or self.hand_rotation_origin is None
            or self.joint6_origin_position is None
        ):
            if self.target_joint6_position is not None:
                return float(self.target_joint6_position)
            if self.current_joint_positions is not None:
                return float(self.current_joint_positions[5])
            return 0.0

        axis_index = self.joint6_axis_index()
        relative_rotation_world = wrist_rotation @ self.wrist_rotation_origin.T
        relative_rotation_local = (
            self.hand_rotation_origin.T @ relative_rotation_world @ self.hand_rotation_origin
        )
        axis_value = self.twist_angle_about_axis(relative_rotation_local, axis_index)
        if self.joint6_deadband_rad > 0.0:
            abs_axis = abs(axis_value)
            if abs_axis <= self.joint6_deadband_rad:
                axis_value = 0.0
            else:
                axis_value = np.sign(axis_value) * (abs_axis - self.joint6_deadband_rad)
        desired_joint6 = float(
            self.joint6_origin_position + axis_value * self.joint6_scale
        )
        desired_joint6 = float(np.clip(desired_joint6, self.joint6_min, self.joint6_max))

        if self.target_joint6_position is None:
            if self.current_joint_positions is not None:
                self.target_joint6_position = float(self.current_joint_positions[5])
            else:
                self.target_joint6_position = desired_joint6

        if self.max_joint6_step > 0.0:
            delta = desired_joint6 - self.target_joint6_position
            delta = float(np.clip(delta, -self.max_joint6_step, self.max_joint6_step))
            self.target_joint6_position = float(self.target_joint6_position + delta)
        else:
            self.target_joint6_position = desired_joint6

        return float(self.target_joint6_position)

    def log_debug(self, status: str):
        if self.debug_log_period <= 0.0:
            return
        now = time.time()
        if now - self.last_debug_log_time < self.debug_log_period:
            return
        self.last_debug_log_time = now
        wrist_text = "None" if self.latest_wrist is None else np.round(self.latest_wrist, 4).tolist()
        message = (
            f"{status}: source={self.last_udp_source}, packets={self.packet_count}, "
            f"ik={self.ik_count}, ik_fail={self.ik_fail_count}, commands={self.command_count}, "
            f"wrist_pose={wrist_text}"
        )
        self.get_logger().info(message)
        self.write_debug_file(message)

    def log_ik_debug(
        self,
        status: str,
        wrist_position: np.ndarray,
        ee_position: np.ndarray,
        delta_position_world: np.ndarray,
        delta_rotation_world: np.ndarray,
        target_quat_xyzw: np.ndarray,
        current_joints: np.ndarray,
        ik_joints: Optional[np.ndarray],
        command_joints: Optional[np.ndarray],
        position_clipped: bool,
        ik_ret: int,
        singularity_ret: Optional[int],
        singularity_distance: Optional[float] = None,
        limit_ret: Optional[int] = None,
    ):
        if self.debug_log_period <= 0.0:
            return
        now = time.time()
        if now - self.last_debug_log_time < self.debug_log_period:
            return
        self.last_debug_log_time = now
        ik_text = "None" if ik_joints is None else np.round(ik_joints, 3).tolist()
        cmd_text = "None" if command_joints is None else np.round(command_joints, 3).tolist()
        singularity_text = (
            "None"
            if singularity_ret is None
            else f"{singularity_ret}, distance={singularity_distance}"
        )
        message = (
            f"{status}: source={self.last_udp_source}, packets={self.packet_count}, "
            f"ik={self.ik_count}, ik_fail={self.ik_fail_count}, commands={self.command_count}, "
            f"wrist_robot={np.round(wrist_position, 3).tolist()}, "
            f"ee_fk={np.round(ee_position, 3).tolist()}, "
            f"target={np.round(self.target_position, 3).tolist()}, "
            f"target_quat_xyzw={np.round(target_quat_xyzw, 3).tolist()}, "
            f"delta_pos={np.round(delta_position_world, 4).tolist()}, "
            f"delta_rot={np.round(delta_rotation_world, 4).tolist()}, "
            f"position_clipped={position_clipped}, ik_ret={ik_ret}, "
            f"limit_ret={limit_ret}, singularity={singularity_text}, "
            f"current={np.round(current_joints, 3).tolist()}, ik_joints={ik_text}, command={cmd_text}"
        )
        self.get_logger().info(message)
        self.write_debug_file(message)

    def write_debug_file(self, message: str):
        if not self.debug_log_file:
            return
        path = Path(self.debug_log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with path.open("a", encoding="utf-8") as file:
            file.write(f"[{timestamp}] [ROS2-IK] {message}\n")

    def parse_wrist_packet(
        self, data: bytes
    ) -> Optional[tuple[np.ndarray, Optional[np.ndarray]]]:
        text = data.decode("utf-8", errors="ignore").strip()
        if not text:
            return None

        try:
            packet = json.loads(text)
            wrist = np.array(packet["wrist_pose"], dtype=np.float32)
            landmarks = self.parse_landmark_array(packet.get("landmarks"))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            try:
                values = [float(v.strip()) for v in text.split(":", 1)[1].split(",") if v.strip()]
                wrist = np.array(values, dtype=np.float32)
                landmarks = None
            except (IndexError, ValueError):
                self.get_logger().warning(f"Cannot parse wrist packet: {text[:80]}")
                return None

        if wrist.shape[0] < 3:
            self.get_logger().warning("Wrist packet must contain at least x,y,z")
            return None
        return wrist, landmarks

    @staticmethod
    def parse_landmark_array(landmarks_value: object) -> Optional[np.ndarray]:
        if landmarks_value is None:
            return None
        try:
            landmarks = np.array(landmarks_value, dtype=np.float32)
        except (TypeError, ValueError):
            return None
        if landmarks.shape != (21, 3):
            return None
        return landmarks

    def should_use_landmark_frame(self, landmarks: Optional[np.ndarray]) -> bool:
        if landmarks is None:
            return False
        if self.orientation_input_source == "landmark_frame":
            return True
        if self.orientation_input_source == "wrist_quaternion":
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

        # Re-orthogonalize once to keep the frame well-conditioned.
        x_axis = self.normalize_vector(np.cross(y_axis, z_axis))
        if x_axis is None:
            return None

        return np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32)

    def resolve_wrist_tracking_inputs(
        self,
        wrist: np.ndarray,
        landmarks: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        wrist_position_raw = self.map_quest_to_robot_vector(wrist[:3])
        wrist_quat_xyzw_raw = self.quest_quat_to_robot(wrist[3:7])

        if not self.should_use_landmark_frame(landmarks):
            return wrist_position_raw, wrist_quat_xyzw_raw

        landmarks_robot = self.map_quest_landmarks_to_robot(landmarks)
        landmark_rotation = self.landmark_frame_to_rotation(landmarks_robot)
        if landmark_rotation is None:
            return wrist_position_raw, wrist_quat_xyzw_raw

        return wrist_position_raw, self.matrix_to_quat_xyzw(landmark_rotation)

    @staticmethod
    def make_quest_to_robot_matrix(axis_mapping: str) -> np.ndarray:
        if axis_mapping == "quest3_teleop":
            return np.array(
                [
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            )
        if axis_mapping == "quest3_teleop_flip_forward":
            return np.array(
                [
                    [0.0, 0.0, -1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            )
        if axis_mapping == "quest3_teleop_flip_forward_up_y_left_z":
            # Keep Quest forward/back on robot base x, but remap hand up to base +y
            # and hand left/right to base +/-z for the current wrist teleop workflow.
            return np.array(
                [
                    [0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            )
        if axis_mapping == "base_y_left":
            return np.array(
                [
                    [0.0, 0.0, 1.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            )
        if axis_mapping == "base_y_left_flip_forward":
            return np.array(
                [
                    [0.0, 0.0, -1.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            )
        if axis_mapping == "identity":
            return np.eye(3, dtype=np.float32)
        raise ValueError(
            "axis_mapping must be 'quest3_teleop', 'quest3_teleop_flip_forward', "
            "'quest3_teleop_flip_forward_up_y_left_z', 'base_y_left', "
            "'base_y_left_flip_forward', or 'identity'"
        )

    def map_quest_to_robot_vector(self, vector: np.ndarray) -> np.ndarray:
        return self.quest_to_robot @ vector

    def tool_pose_to_hand_pose(
        self,
        tool_position: np.ndarray,
        tool_rotation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        hand_rotation = tool_rotation @ self.tool_to_hand_rotation
        hand_position = tool_position + tool_rotation @ self.tool_to_hand_translation
        return hand_position, hand_rotation

    def hand_pose_to_tool_pose(
        self,
        hand_position: np.ndarray,
        hand_rotation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        tool_rotation = hand_rotation @ self.tool_to_hand_rotation.T
        tool_position = hand_position - tool_rotation @ self.tool_to_hand_translation
        return tool_position, tool_rotation

    def quest_quat_to_robot(self, quat_xyzw: np.ndarray) -> np.ndarray:
        rotation = self.quest_rotation_to_robot(quat_xyzw)
        return self.matrix_to_quat_xyzw(rotation)

    def quest_rotation_to_robot(self, quat_xyzw: np.ndarray) -> np.ndarray:
        quest_rotation = self.quat_xyzw_to_matrix(quat_xyzw)
        return self.quest_to_robot @ quest_rotation @ self.quest_to_robot.T

    @staticmethod
    def low_pass_vector(previous: Optional[np.ndarray], current: np.ndarray, alpha: float) -> np.ndarray:
        if previous is None or alpha >= 1.0:
            return current.copy()
        if alpha <= 0.0:
            return previous.copy()
        return previous + alpha * (current - previous)

    @staticmethod
    def low_pass_quat_xyzw(
        previous: Optional[np.ndarray],
        current: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        current_norm = WristIKBridge.normalized_quat_xyzw(current)
        if previous is None or alpha >= 1.0:
            return current_norm
        if alpha <= 0.0:
            return WristIKBridge.normalized_quat_xyzw(previous)
        prev_norm = WristIKBridge.normalized_quat_xyzw(previous)
        if float(np.dot(prev_norm, current_norm)) < 0.0:
            current_norm = -current_norm
        blended = prev_norm + alpha * (current_norm - prev_norm)
        return WristIKBridge.normalized_quat_xyzw(blended)

    def joint6_axis_index(self) -> int:
        return {"x": 0, "y": 1, "z": 2}[self.joint6_axis]

    def apply_rotvec_deadband(self, rotvec: np.ndarray) -> np.ndarray:
        if self.orientation_deadband_rad <= 0.0:
            return rotvec.copy()
        abs_rotvec = np.abs(rotvec)
        reduced = np.maximum(abs_rotvec - self.orientation_deadband_rad, 0.0)
        return np.sign(rotvec) * reduced

    def relative_wrist_rotvec_local(self, wrist_rotation: np.ndarray) -> np.ndarray:
        if self.wrist_rotation_origin is None or self.hand_rotation_origin is None:
            return np.zeros(3, dtype=np.float32)
        relative_rotation_world = wrist_rotation @ self.wrist_rotation_origin.T
        relative_rotvec_world = self.rotation_matrix_to_rotvec(relative_rotation_world)
        relative_rotvec_local = self.hand_rotation_origin.T @ relative_rotvec_world
        return self.apply_rotvec_deadband(relative_rotvec_local)

    def twist_angle_about_axis(self, rotation_local: np.ndarray, axis_index: int) -> float:
        quat = self.matrix_to_quat_xyzw(rotation_local)
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
        pose_rotvec_local = pose_rotvec_local * self.hybrid_pose_rotation_scale
        return self.hand_rotation_origin @ self.rotvec_to_matrix(pose_rotvec_local)

    @staticmethod
    def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(quat)
        if norm < 1e-9:
            return np.eye(3, dtype=np.float32)
        x, y, z, w = quat / norm
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float32,
        )

    @staticmethod
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
        quat = np.array([x, y, z, w], dtype=np.float32)
        norm = float(np.linalg.norm(quat))
        if norm < 1e-9:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return quat / norm

    @staticmethod
    def rotation_matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
        cos_angle = (np.trace(rotation) - 1.0) * 0.5
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
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
        axis /= 2.0 * np.sin(angle)
        return axis * angle

    @staticmethod
    def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
        angle = float(np.linalg.norm(rotvec))
        if angle < 1e-6:
            return np.eye(3, dtype=np.float32)
        axis = rotvec / angle
        x, y, z = axis
        skew = np.array(
            [
                [0.0, -z, y],
                [z, 0.0, -x],
                [-y, x, 0.0],
            ],
            dtype=np.float32,
        )
        return (
            np.eye(3, dtype=np.float32)
            + np.sin(angle) * skew
            + (1.0 - np.cos(angle)) * (skew @ skew)
        )

    @staticmethod
    def scale_rotation_matrix(rotation: np.ndarray, scale: float) -> np.ndarray:
        if abs(scale - 1.0) < 1e-6:
            return rotation.copy()
        rotvec = WristIKBridge.rotation_matrix_to_rotvec(rotation)
        return WristIKBridge.rotvec_to_matrix(rotvec * scale)

    @staticmethod
    def normalized_quat_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
        quat = np.array(quat_xyzw, dtype=np.float32)
        norm = float(np.linalg.norm(quat))
        if norm < 1e-9:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return quat / norm

    @staticmethod
    def limit_vector(vector: np.ndarray, max_norm: float) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if max_norm > 0.0 and norm > max_norm:
            return vector / norm * max_norm
        return vector


def main():
    rclpy.init()
    node = WristIKBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
