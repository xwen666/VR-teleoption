import json
import socket
import time
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from rclpy.duration import Duration
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, ConnectivityException, ExtrapolationException, LookupException
from tf2_ros import TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class WristTwistBridge(Node):
    def __init__(self):
        super().__init__("wrist_twist_bridge")

        self.declare_parameter("port", 5005)
        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("ee_frame", "rm65_tool0")
        self.declare_parameter("control_mode", "incremental")
        self.declare_parameter("axis_mapping", "quest3_teleop")
        self.declare_parameter("linear_scale", 1.0)
        self.declare_parameter("max_linear", 0.25)
        self.declare_parameter("angular_scale", 1.0)
        self.declare_parameter("max_angular", 0.8)
        self.declare_parameter("deadband", 0.003)
        self.declare_parameter("position_scale", 1.0)
        self.declare_parameter("rotation_scale", 1.0)
        self.declare_parameter("follow_gain", 2.5)
        self.declare_parameter("angular_follow_gain", 3.0)
        self.declare_parameter("packet_timeout", 0.25)
        self.declare_parameter("max_position_offset", 0.35)
        self.declare_parameter("max_rotation_error", 0.8)
        self.declare_parameter("joint6_scale", 1.0)
        self.declare_parameter("joint6_axis", "z")
        self.declare_parameter("joint6_min", -6.28)
        self.declare_parameter("joint6_max", 6.28)
        self.declare_parameter("max_joint6_step", 0.08)
        self.declare_parameter("joint6_trajectory_time", 0.08)
        self.declare_parameter("debug_log_period", 1.0)
        self.declare_parameter("debug_log_file", "")

        self.port = int(self.get_parameter("port").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.ee_frame = str(self.get_parameter("ee_frame").value)
        self.control_mode = str(self.get_parameter("control_mode").value)
        self.axis_mapping = str(self.get_parameter("axis_mapping").value)
        self.linear_scale = float(self.get_parameter("linear_scale").value)
        self.max_linear = float(self.get_parameter("max_linear").value)
        self.angular_scale = float(self.get_parameter("angular_scale").value)
        self.max_angular = float(self.get_parameter("max_angular").value)
        self.deadband = float(self.get_parameter("deadband").value)
        self.position_scale = float(self.get_parameter("position_scale").value)
        self.rotation_scale = float(self.get_parameter("rotation_scale").value)
        self.follow_gain = float(self.get_parameter("follow_gain").value)
        self.angular_follow_gain = float(self.get_parameter("angular_follow_gain").value)
        self.packet_timeout = float(self.get_parameter("packet_timeout").value)
        self.max_position_offset = float(self.get_parameter("max_position_offset").value)
        self.max_rotation_error = float(self.get_parameter("max_rotation_error").value)
        self.joint6_scale = float(self.get_parameter("joint6_scale").value)
        self.joint6_axis = str(self.get_parameter("joint6_axis").value)
        self.joint6_min = float(self.get_parameter("joint6_min").value)
        self.joint6_max = float(self.get_parameter("joint6_max").value)
        self.max_joint6_step = float(self.get_parameter("max_joint6_step").value)
        self.joint6_trajectory_time = float(self.get_parameter("joint6_trajectory_time").value)
        self.debug_log_period = float(self.get_parameter("debug_log_period").value)
        self.debug_log_file = str(self.get_parameter("debug_log_file").value)
        self.quest_to_robot = self.make_quest_to_robot_matrix(self.axis_mapping)

        if self.control_mode not in ("incremental", "joint6", "frame", "position", "velocity"):
            raise ValueError("control_mode must be 'incremental', 'joint6', 'frame', 'position', or 'velocity'")
        if self.joint6_axis not in ("x", "y", "z"):
            raise ValueError("joint6_axis must be 'x', 'y', or 'z'")

        self.pub = self.create_publisher(
            TwistStamped,
            "/servo_node/delta_twist_cmds",
            10,
        )
        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            "/arm_controller/joint_trajectory",
            10,
        )
        self.joint_state_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.on_joint_state,
            10,
        )

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", self.port))
        self.sock.setblocking(False)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.latest_wrist: Optional[np.ndarray] = None
        self.latest_wrist_time: Optional[float] = None
        self.wrist_origin: Optional[np.ndarray] = None
        self.wrist_rotation_origin: Optional[np.ndarray] = None
        self.ee_origin: Optional[np.ndarray] = None
        self.ee_rotation_origin: Optional[np.ndarray] = None
        self.prev_wrist_position: Optional[np.ndarray] = None
        self.prev_wrist_rotation: Optional[np.ndarray] = None
        self.target_position: Optional[np.ndarray] = None
        self.target_rotation: Optional[np.ndarray] = None
        self.last_pos: Optional[np.ndarray] = None
        self.last_t: Optional[float] = None
        self.last_tf_warning_time = 0.0
        self.last_debug_log_time = 0.0
        self.packet_count = 0
        self.twist_count = 0
        self.last_udp_source = None
        self.last_published_linear: Optional[np.ndarray] = None
        self.last_published_angular: Optional[np.ndarray] = None
        self.arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.current_joint_positions: Optional[np.ndarray] = None
        self.target_joint_positions: Optional[np.ndarray] = None
        self.last_joint6_delta = 0.0
        self.timer = self.create_timer(0.01, self.step)

        self.get_logger().info(
            f"Listening for Quest wrist UDP on 0.0.0.0:{self.port}; "
            f"mode={self.control_mode}, axis_mapping={self.axis_mapping}, "
            f"frame={self.frame_id}, ee_frame={self.ee_frame}"
        )
        self.write_debug_file(
            f"start wrist bridge listen=0.0.0.0:{self.port}, mode={self.control_mode}, "
            f"axis_mapping={self.axis_mapping}, frame={self.frame_id}, ee_frame={self.ee_frame}"
        )

    def on_joint_state(self, msg: JointState):
        positions = []
        for joint_name in self.arm_joint_names:
            try:
                index = msg.name.index(joint_name)
            except ValueError:
                return
            positions.append(msg.position[index])
        self.current_joint_positions = np.array(positions, dtype=np.float32)

    def step(self):
        latest_data = None
        try:
            while True:
                latest_data, self.last_udp_source = self.sock.recvfrom(65535)
        except BlockingIOError:
            pass

        if latest_data is not None:
            wrist = self.parse_wrist_packet(latest_data)
            if wrist is not None:
                self.latest_wrist = wrist
                self.latest_wrist_time = time.time()
                self.packet_count += 1

        if self.latest_wrist is None:
            self.log_wrist_io_debug("ROS UDP input: waiting for wrist_pose")
            return
        if (
            self.latest_wrist_time is not None
            and time.time() - self.latest_wrist_time > self.packet_timeout
        ):
            self.reset_tracking()
            self.log_wrist_io_debug("ROS UDP input stale: holding output until a fresh wrist_pose arrives")
            return

        if self.control_mode == "incremental":
            self.publish_incremental_follow(self.latest_wrist)
        elif self.control_mode == "joint6":
            self.publish_joint6_follow(self.latest_wrist)
        elif self.control_mode == "frame":
            self.publish_frame_follow(self.latest_wrist)
        elif self.control_mode == "position":
            self.publish_position_follow(self.latest_wrist)
        else:
            if latest_data is None:
                return
            self.publish_velocity_follow(self.latest_wrist)

    def reset_tracking(self):
        self.wrist_origin = None
        self.wrist_rotation_origin = None
        self.ee_origin = None
        self.ee_rotation_origin = None
        self.prev_wrist_position = None
        self.prev_wrist_rotation = None
        self.target_position = None
        self.target_rotation = None
        self.target_joint_positions = None
        self.last_pos = None
        self.last_t = None

    def publish_velocity_follow(self, wrist: np.ndarray):
        pos = wrist[:3]
        now = time.time()

        if self.last_pos is None:
            self.last_pos = pos
            self.last_t = now
            return

        dt = max(now - self.last_t, 1e-3)
        velocity = (pos - self.last_pos) / dt
        robot_velocity = self.map_quest_to_robot_velocity(velocity)

        speed = np.linalg.norm(robot_velocity)
        if speed < self.deadband:
            robot_velocity[:] = 0.0
        elif speed > self.max_linear:
            robot_velocity = robot_velocity / speed * self.max_linear

        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = self.frame_id
        cmd.twist.linear.x = float(robot_velocity[0] * self.linear_scale)
        cmd.twist.linear.y = float(robot_velocity[1] * self.linear_scale)
        cmd.twist.linear.z = float(robot_velocity[2] * self.linear_scale)
        cmd.twist.angular.x = 0.0
        cmd.twist.angular.y = 0.0
        cmd.twist.angular.z = 0.0
        self.pub.publish(cmd)

        self.last_pos = pos
        self.last_t = now

    def publish_joint6_follow(self, wrist: np.ndarray):
        if wrist.shape[0] < 7:
            self.get_logger().warning("joint6 follow needs wrist_pose [x, y, z, qx, qy, qz, qw]")
            return
        if self.current_joint_positions is None:
            self.log_wrist_io_debug("ROS UDP input received, ROS output blocked: waiting for /joint_states")
            return

        ee_pose = self.lookup_ee_pose()
        if ee_pose is None:
            self.log_wrist_io_debug("ROS UDP input received, ROS joint6 output blocked: waiting for TF")
            return
        _, ee_rotation = ee_pose

        wrist_position = self.map_quest_to_robot_vector(wrist[:3])
        wrist_rotation = self.quest_rotation_to_robot(wrist[3:7])

        if self.prev_wrist_rotation is None or self.target_joint_positions is None:
            self.prev_wrist_position = wrist_position.copy()
            self.prev_wrist_rotation = wrist_rotation.copy()
            self.target_joint_positions = self.current_joint_positions.copy()
            self.get_logger().info(
                "Calibrated wrist joint6 follow origin. Wrist rotation will control joint6 only."
            )
            self.log_wrist_io_debug("ROS UDP input received, ROS joint6 output not sent yet: calibrated origin")
            return

        delta_rotation_world = self.rotation_matrix_to_rotvec(
            wrist_rotation @ self.prev_wrist_rotation.T
        )
        delta_rotation_local = ee_rotation.T @ delta_rotation_world
        axis_index = {"x": 0, "y": 1, "z": 2}[self.joint6_axis]
        joint6_delta = float(delta_rotation_local[axis_index] * self.joint6_scale)
        joint6_delta = float(np.clip(joint6_delta, -self.max_joint6_step, self.max_joint6_step))

        self.prev_wrist_position = wrist_position.copy()
        self.prev_wrist_rotation = wrist_rotation.copy()

        self.target_joint_positions[5] = float(
            np.clip(
                self.target_joint_positions[5] + joint6_delta,
                self.joint6_min,
                self.joint6_max,
            )
        )
        command_positions = self.current_joint_positions.copy()
        command_positions[5] = self.target_joint_positions[5]
        self.last_joint6_delta = joint6_delta

        self.publish_joint_trajectory(command_positions)
        self.log_joint6_follow_debug(
            wrist_position,
            delta_rotation_world,
            delta_rotation_local,
            joint6_delta,
            command_positions,
        )

    def publish_joint_trajectory(self, positions: np.ndarray):
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = list(self.arm_joint_names)

        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in positions]
        point.time_from_start = DurationMsg(
            sec=int(self.joint6_trajectory_time),
            nanosec=int((self.joint6_trajectory_time % 1.0) * 1e9),
        )
        msg.points.append(point)
        self.trajectory_pub.publish(msg)

    def publish_incremental_follow(self, wrist: np.ndarray):
        """Quest3-Teleoperation style: integrate wrist pose deltas into a persistent EE target."""
        ee_pose = self.lookup_ee_pose()
        if ee_pose is None:
            self.log_wrist_io_debug("ROS UDP input received, ROS output blocked: waiting for TF")
            return
        if wrist.shape[0] < 7:
            self.get_logger().warning("Incremental follow needs wrist_pose [x, y, z, qx, qy, qz, qw]")
            return

        ee_position, ee_rotation = ee_pose
        wrist_position = self.map_quest_to_robot_vector(wrist[:3])
        wrist_rotation = self.quest_rotation_to_robot(wrist[3:7])

        if (
            self.prev_wrist_position is None
            or self.prev_wrist_rotation is None
            or self.target_position is None
            or self.target_rotation is None
        ):
            self.prev_wrist_position = wrist_position.copy()
            self.prev_wrist_rotation = wrist_rotation.copy()
            self.target_position = ee_position.copy()
            self.target_rotation = ee_rotation.copy()
            self.ee_origin = ee_position.copy()
            self.ee_rotation_origin = ee_rotation.copy()
            self.get_logger().info(
                "Calibrated incremental wrist follow origin. Current wrist motion will be "
                f"integrated onto {self.ee_frame} like Quest3-Teleoperation controller deltas."
            )
            self.log_wrist_io_debug("ROS UDP input received, ROS output not sent yet: calibrated incremental origin")
            return

        delta_position_world = (wrist_position - self.prev_wrist_position) * self.position_scale
        delta_rotation_world = self.rotation_matrix_to_rotvec(
            wrist_rotation @ self.prev_wrist_rotation.T
        ) * self.rotation_scale

        self.prev_wrist_position = wrist_position.copy()
        self.prev_wrist_rotation = wrist_rotation.copy()

        # Match Quest3-Teleoperation's incremental local-frame update:
        # convert the XR delta through the current actual EE frame, then compose it onto
        # the persistent target EE frame.
        delta_position_local_current = ee_rotation.T @ delta_position_world
        delta_rotation_local_current = ee_rotation.T @ delta_rotation_world
        target_to_current = self.target_rotation.T @ ee_rotation
        delta_position_local_target = target_to_current @ delta_position_local_current
        delta_rotation_local_target = target_to_current @ delta_rotation_local_current

        target_rotation_before = self.target_rotation.copy()
        self.target_position = self.target_position + target_rotation_before @ delta_position_local_target
        self.target_rotation = target_rotation_before @ self.rotvec_to_matrix(delta_rotation_local_target)

        position_clipped = self.limit_incremental_target()

        linear_error = self.target_position - ee_position
        angular_error = self.rotation_matrix_to_rotvec(self.target_rotation @ ee_rotation.T)
        angular_error = self.limit_vector(angular_error, self.max_rotation_error)

        linear_velocity = self.limit_vector(
            linear_error * self.follow_gain,
            self.max_linear,
            zero_norm_below=self.deadband,
            reference_norm=np.linalg.norm(linear_error),
        )
        angular_velocity = self.limit_vector(
            angular_error * self.angular_follow_gain,
            self.max_angular,
        )

        self.publish_twist(linear_velocity, angular_velocity)
        self.log_incremental_follow_debug(
            wrist_position,
            ee_position,
            self.target_position,
            delta_position_world,
            delta_rotation_world,
            linear_error,
            linear_velocity,
            angular_error,
            angular_velocity,
            position_clipped,
        )

    def limit_incremental_target(self) -> bool:
        if self.ee_origin is None or self.ee_rotation_origin is None:
            return False

        position_clipped = False
        if self.max_position_offset > 0.0 and self.target_position is not None:
            target_offset = self.target_position - self.ee_origin
            limited_offset = self.limit_vector(target_offset, self.max_position_offset)
            position_clipped = np.linalg.norm(limited_offset - target_offset) > 1e-6
            self.target_position = self.ee_origin + limited_offset

        if self.max_rotation_error > 0.0 and self.target_rotation is not None:
            target_rotvec = self.rotation_matrix_to_rotvec(
                self.target_rotation @ self.ee_rotation_origin.T
            )
            limited_rotvec = self.limit_vector(target_rotvec, self.max_rotation_error)
            self.target_rotation = self.rotvec_to_matrix(limited_rotvec) @ self.ee_rotation_origin

        return position_clipped

    def publish_frame_follow(self, wrist: np.ndarray):
        ee_pose = self.lookup_ee_pose()
        if ee_pose is None:
            self.log_wrist_io_debug("ROS UDP input received, ROS output blocked: waiting for TF")
            return
        if wrist.shape[0] < 7:
            self.get_logger().warning("Frame follow needs wrist_pose [x, y, z, qx, qy, qz, qw]")
            return

        ee_position, ee_rotation = ee_pose
        wrist_position = wrist[:3]
        wrist_rotation = self.quest_rotation_to_robot(wrist[3:7])

        if (
            self.wrist_origin is None
            or self.wrist_rotation_origin is None
            or self.ee_origin is None
            or self.ee_rotation_origin is None
        ):
            self.wrist_origin = wrist_position.copy()
            self.wrist_rotation_origin = wrist_rotation.copy()
            self.ee_origin = ee_position.copy()
            self.ee_rotation_origin = ee_rotation.copy()
            self.get_logger().info(
                "Calibrated wrist frame follow origin. The current Quest wrist frame is now "
                f"aligned to {self.ee_frame}."
            )
            self.log_wrist_io_debug("ROS UDP input received, ROS output not sent yet: calibrated origin")
            return

        wrist_offset = wrist_position - self.wrist_origin
        robot_offset = self.position_scale * self.map_quest_to_robot_vector(wrist_offset)
        target_offset = self.limit_vector(robot_offset, self.max_position_offset)
        position_clipped = np.linalg.norm(target_offset - robot_offset) > 1e-6
        target_position = self.ee_origin + target_offset

        wrist_delta_rotation = wrist_rotation @ self.wrist_rotation_origin.T
        target_rotation = self.slerp_from_identity(
            wrist_delta_rotation,
            self.rotation_scale,
        ) @ self.ee_rotation_origin

        linear_error = target_position - ee_position
        angular_error = self.rotation_matrix_to_rotvec(target_rotation @ ee_rotation.T)
        angular_error = self.limit_vector(angular_error, self.max_rotation_error)

        linear_velocity = linear_error * self.follow_gain
        angular_velocity = angular_error * self.angular_follow_gain

        linear_velocity = self.limit_vector(
            linear_velocity,
            self.max_linear,
            zero_norm_below=self.deadband,
            reference_norm=np.linalg.norm(linear_error),
        )
        angular_velocity = self.limit_vector(angular_velocity, self.max_angular)

        self.publish_twist(linear_velocity, angular_velocity)
        self.log_frame_follow_debug(
            wrist_position,
            ee_position,
            target_position,
            linear_error,
            linear_velocity,
            angular_error,
            angular_velocity,
            position_clipped,
        )

    def publish_position_follow(self, wrist: np.ndarray):
        ee_pose = self.lookup_ee_pose()
        if ee_pose is None:
            return
        ee_position, _ = ee_pose

        wrist_position = wrist[:3]
        if self.wrist_origin is None or self.ee_origin is None:
            self.wrist_origin = wrist_position.copy()
            self.ee_origin = ee_position.copy()
            self.get_logger().info(
                "Calibrated wrist follow origin. Keep the Quest hand near the desired start pose "
                "when launching this node."
            )
            return

        wrist_offset = wrist_position - self.wrist_origin
        target_position = (
            self.ee_origin
            + self.position_scale * self.map_quest_to_robot_vector(wrist_offset)
        )
        error = target_position - ee_position
        robot_velocity = error * self.follow_gain

        robot_velocity = self.limit_vector(
            robot_velocity,
            self.max_linear,
            zero_norm_below=self.deadband,
            reference_norm=np.linalg.norm(error),
        )

        self.publish_twist(robot_velocity)
        self.log_position_follow_debug(wrist_position, ee_position, target_position, error, robot_velocity)

    def lookup_ee_pose(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.frame_id,
                self.ee_frame,
                Time(),
                timeout=Duration(seconds=0.01),
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as exc:
            now = time.time()
            if now - self.last_tf_warning_time > 1.0:
                self.get_logger().warning(
                    f"Waiting for TF {self.frame_id} <- {self.ee_frame}: {exc}"
                )
                self.last_tf_warning_time = now
            return None

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        position = np.array([translation.x, translation.y, translation.z], dtype=np.float32)
        rotation_matrix = self.quat_xyzw_to_matrix(
            np.array([rotation.x, rotation.y, rotation.z, rotation.w], dtype=np.float32)
        )
        return position, rotation_matrix

    def publish_twist(
        self,
        robot_velocity: np.ndarray,
        angular_velocity: Optional[np.ndarray] = None,
    ):
        if angular_velocity is None:
            angular_velocity = np.zeros(3, dtype=np.float32)

        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = self.frame_id
        cmd.twist.linear.x = float(robot_velocity[0] * self.linear_scale)
        cmd.twist.linear.y = float(robot_velocity[1] * self.linear_scale)
        cmd.twist.linear.z = float(robot_velocity[2] * self.linear_scale)
        cmd.twist.angular.x = float(angular_velocity[0] * self.angular_scale)
        cmd.twist.angular.y = float(angular_velocity[1] * self.angular_scale)
        cmd.twist.angular.z = float(angular_velocity[2] * self.angular_scale)
        self.pub.publish(cmd)
        self.twist_count += 1
        self.last_published_linear = robot_velocity.copy()
        self.last_published_angular = angular_velocity.copy()

    def log_wrist_io_debug(self, status: str):
        if self.debug_log_period <= 0.0:
            return
        now = time.time()
        if now - self.last_debug_log_time < self.debug_log_period:
            return
        self.last_debug_log_time = now
        wrist_text = (
            "None"
            if self.latest_wrist is None
            else np.round(self.latest_wrist, 4).tolist()
        )
        linear_text = (
            "None"
            if self.last_published_linear is None
            else np.round(self.last_published_linear, 4).tolist()
        )
        angular_text = (
            "None"
            if self.last_published_angular is None
            else np.round(self.last_published_angular, 4).tolist()
        )
        self.get_logger().info(
            f"{status}: source={self.last_udp_source}, "
            f"packets={self.packet_count}, twists={self.twist_count}, "
            f"wrist_pose={wrist_text}, "
            f"last_twist_linear={linear_text}, last_twist_angular={angular_text}"
        )
        self.write_debug_file(
            f"{status}: source={self.last_udp_source}, "
            f"packets={self.packet_count}, twists={self.twist_count}, "
            f"wrist_pose={wrist_text}, "
            f"last_twist_linear={linear_text}, last_twist_angular={angular_text}"
        )

    def log_frame_follow_debug(
        self,
        wrist_position: np.ndarray,
        ee_position: np.ndarray,
        target_position: np.ndarray,
        linear_error: np.ndarray,
        linear_velocity: np.ndarray,
        angular_error: np.ndarray,
        angular_velocity: np.ndarray,
        position_clipped: bool,
    ):
        if self.debug_log_period <= 0.0:
            return
        now = time.time()
        if now - self.last_debug_log_time < self.debug_log_period:
            return
        self.last_debug_log_time = now
        message = (
            "wrist frame follow: "
            f"source={self.last_udp_source}, packets={self.packet_count}, twists={self.twist_count}, "
            f"wrist={np.round(wrist_position, 3).tolist()}, "
            f"ee={np.round(ee_position, 3).tolist()}, "
            f"target={np.round(target_position, 3).tolist()}, "
            f"position_clipped={position_clipped}, "
            f"pos_error={float(np.linalg.norm(linear_error)):.4f}, "
            f"rot_error={float(np.linalg.norm(angular_error)):.4f}, "
            f"linear_cmd={np.round(linear_velocity, 3).tolist()}, "
            f"angular_cmd={np.round(angular_velocity, 3).tolist()}"
        )
        self.get_logger().info(message)
        self.write_debug_file(message)

    def log_position_follow_debug(
        self,
        wrist_position: np.ndarray,
        ee_position: np.ndarray,
        target_position: np.ndarray,
        error: np.ndarray,
        robot_velocity: np.ndarray,
    ):
        if self.debug_log_period <= 0.0:
            return
        now = time.time()
        if now - self.last_debug_log_time < self.debug_log_period:
            return
        self.last_debug_log_time = now
        message = (
            "wrist follow: "
            f"packets={self.packet_count}, twists={self.twist_count}, "
            f"wrist={np.round(wrist_position, 3).tolist()}, "
            f"ee={np.round(ee_position, 3).tolist()}, "
            f"target={np.round(target_position, 3).tolist()}, "
            f"error_norm={float(np.linalg.norm(error)):.4f}, "
            f"cmd={np.round(robot_velocity, 3).tolist()}"
        )
        self.get_logger().info(message)
        self.write_debug_file(message)

    def log_incremental_follow_debug(
        self,
        wrist_position: np.ndarray,
        ee_position: np.ndarray,
        target_position: np.ndarray,
        delta_position_world: np.ndarray,
        delta_rotation_world: np.ndarray,
        linear_error: np.ndarray,
        linear_velocity: np.ndarray,
        angular_error: np.ndarray,
        angular_velocity: np.ndarray,
        position_clipped: bool,
    ):
        if self.debug_log_period <= 0.0:
            return
        now = time.time()
        if now - self.last_debug_log_time < self.debug_log_period:
            return
        self.last_debug_log_time = now
        message = (
            "wrist incremental follow: "
            f"source={self.last_udp_source}, packets={self.packet_count}, twists={self.twist_count}, "
            f"wrist_robot={np.round(wrist_position, 3).tolist()}, "
            f"ee={np.round(ee_position, 3).tolist()}, "
            f"target={np.round(target_position, 3).tolist()}, "
            f"delta_pos={np.round(delta_position_world, 4).tolist()}, "
            f"delta_rot={np.round(delta_rotation_world, 4).tolist()}, "
            f"position_clipped={position_clipped}, "
            f"pos_error={float(np.linalg.norm(linear_error)):.4f}, "
            f"rot_error={float(np.linalg.norm(angular_error)):.4f}, "
            f"linear_cmd={np.round(linear_velocity, 3).tolist()}, "
            f"angular_cmd={np.round(angular_velocity, 3).tolist()}"
        )
        self.get_logger().info(message)
        self.write_debug_file(message)

    def log_joint6_follow_debug(
        self,
        wrist_position: np.ndarray,
        delta_rotation_world: np.ndarray,
        delta_rotation_local: np.ndarray,
        joint6_delta: float,
        command_positions: np.ndarray,
    ):
        if self.debug_log_period <= 0.0:
            return
        now = time.time()
        if now - self.last_debug_log_time < self.debug_log_period:
            return
        self.last_debug_log_time = now
        current_joint6 = (
            None
            if self.current_joint_positions is None
            else float(self.current_joint_positions[5])
        )
        message = (
            "wrist joint6 follow: "
            f"source={self.last_udp_source}, packets={self.packet_count}, "
            f"wrist_robot={np.round(wrist_position, 3).tolist()}, "
            f"joint6_axis={self.joint6_axis}, "
            f"delta_rot_world={np.round(delta_rotation_world, 4).tolist()}, "
            f"delta_rot_local={np.round(delta_rotation_local, 4).tolist()}, "
            f"joint6_delta={joint6_delta:.4f}, "
            f"current_joint6={current_joint6}, "
            f"target_joint6={float(command_positions[5]):.4f}, "
            f"command={np.round(command_positions, 3).tolist()}"
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
            file.write(f"[{timestamp}] [ROS2] {message}\n")

    def parse_wrist_packet(self, data: bytes) -> Optional[np.ndarray]:
        text = data.decode("utf-8", errors="ignore").strip()
        if not text:
            return None

        try:
            packet = json.loads(text)
            wrist = np.array(packet["wrist_pose"], dtype=np.float32)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            try:
                values = [float(v.strip()) for v in text.split(":", 1)[1].split(",") if v.strip()]
                wrist = np.array(values, dtype=np.float32)
            except (IndexError, ValueError):
                self.get_logger().warning(f"Cannot parse wrist packet: {text[:80]}")
                return None

        if wrist.shape[0] < 3:
            self.get_logger().warning("Wrist packet must contain at least x,y,z")
            return None
        return wrist

    @staticmethod
    def make_quest_to_robot_matrix(axis_mapping: str) -> np.ndarray:
        if axis_mapping == "quest3_teleop":
            # Same convention used by Quest3-Teleoperation's R_HEADSET_TO_WORLD.
            return np.array(
                [
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            )
        if axis_mapping == "quest3_teleop_flip_forward":
            # Same as quest3_teleop, but flips Quest forward/backward. Some OpenXR
            # hand streams report "forward from the user" as negative Quest z.
            return np.array(
                [
                    [0.0, 0.0, -1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            )
        if axis_mapping == "quest3_teleop_flip_forward_up_y_left_z":
            # Same forward/back convention as quest3_teleop_flip_forward, but hand up
            # maps to robot base +y and hand left/right maps to robot base +/-z.
            return np.array(
                [
                    [0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            )
        if axis_mapping == "base_y_left":
            # Earlier bridge convention: Quest x-right maps to robot y-left.
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

    def map_quest_to_robot_velocity(self, velocity: np.ndarray) -> np.ndarray:
        return self.map_quest_to_robot_vector(velocity)

    def quest_rotation_to_robot(self, quat_xyzw: np.ndarray) -> np.ndarray:
        quest_rotation = WristTwistBridge.quat_xyzw_to_matrix(quat_xyzw)
        return self.quest_to_robot @ quest_rotation @ self.quest_to_robot.T

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
    def slerp_from_identity(rotation: np.ndarray, scale: float) -> np.ndarray:
        if abs(scale - 1.0) < 1e-6:
            return rotation
        rotvec = WristTwistBridge.rotation_matrix_to_rotvec(rotation)
        return WristTwistBridge.rotvec_to_matrix(rotvec * scale)

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
    def limit_vector(
        vector: np.ndarray,
        max_norm: float,
        zero_norm_below: float = 0.0,
        reference_norm: Optional[float] = None,
    ) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if reference_norm is None:
            reference_norm = norm
        if reference_norm < zero_norm_below:
            return np.zeros_like(vector)
        if max_norm > 0.0 and norm > max_norm:
            return vector / norm * max_norm
        return vector


def main():
    rclpy.init()
    node = WristTwistBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
