#!/usr/bin/env python3

import sys
import os
import math

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from xrobotoolkit_teleop.hardware.interface.arm import my_Arm


UDP_IP = "192.168.1.104"
UDP_PORT = 8089
UDP_CYCLE = 100  # ms

GRIPPPER_DEFAULT_SPEED = 0.05  # m/s

is_wifi = False
class ArmInterfaceNode(Node):
    def __init__(self):
        super().__init__('arm_interface_node')
        self.get_logger().info("Connecting to RML...")

        self.arm = my_Arm(is_wifi)

        self.last_joint_rad = [self.arm.get_joint_degree()[i] * math.pi / 180.0 for i in range(6)]
        self.last_joint_rad_ns = self.get_clock().now().nanoseconds
        
        # try:
        #     self.arm.start_udp(host_ip=UDP_IP, udp_port=UDP_PORT, rate_cycle=UDP_CYCLE)
        #     self.get_logger().info(f"RML UDP push enabled: host={UDP_IP}, port={UDP_PORT}, rate={UDP_CYCLE}Hz")
        # except Exception as e:
        #     self.get_logger().error(f"Enable UDP realtime push failed: {e}")

        self.get_logger().info("RML connected.")

        # Collection control: disabled by default, wait for start command
        self.enabled = False
        self.cmd_sub = self.create_subscription(
            String, '/collect_cmd', self.cmd_callback,
            QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST, depth=1)
        )

        # Initialize ROS 2 publisher
        self.state_pub = self.create_publisher(JointState, '/robot_state', 10)
        
        # Create timer to publish state every 0.1 seconds
        self.create_timer(0.1, self.publish_state)

        self._last_log_ns = 0

    def cmd_callback(self, msg: String):
        cmd = (msg.data or "").strip().lower()
        if cmd == "start":
            if not self.enabled:
                self.enabled = True
                self.get_logger().info("Collect START -> enable state publishing")
        elif cmd == "stop":
            if self.enabled:
                self.enabled = False
                self.get_logger().info("Collect STOP -> disable state publishing")
        else:
            self.get_logger().warn(f"Unknown collect cmd: {msg.data!r} (use 'start'/'stop')")


    def publish_state(self):
        if not self.enabled:
            return
        # Get joint angles
        joint_angles, ee_pose = self.arm.get_joints_angles_and_pose()
        now_ns = self.get_clock().now().nanoseconds
        joint_speeds = [(joint_angles[i] - self.last_joint_rad[i]) * 1_000_000_000 / (now_ns - self.last_joint_rad_ns) for i in range(6)]
        
        self.last_joint_rad = joint_angles
        self.last_joint_rad_ns = now_ns

        # Get gripper position
        gripper_state = self.arm.get_gripper_state()[1]
        gripper_pos = gripper_state['actpos'] / 1000.0
        if gripper_pos < 0.0:
            gripper_pos = 0.0
        elif gripper_pos > 1.0:
            gripper_pos = 1.0

        if gripper_state['mode'] == 4 or gripper_state['mode'] == 5:
            gripper_speed = GRIPPPER_DEFAULT_SPEED
        else:   
            gripper_speed = 0.0

        # Combine complete state (6 joint angles + 1 gripper position)
        full_state = joint_angles + [gripper_pos] + ee_pose + joint_speeds + [gripper_speed]
        # full_state = joint_angles + [gripper_pos] + ee_pose

        # Publish message
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.name = [
            "joint1","joint2","joint3","joint4","joint5","joint6","gripper",
            "ee_x","ee_y","ee_z","ee_rx","ee_ry","ee_rz",
            "joint1_speed","joint2_speed","joint3_speed","joint4_speed","joint5_speed","joint6_speed","gripper_speed",
        ]
        msg.position = full_state
        self.state_pub.publish(msg)
        # Print once per 5 seconds
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self._last_log_ns >= 5_000_000_000:
            self.get_logger().info(f"Published robot qpose: {msg.position[:7]}")
            self.get_logger().info(f"Published robot ee pose: {msg.position[7:13]}")
            self.get_logger().info(f"Published robot joint speeds: {msg.position[13:20]}")
            # self.get_logger().info(f"==============================================")
            self._last_log_ns = now_ns

    # def publish_state(self):
    #     if not self.enabled:
    #         return
    #     # Get joint angles
    #     joint_angles, ee_pose = self.arm._rt_joint_rad, self.arm._rt_pose
    #     joint_speeds = self.arm._rt_joint_vel_rad

    #     # Get gripper position
    #     # gripper_pos = self.arm.get_gripper_position()
    #     gripper_state = self.arm.get_gripper_state()[1]
    #     gripper_pos = gripper_state['actpos'] / 1000.0
    #     if gripper_pos < 0.0:
    #         gripper_pos = 0.0
    #     elif gripper_pos > 1.0:
    #         gripper_pos = 1.0

    #     if gripper_state['mode'] == 4 or gripper_state['mode'] == 5:
    #         gripper_speed = GRIPPPER_DEFAULT_SPEED
    #     else:   
    #         gripper_speed = 0.0

    #     # Combine complete state (6 joint angles + 1 gripper position)
    #     full_state = joint_angles + [gripper_pos] + ee_pose + joint_speeds + [gripper_speed]

    #     # Publish message
    #     msg = JointState()
    #     msg.header.stamp = self.get_clock().now().to_msg()
    #     msg.header.frame_id = "base_link"
    #     msg.name = [
    #         "joint1","joint2","joint3","joint4","joint5","joint6","gripper",
    #         "ee_x","ee_y","ee_z","ee_rx","ee_ry","ee_rz",
    #         "joint1_speed","joint2_speed","joint3_speed","joint4_speed","joint5_speed","joint6_speed","gripper_speed",
    #     ]
    #     msg.position = full_state
    #     self.state_pub.publish(msg)
    #     # Print once per 5 seconds
    #     now_ns = self.get_clock().now().nanoseconds
    #     if now_ns - self._last_log_ns >= 5_000_000_000:
    #         self.get_logger().info(f"Published robot qpose: {msg.position[:7]}")
    #         self.get_logger().info(f"Published robot ee pose: {msg.position[7:13]}")
    #         self.get_logger().info(f"Published robot joint speeds: {msg.position[13:20]}")
    #         self.get_logger().info(f"==============================================")
    #         self._last_log_ns = now_ns


def main(args=None):
    rclpy.init(args=args)
    arm_node = None

    try:
        arm_node = ArmInterfaceNode()
        rclpy.spin(arm_node)
    except KeyboardInterrupt:
        print("[robot_pub] Node interrupted by user")
    except Exception as e:
        print(f"[robot_pub] Unexpected error: {e}")
    finally:
        if arm_node:
            try:
                arm_node.destroy_node()
            except Exception:
                pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()
