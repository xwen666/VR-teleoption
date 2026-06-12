import json
import time
from pathlib import Path
from typing import Dict, List

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray, String
from trajectory_msgs.msg import JointTrajectory


class SmokeRecorder(Node):
    def __init__(self):
        super().__init__("smoke_recorder")

        self.declare_parameter("output_dir", "/data/smoke_records")
        self.declare_parameter("max_frames", 20)
        self.declare_parameter("arm_joint_names", [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ])
        self.declare_parameter("hand_joint_names", [
            "hand_thumb_metacarpal_joint",
            "hand_thumb_proximal_joint",
            "hand_index_proximal_joint",
            "hand_middle_proximal_joint",
            "hand_ring_proximal_joint",
            "hand_pinky_proximal_joint",
        ])

        self.output_dir = Path(str(self.get_parameter("output_dir").value))
        self.max_frames = int(self.get_parameter("max_frames").value)
        self.arm_joint_names = list(self.get_parameter("arm_joint_names").value)
        self.hand_joint_names = list(self.get_parameter("hand_joint_names").value)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_path = self.output_dir / f"smoke_episode_{stamp}.jsonl"

        self.latest_joint_state: Dict[str, float] = {}
        self.latest_arm_action: List[float] = []
        self.latest_hand_action: List[float] = []
        self.latest_task = ""
        self.front_image_count = 0
        self.wrist_image_count = 0
        self.frame_index = 0

        self.create_subscription(JointState, "/joint_states", self.on_joint_state, 20)
        self.create_subscription(JointTrajectory, "/arm_controller/joint_trajectory", self.on_arm_action, 10)
        self.create_subscription(Float64MultiArray, "/hand_position_controller/commands", self.on_hand_action, 10)
        self.create_subscription(String, "/task_text", self.on_task_text, 10)
        self.create_subscription(Image, "/front_camera/image_raw", self.on_front_image, 5)
        self.create_subscription(Image, "/wrist_camera/image_raw", self.on_wrist_image, 5)

        self.timer = self.create_timer(0.05, self.record_step)
        self.get_logger().info(f"Writing smoke records to {self.output_path}")

    def on_joint_state(self, msg: JointState):
        self.latest_joint_state = dict(zip(msg.name, msg.position))

    def on_arm_action(self, msg: JointTrajectory):
        if msg.points:
            self.latest_arm_action = list(msg.points[-1].positions)

    def on_hand_action(self, msg: Float64MultiArray):
        self.latest_hand_action = list(msg.data)

    def on_task_text(self, msg: String):
        self.latest_task = msg.data

    def on_front_image(self, _msg: Image):
        self.front_image_count += 1

    def on_wrist_image(self, _msg: Image):
        self.wrist_image_count += 1

    def record_step(self):
        if self.frame_index >= self.max_frames:
            self.get_logger().info("Smoke recorder reached max_frames; shutting down.")
            rclpy.shutdown()
            return

        if not self.latest_joint_state:
            return

        state = [
            self.latest_joint_state.get(name, 0.0)
            for name in [*self.arm_joint_names, *self.hand_joint_names]
        ]
        action = [
            *self.pad_or_trim(self.latest_arm_action, len(self.arm_joint_names)),
            *self.pad_or_trim(self.latest_hand_action, len(self.hand_joint_names)),
        ]
        row = {
            "timestamp": self.get_clock().now().nanoseconds / 1e9,
            "frame_index": self.frame_index,
            "task": self.latest_task,
            "observation.state": state,
            "action": action,
            "debug": {
                "front_image_count": self.front_image_count,
                "wrist_image_count": self.wrist_image_count,
            },
        }
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        self.frame_index += 1

    @staticmethod
    def pad_or_trim(values: List[float], size: int) -> List[float]:
        values = list(values[:size])
        return values + [0.0] * (size - len(values))


def main():
    rclpy.init()
    node = SmokeRecorder()
    rclpy.spin(node)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
