import json
import socket
from typing import Optional

import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class HandQposNode(Node):
    def __init__(self):
        super().__init__("hand_qpos_node")

        self.declare_parameter("port", 5010)
        self.declare_parameter("controller_topic", "/hand_position_controller/commands")
        self.declare_parameter("max_delta", 0.08)
        double_array_descriptor = ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE_ARRAY
        )
        self.declare_parameter("lower", None, double_array_descriptor)
        self.declare_parameter("upper", None, double_array_descriptor)

        self.port = int(self.get_parameter("port").value)
        controller_topic = str(self.get_parameter("controller_topic").value)
        self.max_delta = float(self.get_parameter("max_delta").value)
        self.lower = np.array(self.get_parameter("lower").value, dtype=np.float32)
        self.upper = np.array(self.get_parameter("upper").value, dtype=np.float32)

        self.pub = self.create_publisher(Float64MultiArray, controller_topic, 10)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", self.port))
        self.sock.setblocking(False)

        self.last_qpos: Optional[np.ndarray] = None
        self.timer = self.create_timer(0.01, self.step)

        self.get_logger().info(
            f"Listening for hand qpos UDP on 0.0.0.0:{self.port}; publishing {controller_topic}"
        )

    def step(self):
        latest_data = None
        try:
            while True:
                latest_data, _ = self.sock.recvfrom(65535)
        except BlockingIOError:
            pass

        if latest_data is None:
            return

        qpos = self.parse_qpos_packet(latest_data)
        if qpos is None:
            return

        if self.lower.size and self.upper.size:
            if self.lower.size != qpos.size or self.upper.size != qpos.size:
                self.get_logger().warning("lower/upper length does not match qpos length; skipping clip")
            else:
                qpos = np.clip(qpos, self.lower, self.upper)

        if self.last_qpos is not None:
            if self.last_qpos.size == qpos.size:
                delta = np.clip(qpos - self.last_qpos, -self.max_delta, self.max_delta)
                qpos = self.last_qpos + delta
            else:
                self.get_logger().warning("qpos length changed; resetting rate limiter")

        self.last_qpos = qpos

        out = Float64MultiArray()
        out.data = qpos.astype(float).tolist()
        self.pub.publish(out)

    def parse_qpos_packet(self, data: bytes) -> Optional[np.ndarray]:
        text = data.decode("utf-8", errors="ignore").strip()
        try:
            packet = json.loads(text)
            qpos = np.array(packet["hand_qpos"], dtype=np.float32)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            self.get_logger().warning(f"Cannot parse hand_qpos packet: {text[:80]}")
            return None

        if qpos.ndim != 1:
            self.get_logger().warning("hand_qpos must be a 1-D list")
            return None
        return qpos


def main():
    rclpy.init()
    node = HandQposNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
