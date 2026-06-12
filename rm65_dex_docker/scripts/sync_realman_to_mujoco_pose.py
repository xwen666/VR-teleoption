#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from realman_sdk_common import (
    DEFAULT_ARM_IP,
    DEFAULT_ARM_PORT,
    RealmanSdkClient,
    can_import_sdk,
    deg_to_rad,
    rad_to_deg,
    wait_for_joint_state_sdk,
)

ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
DEFAULT_SNAPSHOT_PATH = Path("/tmp/rm65_mujoco_arm_snapshot.json")


def parse_joint_values(text: str) -> list[float]:
    values = [chunk for chunk in text.replace(",", " ").split() if chunk]
    if len(values) != 6:
        raise ValueError(f"Expected 6 joint values, got {len(values)}")
    return [float(value) for value in values]


def load_target_joints(snapshot_path: Path, joint_text: str) -> list[float]:
    if joint_text.strip():
        return parse_joint_values(joint_text)

    if not snapshot_path.exists():
        raise FileNotFoundError(
            f"MuJoCo joint snapshot not found: {snapshot_path}. "
            "Run the MuJoCo validator first, or pass --joints explicitly."
        )

    data = json.loads(snapshot_path.read_text(encoding="utf-8"))
    raw_values = data.get("arm_qpos")
    if not isinstance(raw_values, list) or len(raw_values) < 6:
        raise ValueError(f"Snapshot file {snapshot_path} does not contain arm_qpos[6].")
    return [float(value) for value in raw_values[:6]]


def max_abs_delta(values_a: list[float], values_b: list[float]) -> float:
    return max(abs(a - b) for a, b in zip(values_a, values_b))


def l2_delta(values_a: list[float], values_b: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(values_a, values_b)))


def print_target_pose(target_joints_rad: list[float], source: str, backend: str) -> None:
    print(f"Pre-sync backend: {backend}")
    print("Target real-robot joint pose:")
    print("  rad: " + ", ".join(f"{value:.4f}" for value in target_joints_rad))
    print("  deg: " + ", ".join(f"{value:.2f}" for value in rad_to_deg(target_joints_rad)))
    print(f"Target source: {source}")


def build_movej_message(target_joints: list[float], speed: int, dof: int) -> str:
    joints_text = ", ".join(f"{value:.6f}" for value in target_joints)
    return (
        "{"
        f"joint: [{joints_text}], "
        f"speed: {int(speed)}, "
        "block: true, "
        "trajectory_connect: 0, "
        f"dof: {int(dof)}"
        "}"
    )


class RosRealmanJointStateMonitor:
    def __init__(self, node, JointState, Bool, joint_state_topic: str, move_result_topic: str):
        self.node = node
        self.latest_positions: Optional[list[float]] = None
        self.latest_time: Optional[float] = None
        self.last_move_result: Optional[bool] = None
        self.node.create_subscription(JointState, joint_state_topic, self.on_joint_state, 20)
        self.node.create_subscription(Bool, move_result_topic, self.on_move_result, 10)

    def on_joint_state(self, msg) -> None:
        state_by_name = dict(zip(msg.name, msg.position))
        if not all(name in state_by_name for name in ARM_JOINT_NAMES):
            return
        self.latest_positions = [float(state_by_name[name]) for name in ARM_JOINT_NAMES]
        self.latest_time = time.monotonic()

    def on_move_result(self, msg) -> None:
        self.last_move_result = bool(msg.data)


def wait_for_joint_state_ros(node: RosRealmanJointStateMonitor, rclpy_module, timeout_sec: float) -> list[float]:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline and rclpy_module.ok():
        rclpy_module.spin_once(node.node, timeout_sec=0.1)
        if node.latest_positions is not None:
            return list(node.latest_positions)
    raise TimeoutError("Timed out waiting for /joint_states from the real robot.")


def publish_movej_ros(movej_topic: str, target_joints: list[float], speed: int, dof: int) -> None:
    message = build_movej_message(target_joints, speed=speed, dof=dof)
    command = [
        "ros2",
        "topic",
        "pub",
        "--once",
        movej_topic,
        "rm_ros_interfaces/msg/Movej",
        message,
    ]
    subprocess.run(command, check=True)


def can_import_ros() -> bool:
    try:
        import rclpy  # noqa: F401
        from sensor_msgs.msg import JointState  # noqa: F401
        from std_msgs.msg import Bool  # noqa: F401
    except Exception:
        return False
    return True


def resolve_backend(requested: str) -> str:
    if requested in {"sdk", "ros"}:
        return requested
    if can_import_sdk():
        return "sdk"
    if can_import_ros():
        return "ros"
    raise RuntimeError(
        "Could not resolve a usable pre-sync backend. "
        "Neither RealMan Python SDK nor ROS 2 Python dependencies are importable."
    )


def run_ros_backend(args: argparse.Namespace, target_joints_rad: list[float]) -> int:
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Bool
    except Exception as exc:
        print(f"ROS backend import failed: {exc}", file=sys.stderr)
        return 5

    class MonitorNode(Node):
        def __init__(self) -> None:
            super().__init__("realman_joint_state_monitor")

    rclpy.init()
    node = MonitorNode()
    monitor = RosRealmanJointStateMonitor(
        node=node,
        JointState=JointState,
        Bool=Bool,
        joint_state_topic=args.joint_state_topic,
        move_result_topic=args.move_result_topic,
    )

    try:
        current_joints = wait_for_joint_state_ros(monitor, rclpy, args.wait_for_state_timeout)
        print("Current real-robot joint pose (rad):")
        print("  " + ", ".join(f"{value:.4f}" for value in current_joints))
        print(
            "Current -> target delta:"
            f" max_abs={max_abs_delta(current_joints, target_joints_rad):.4f} rad,"
            f" l2={l2_delta(current_joints, target_joints_rad):.4f} rad"
        )

        if (
            max_abs_delta(current_joints, target_joints_rad) <= args.max_abs_error
            and l2_delta(current_joints, target_joints_rad) <= args.l2_error
        ):
            print("Real robot is already close to the MuJoCo pose; nothing to do.")
            return 0

        publish_movej_ros(
            movej_topic=args.movej_topic,
            target_joints=target_joints_rad,
            speed=args.speed,
            dof=6,
        )
        print(
            f"Published MoveJ to {args.movej_topic} at speed={args.speed}. "
            "Waiting for the robot to settle..."
        )

        deadline = time.monotonic() + args.motion_timeout
        stable_since: Optional[float] = None
        last_log_time = 0.0
        while time.monotonic() < deadline and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if monitor.last_move_result is False:
                print("rm_driver reported MoveJ failure on /rm_driver/movej_result.", file=sys.stderr)
                return 2
            if monitor.latest_positions is None:
                continue

            current = monitor.latest_positions
            error_max = max_abs_delta(current, target_joints_rad)
            error_l2 = l2_delta(current, target_joints_rad)
            now = time.monotonic()
            if now - last_log_time > 1.0:
                last_log_time = now
                print(
                    "Sync progress:"
                    f" max_abs={error_max:.4f} rad,"
                    f" l2={error_l2:.4f} rad,"
                    f" move_result={monitor.last_move_result}"
                )

            if error_max <= args.max_abs_error and error_l2 <= args.l2_error:
                if stable_since is None:
                    stable_since = now
                if now - stable_since >= args.settle_time:
                    print("Real robot reached the MuJoCo pose and is stable.")
                    print("You can start teleop now.")
                    return 0
            else:
                stable_since = None

        print(
            "Timed out waiting for the real robot to reach the MuJoCo pose.",
            file=sys.stderr,
        )
        return 3
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def run_sdk_backend(args: argparse.Namespace, target_joints_rad: list[float]) -> int:
    if not can_import_sdk():
        print("SDK backend import failed: Robotic_Arm is not importable in this Python.", file=sys.stderr)
        return 5

    target_joints_deg = rad_to_deg(target_joints_rad)
    client = RealmanSdkClient(args.arm_ip, args.arm_port)
    client.connect()

    try:
        current_deg = wait_for_joint_state_sdk(client, args.wait_for_state_timeout)
        current_rad = deg_to_rad(current_deg)
        print("Current real-robot joint pose:")
        print("  rad: " + ", ".join(f"{value:.4f}" for value in current_rad))
        print("  deg: " + ", ".join(f"{value:.2f}" for value in current_deg))
        print(
            "Current -> target delta:"
            f" max_abs={max_abs_delta(current_rad, target_joints_rad):.4f} rad,"
            f" l2={l2_delta(current_rad, target_joints_rad):.4f} rad"
        )

        if (
            max_abs_delta(current_rad, target_joints_rad) <= args.max_abs_error
            and l2_delta(current_rad, target_joints_rad) <= args.l2_error
        ):
            print("Real robot is already close to the MuJoCo pose; nothing to do.")
            return 0

        if not args.skip_self_collision_check:
            collision_state = client.check_self_collision(target_joints_deg)
            if collision_state == 1:
                print(
                    "SDK self-collision check rejected the target joint pose. "
                    "Aborting pre-sync before sending rm_movej.",
                    file=sys.stderr,
                )
                return 4

        ret = client.movej_deg(target_joints_deg, speed=args.speed, block=0)
        if ret != 0:
            print(f"SDK rm_movej failed immediately with code {ret}.", file=sys.stderr)
            return 2
        print(
            f"Sent SDK rm_movej at speed={args.speed} to {args.arm_ip}:{args.arm_port}. "
            "Waiting for the robot to settle..."
        )

        deadline = time.monotonic() + args.motion_timeout
        stable_since: Optional[float] = None
        last_log_time = 0.0
        while time.monotonic() < deadline:
            current_deg = client.get_joint_deg()
            current_rad = deg_to_rad(current_deg)
            error_max = max_abs_delta(current_rad, target_joints_rad)
            error_l2 = l2_delta(current_rad, target_joints_rad)
            now = time.monotonic()
            if now - last_log_time > 1.0:
                last_log_time = now
                print(
                    "Sync progress:"
                    f" max_abs={error_max:.4f} rad,"
                    f" l2={error_l2:.4f} rad"
                )

            if error_max <= args.max_abs_error and error_l2 <= args.l2_error:
                if stable_since is None:
                    stable_since = now
                if now - stable_since >= args.settle_time:
                    print("Real robot reached the MuJoCo pose and is stable.")
                    print("You can start teleop now.")
                    return 0
            else:
                stable_since = None
            time.sleep(0.1)

        print(
            "Timed out waiting for the real robot to reach the MuJoCo pose.",
            file=sys.stderr,
        )
        return 3
    finally:
        client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Safely move the real RealMan arm to the latest MuJoCo arm joint pose "
            "before starting teleoperation."
        )
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "ros", "sdk"],
        default="auto",
        help="Pre-sync transport: Docker ROS topics or direct RealMan Python SDK.",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=DEFAULT_SNAPSHOT_PATH,
        help="MuJoCo arm joint snapshot JSON written by the validator.",
    )
    parser.add_argument(
        "--joints",
        type=str,
        default="",
        help="Optional explicit target joint list in radians, e.g. '0 -1.2 1.9 0 0.9 0'.",
    )
    parser.add_argument(
        "--movej-topic",
        type=str,
        default="/rm_driver/movej_cmd",
        help="MoveJ command topic exposed by rm_driver.",
    )
    parser.add_argument(
        "--move-result-topic",
        type=str,
        default="/rm_driver/movej_result",
        help="MoveJ result topic exposed by rm_driver.",
    )
    parser.add_argument(
        "--joint-state-topic",
        type=str,
        default="/joint_states",
        help="Joint state topic that reflects the real robot state.",
    )
    parser.add_argument(
        "--arm-ip",
        type=str,
        default=DEFAULT_ARM_IP,
        help="RealMan controller IP used by the direct Python SDK backend.",
    )
    parser.add_argument(
        "--arm-port",
        type=int,
        default=DEFAULT_ARM_PORT,
        help="RealMan controller TCP port used by the direct Python SDK backend.",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=5,
        help="RealMan MoveJ speed percentage, 0-100. Default keeps the pre-sync conservative.",
    )
    parser.add_argument(
        "--wait-for-state-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for the first real-robot joint state.",
    )
    parser.add_argument(
        "--motion-timeout",
        type=float,
        default=90.0,
        help="Seconds to wait for the robot to reach the target.",
    )
    parser.add_argument(
        "--max-abs-error",
        type=float,
        default=0.02,
        help="Maximum absolute joint error in radians for declaring success.",
    )
    parser.add_argument(
        "--l2-error",
        type=float,
        default=0.04,
        help="Joint-space L2 error threshold in radians for declaring success.",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=1.0,
        help="How long the robot must stay within tolerance before success.",
    )
    parser.add_argument(
        "--skip-self-collision-check",
        action="store_true",
        help="Disable the SDK-side self-collision check before rm_movej.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the target and exit without commanding the robot.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_joints = load_target_joints(args.snapshot, args.joints)
    backend = resolve_backend(args.backend)

    source = "--joints" if args.joints.strip() else f"MuJoCo snapshot {args.snapshot}"
    print_target_pose(target_joints, source=source, backend=backend)

    if args.dry_run:
        return 0

    if backend == "sdk":
        return run_sdk_backend(args, target_joints)
    return run_ros_backend(args, target_joints)


if __name__ == "__main__":
    raise SystemExit(main())
