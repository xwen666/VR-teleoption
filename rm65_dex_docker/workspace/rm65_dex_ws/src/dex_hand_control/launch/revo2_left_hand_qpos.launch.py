from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = PathJoinSubstitution(
        [FindPackageShare("dex_hand_control"), "config", "revo2_left_qpos.yaml"]
    )
    return LaunchDescription(
        [
            Node(
                package="dex_hand_control",
                executable="hand_qpos_node",
                name="hand_qpos_node",
                output="screen",
                parameters=[config],
            )
        ]
    )
