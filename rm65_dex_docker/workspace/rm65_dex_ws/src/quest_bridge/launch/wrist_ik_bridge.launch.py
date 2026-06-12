from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare("quest_bridge"), "config", "wrist_cube_side_mount.yaml"]
    )
    config = LaunchConfiguration("config")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value=default_config,
                description="Path to the wrist IK bridge parameter file.",
            ),
            Node(
                package="quest_bridge",
                executable="wrist_ik_bridge",
                name="wrist_ik_bridge",
                output="screen",
                parameters=[config],
            )
        ]
    )
