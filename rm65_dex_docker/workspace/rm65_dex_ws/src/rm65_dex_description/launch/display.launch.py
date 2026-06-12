from launch import LaunchDescription
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    xacro_path = PathJoinSubstitution(
        [
            FindPackageShare("rm65_dex_description"),
            "urdf",
            "rm65_b_v_dexhand.urdf.xacro",
        ]
    )
    rviz_config_path = PathJoinSubstitution(
        [
            FindPackageShare("rm65_dex_description"),
            "rviz",
            "display.rviz",
        ]
    )
    robot_description = {"robot_description": Command(["xacro ", xacro_path])}

    return LaunchDescription(
        [
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[robot_description],
                output="screen",
            ),
            Node(
                package="joint_state_publisher_gui",
                executable="joint_state_publisher_gui",
                output="screen",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", rviz_config_path],
                output="screen",
            ),
        ]
    )
