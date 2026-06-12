from launch import LaunchDescription
from launch_ros.actions import Node


def spawner(controller_name):
    return Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            controller_name,
            "--controller-manager",
            "/controller_manager",
            "--controller-manager-timeout",
            "60",
            "--switch-timeout",
            "60",
            "--service-call-timeout",
            "60",
        ],
        output="screen",
    )


def generate_launch_description():
    return LaunchDescription(
        [
            spawner("joint_state_broadcaster"),
            spawner("arm_controller"),
            spawner("hand_position_controller"),
        ]
    )
