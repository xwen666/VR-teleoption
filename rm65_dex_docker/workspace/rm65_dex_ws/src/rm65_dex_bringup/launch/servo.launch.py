import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    with open(absolute_file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder(
            "rm65_b_v_dexhand", package_name="rm65_dex_moveit_config"
        )
        .planning_pipelines(default_planning_pipeline="ompl", pipelines=["ompl"])
        .to_moveit_configs()
    )
    servo_params = {
        "moveit_servo": load_yaml("rm65_dex_moveit_config", "config/servo.yaml")
    }

    return LaunchDescription(
        [
            Node(
                package="moveit_servo",
                executable="servo_node_main",
                name="servo_node",
                output="screen",
                parameters=[moveit_config.to_dict(), servo_params],
            )
        ]
    )
