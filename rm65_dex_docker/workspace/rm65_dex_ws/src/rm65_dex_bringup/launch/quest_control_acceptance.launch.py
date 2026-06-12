from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def include_launch(package_name, *path_parts, launch_arguments=None):
    launch_path = PathJoinSubstitution([FindPackageShare(package_name), *path_parts])
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_path),
        launch_arguments=(launch_arguments or {}).items(),
    )


def generate_launch_description():
    use_rviz = LaunchConfiguration("use_rviz")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_rviz",
                default_value="true",
                description="Start MoveIt RViz for visual acceptance.",
            ),
            include_launch(
                "rm65_dex_moveit_config",
                "launch",
                "demo.launch.py",
                launch_arguments={"use_rviz": use_rviz},
            ),
            include_launch("quest_bridge", "launch", "wrist_ik_bridge.launch.py"),
            include_launch(
                "dex_hand_control",
                "launch",
                "revo2_left_hand_qpos.launch.py",
            ),
        ]
    )
