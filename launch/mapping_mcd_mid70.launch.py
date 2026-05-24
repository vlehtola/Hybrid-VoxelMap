from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("hybrid_voxel_map")
    config_file = PathJoinSubstitution([pkg_share, "config", "MCD_mid70_pure_ndt.yaml"])
    rviz_config = PathJoinSubstitution([pkg_share, "rviz_cfg", "voxel_mapping.rviz"])

    return LaunchDescription([
        DeclareLaunchArgument("rviz", default_value="false"),
        Node(
            package="hybrid_voxel_map",
            executable="voxel_mapping_odom",
            name="voxel_mapping_odom",
            output="screen",
            parameters=[config_file],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", rviz_config],
            condition=IfCondition(LaunchConfiguration("rviz")),
        ),
    ])
