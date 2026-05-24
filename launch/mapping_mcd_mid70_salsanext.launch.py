from datetime import datetime

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackagePrefix, FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("hybrid_voxel_map")
    pkg_prefix = FindPackagePrefix("hybrid_voxel_map")
    config_file = PathJoinSubstitution([pkg_share, "config", "MCD_mid70_body_salsanext_semantic_adapted.yaml"])
    rviz_config = PathJoinSubstitution([pkg_share, "rviz_cfg", "voxel_mapping.rviz"])
    salsa_script = PathJoinSubstitution(
        [pkg_prefix, "lib", "hybrid_voxel_map", "salsanext_mcd_livox_semantic_ros2_node.py"]
    )

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = PathJoinSubstitution([EnvironmentVariable("HOME", default_value="."), "hvm_results"])
    default_tum_result_path = PathJoinSubstitution(
        [result_dir, f"mcd_ntu_day_10_mid70_body_salsanext_semantic_{run_timestamp}_tum.txt"]
    )
    default_efficiency_path = PathJoinSubstitution(
        [result_dir, f"mcd_ntu_day_10_mid70_body_salsanext_semantic_{run_timestamp}_efficiency.txt"]
    )

    input_topic = LaunchConfiguration("input_topic")
    input_message_type = LaunchConfiguration("input_message_type")
    semantic_topic = LaunchConfiguration("semantic_topic")
    tum_result_path = LaunchConfiguration("tum_result_path")
    efficiency_path = LaunchConfiguration("efficiency_path")
    salsa_python = LaunchConfiguration("salsa_python")
    salsa_repo = LaunchConfiguration("salsa_repo")
    salsa_model_dir = LaunchConfiguration("salsa_model_dir")
    salsa_engine = LaunchConfiguration("salsa_engine")
    salsa_gpu = LaunchConfiguration("salsa_gpu")
    salsa_warmup_frames = LaunchConfiguration("salsa_warmup_frames")
    salsa_warmup_points = LaunchConfiguration("salsa_warmup_points")
    salsa_queue_depth = LaunchConfiguration("salsa_queue_depth")
    async_pipeline = LaunchConfiguration("async_pipeline")
    plane_labels = LaunchConfiguration("plane_labels")
    object_labels = LaunchConfiguration("object_labels")

    return LaunchDescription([
        DeclareLaunchArgument("input_topic", default_value="/livox/lidar"),
        DeclareLaunchArgument("input_message_type", default_value="livox"),
        DeclareLaunchArgument("semantic_topic", default_value="/livox/lidar_semantic_salsanext"),
        DeclareLaunchArgument("tum_result_path", default_value=default_tum_result_path),
        DeclareLaunchArgument("efficiency_path", default_value=default_efficiency_path),
        DeclareLaunchArgument(
            "salsa_python",
            default_value=EnvironmentVariable("SALSANEXT_MCD_PYTHON", default_value="python3"),
        ),
        DeclareLaunchArgument("salsa_repo", default_value=EnvironmentVariable("SALSANEXT_MCD_REPO", default_value="")),
        DeclareLaunchArgument(
            "salsa_model_dir",
            default_value=EnvironmentVariable("SALSANEXT_MCD_MODEL_DIR", default_value=""),
        ),
        DeclareLaunchArgument(
            "salsa_engine",
            default_value=EnvironmentVariable("SALSANEXT_MCD_ENGINE", default_value=""),
        ),
        DeclareLaunchArgument("salsa_gpu", default_value="0"),
        DeclareLaunchArgument("salsa_warmup_frames", default_value="5"),
        DeclareLaunchArgument("salsa_warmup_points", default_value="10000"),
        DeclareLaunchArgument("salsa_queue_depth", default_value="4096"),
        DeclareLaunchArgument("async_pipeline", default_value="false"),
        DeclareLaunchArgument("plane_labels", default_value="6,10,13,16,18"),
        DeclareLaunchArgument("object_labels", default_value="1,14,26,27,28"),
        DeclareLaunchArgument("rviz", default_value="false"),
        Node(
            package="hybrid_voxel_map",
            executable="voxel_mapping_odom",
            name="voxel_mapping_odom",
            output="screen",
            parameters=[
                config_file,
                {
                    "common.lid_topic": ParameterValue(semantic_topic, value_type=str),
                    "Result.tum_result_path": ParameterValue(tum_result_path, value_type=str),
                    "Result.efficiency_path": ParameterValue(efficiency_path, value_type=str),
                },
            ],
        ),
        ExecuteProcess(
            cmd=[
                salsa_python,
                salsa_script,
                "--input-topic",
                input_topic,
                "--input-message-type",
                input_message_type,
                "--output-topic",
                semantic_topic,
                "--queue-depth",
                salsa_queue_depth,
                "--async-pipeline",
                async_pipeline,
                "--salsanext-repo",
                salsa_repo,
                "--model-dir",
                salsa_model_dir,
                "--deploy-engine",
                salsa_engine,
                "--gpu",
                salsa_gpu,
                "--device",
                ["cuda:", salsa_gpu],
                "--warmup-frames",
                salsa_warmup_frames,
                "--warmup-points",
                salsa_warmup_points,
                "--plane-labels",
                plane_labels,
                "--object-labels",
                object_labels,
                "--efficiency-path",
                efficiency_path,
            ],
            cwd=salsa_repo,
            additional_env={
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
            output="screen",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", rviz_config],
            condition=IfCondition(LaunchConfiguration("rviz")),
        ),
    ])
