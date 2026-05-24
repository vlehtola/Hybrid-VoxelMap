from datetime import datetime

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackagePrefix, FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("hybrid_voxel_map")
    pkg_prefix = FindPackagePrefix("hybrid_voxel_map")
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = PathJoinSubstitution([EnvironmentVariable("HOME", default_value="."), "hvm_results"])
    default_tum_result_path = PathJoinSubstitution([result_dir, f"kitti08_salsanext_semantic_{run_timestamp}_tum.txt"])
    default_efficiency_path = PathJoinSubstitution([result_dir, f"kitti08_salsanext_semantic_{run_timestamp}_efficiency.txt"])
    default_sequence = PathJoinSubstitution(
        [EnvironmentVariable("HOME", default_value="."), "datasets", "KITTI", "dataset", "sequences", "08"]
    )
    mapping_launch = PathJoinSubstitution([pkg_share, "launch", "mapping_kitti08.launch.py"])
    salsa_script = PathJoinSubstitution(
        [pkg_prefix, "lib", "hybrid_voxel_map", "salsanext_semantic_ros2_node.py"]
    )

    sequence = LaunchConfiguration("sequence")
    rate = LaunchConfiguration("rate")
    start_index = LaunchConfiguration("start_index")
    end_index = LaunchConfiguration("end_index")
    rviz = LaunchConfiguration("rviz")
    loop = LaunchConfiguration("loop")
    raw_topic = LaunchConfiguration("raw_topic")
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

    return LaunchDescription([
        DeclareLaunchArgument(
            "sequence",
            default_value=default_sequence,
        ),
        DeclareLaunchArgument("rate", default_value="10.0"),
        DeclareLaunchArgument("start_index", default_value="0"),
        DeclareLaunchArgument("end_index", default_value="-1"),
        DeclareLaunchArgument("rviz", default_value="false"),
        DeclareLaunchArgument("loop", default_value="false"),
        DeclareLaunchArgument("raw_topic", default_value="/velodyne_points_raw"),
        DeclareLaunchArgument("semantic_topic", default_value="/velodyne_points"),
        DeclareLaunchArgument("tum_result_path", default_value=default_tum_result_path),
        DeclareLaunchArgument("efficiency_path", default_value=default_efficiency_path),
        DeclareLaunchArgument(
            "salsa_python",
            default_value=EnvironmentVariable("SALSANEXT_PYTHON", default_value="python3"),
        ),
        DeclareLaunchArgument("salsa_repo", default_value=EnvironmentVariable("SALSANEXT_REPO", default_value="")),
        DeclareLaunchArgument(
            "salsa_model_dir",
            default_value=EnvironmentVariable("SALSANEXT_MODEL_DIR", default_value=""),
        ),
        DeclareLaunchArgument(
            "salsa_engine",
            default_value=EnvironmentVariable("SALSANEXT_ENGINE", default_value=""),
        ),
        DeclareLaunchArgument("salsa_gpu", default_value="0"),
        DeclareLaunchArgument("salsa_warmup_frames", default_value="5"),
        DeclareLaunchArgument("salsa_warmup_points", default_value="120000"),
        DeclareLaunchArgument("salsa_queue_depth", default_value="512"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(mapping_launch),
            launch_arguments={
                "sequence": sequence,
                "rate": rate,
                "start_index": start_index,
                "end_index": end_index,
                "rviz": rviz,
                "loop": loop,
                "gt_semantics": "false",
                "publisher_topic": raw_topic,
                "lid_topic": semantic_topic,
                "tum_result_path": tum_result_path,
                "efficiency_path": efficiency_path,
            }.items(),
        ),
        ExecuteProcess(
            cmd=[
                salsa_python,
                salsa_script,
                "--input-topic",
                raw_topic,
                "--output-topic",
                semantic_topic,
                "--queue-depth",
                salsa_queue_depth,
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
    ])
