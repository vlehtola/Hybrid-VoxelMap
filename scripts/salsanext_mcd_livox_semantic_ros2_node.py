#!/usr/bin/env python3
"""Stream SalsaNext MCD predictions from Livox CustomMsg into Hybrid-VoxelMap.

The node keeps the MCD Livox points in the body frame for SLAM, transforms a
copy to the Mid70 frame for SalsaNext inference, and republishes the original
body-frame points with the Hybrid-VoxelMap semantic field:

  0 = unknown
  1 = plane
  2 = Gaussian/NDT
  3 = object
"""

from __future__ import annotations

import argparse
import atexit
import os
import queue
import resource
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from livox_ros_driver2.msg import CustomMsg
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from salsanext_semantic_ros2_node import (  # noqa: E402
    SalsaNextSemanticBackend,
    add_hybrid_semantic_field,
    make_field,
    parse_bool,
    pointcloud_to_xyzi,
)


DEFAULT_SALSANEXT_REPO = os.environ.get("SALSANEXT_MCD_REPO", "")
DEFAULT_MODEL_DIR = os.environ.get("SALSANEXT_MCD_MODEL_DIR", "")
DEFAULT_ENGINE = os.environ.get("SALSANEXT_MCD_ENGINE", "")

# MCD raw labels: curb, lanemarking, parkinglot, road, sidewalk.
DEFAULT_PLANE_LABELS = (6, 10, 13, 16, 18)
# MCD raw labels: bike, pedestrian, dynamic/other/static vehicle.
DEFAULT_OBJECT_LABELS = (1, 14, 26, 27, 28)

DEFAULT_T_MID70_VN100 = (
    0.9998581316050806,
    -0.0001711140769215,
    -0.0168430217944847,
    0.01114100505508,
    -0.0002258196228774,
    -0.999994705851753,
    -0.0032461167514233,
    -0.008870848246916,
    -0.0168423771687589,
    0.0032494597148798,
    -0.999852876848822,
    0.03720308969226,
)

_ACTIVE_NODE = None


def parse_float_tuple(text: str, expected: int) -> tuple[float, ...]:
    values = tuple(float(tok.strip()) for tok in text.split(",") if tok.strip())
    if len(values) != expected:
        raise argparse.ArgumentTypeError(f"expected {expected} comma-separated floats, got {len(values)}")
    return values


def write_value(data: np.ndarray, offset: int, values: np.ndarray, dtype: str, width: int):
    data[:, offset : offset + width] = (
        np.ascontiguousarray(values, dtype=np.dtype(dtype)).view(np.uint8).reshape(values.shape[0], width)
    )


def make_body_pointcloud(
    msg: CustomMsg,
    body_xyzi: np.ndarray,
    rel_time: np.ndarray,
    ring: np.ndarray,
    categories: np.ndarray,
    frame_id: str,
) -> PointCloud2:
    count = int(body_xyzi.shape[0])
    raw = np.zeros((count, 32), dtype=np.uint8)
    write_value(raw, 0, body_xyzi[:, 0], "<f4", 4)
    write_value(raw, 4, body_xyzi[:, 1], "<f4", 4)
    write_value(raw, 8, body_xyzi[:, 2], "<f4", 4)
    write_value(raw, 16, body_xyzi[:, 3], "<f4", 4)
    write_value(raw, 20, rel_time, "<f4", 4)
    write_value(raw, 24, ring, "<u2", 2)
    write_value(raw, 28, categories, "<u4", 4)

    out = PointCloud2()
    out.header = msg.header
    if frame_id:
        out.header.frame_id = frame_id
    out.height = 1
    out.width = count
    out.fields = [
        make_field("x", 0, PointField.FLOAT32),
        make_field("y", 4, PointField.FLOAT32),
        make_field("z", 8, PointField.FLOAT32),
        make_field("intensity", 16, PointField.FLOAT32),
        make_field("time", 20, PointField.FLOAT32),
        make_field("ring", 24, PointField.UINT16),
        make_field("semantic", 28, PointField.UINT32),
    ]
    out.is_bigendian = False
    out.point_step = 32
    out.row_step = out.point_step * out.width
    out.is_dense = True
    out.data = raw.reshape(-1).tobytes()
    return out


class SalsaNextMcdLivoxSemanticNode(Node):
    def __init__(self, args):
        super().__init__("salsanext_mcd_livox_semantic_ros2")
        self.args = args
        self.backend = SalsaNextSemanticBackend(args, self.get_logger())
        self.backend.warmup()
        self._reset_runtime_memory_stats()

        t_values = parse_float_tuple(args.body_to_mid70_matrix, 12)
        transform = np.asarray(t_values, dtype=np.float32).reshape(3, 4)
        self.body_to_mid70_r = np.ascontiguousarray(transform[:, :3], dtype=np.float32)
        self.body_to_mid70_t = np.ascontiguousarray(transform[:, 3], dtype=np.float32)

        self.input_is_pointcloud2 = str(args.input_message_type).lower() in ("pointcloud2", "pointcloud")
        pub_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=max(1, int(args.queue_depth)),
            reliability=ReliabilityPolicy.RELIABLE,
        )
        sub_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=max(1, int(args.queue_depth)),
            reliability=ReliabilityPolicy.BEST_EFFORT if self.input_is_pointcloud2 else ReliabilityPolicy.RELIABLE,
        )
        self.publisher = self.create_publisher(PointCloud2, args.output_topic, pub_qos)
        if args.wait_for_output_subscriber:
            self.get_logger().info(f"Waiting for SLAM subscriber on {args.output_topic}")
            while rclpy.ok() and self.publisher.get_subscription_count() == 0:
                rclpy.spin_once(self, timeout_sec=0.1)

        if self.input_is_pointcloud2:
            self.subscription = self.create_subscription(PointCloud2, args.input_topic, self.callback, sub_qos)
        else:
            self.subscription = self.create_subscription(CustomMsg, args.input_topic, self.callback, sub_qos)
        self.msg_queue = queue.Queue(maxsize=max(1, int(args.queue_depth)))
        self.stop_worker = threading.Event()
        self.worker = threading.Thread(target=self.worker_loop, name="salsanext_mcd_worker", daemon=True)
        self.frames = 0
        self.points = 0
        self.prepare_ms_sum = 0.0
        self.project_ms_sum = 0.0
        self.model_ms_sum = 0.0
        self.post_ms_sum = 0.0
        self.pack_ms_sum = 0.0
        self.publish_ms_sum = 0.0
        self.processing_ms_sum = 0.0
        self.efficiency_written = False
        if args.async_pipeline:
            self.worker.start()
        self.get_logger().info(
            f"SalsaNext MCD Livox semantic bridge ready: {args.input_topic} -> {args.output_topic}"
        )
        self.get_logger().info(
            "MCD SalsaNext mapping: plane raw labels {6,10,13,16,18}; "
            "object raw labels {1,14,26,27,28}."
        )

    def _reset_runtime_memory_stats(self):
        try:
            torch = self.backend.torch
            if self.backend.device.type == "cuda":
                torch.cuda.synchronize(self.backend.device)
                torch.cuda.reset_peak_memory_stats(self.backend.device)
        except Exception as exc:
            self.get_logger().warn(f"Failed to reset CUDA memory stats after warmup: {exc}")

    def livox_to_arrays(self, msg: CustomMsg):
        valid_points = []
        rel_time = []
        ring = []
        scan_line = max(1, int(self.args.scan_line))
        filter_num = max(1, int(self.args.point_filter_num))
        for i in range(1, len(msg.points)):
            point = msg.points[i]
            if int(point.line) >= scan_line:
                continue
            tag = int(point.tag) & 0x30
            if tag != 0x10 and tag != 0x00:
                continue
            if i % filter_num != 0:
                continue
            if not (np.isfinite(point.x) and np.isfinite(point.y) and np.isfinite(point.z)):
                continue
            prev = msg.points[i - 1]
            if (
                abs(point.x - prev.x) <= 1e-7
                and abs(point.y - prev.y) <= 1e-7
                and abs(point.z - prev.z) <= 1e-7
            ):
                continue
            valid_points.append((point.x, point.y, point.z, float(point.reflectivity)))
            rel_time.append(float(point.offset_time) * 1e-9)
            ring.append(int(point.line))

        if not valid_points:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.uint16),
            )

        body_xyzi = np.asarray(valid_points, dtype=np.float32)
        mid70_xyz = body_xyzi[:, :3] @ self.body_to_mid70_r.T + self.body_to_mid70_t
        salsa_xyzi = np.empty_like(body_xyzi)
        salsa_xyzi[:, :3] = mid70_xyz
        salsa_xyzi[:, 3] = body_xyzi[:, 3]
        return (
            body_xyzi,
            salsa_xyzi,
            np.asarray(rel_time, dtype=np.float32),
            np.asarray(ring, dtype=np.uint16),
        )

    def callback(self, msg: CustomMsg):
        if self.args.async_pipeline:
            # Keep the ROS callback short; the worker drains this queue in order.
            # This avoids slowing rosbag playback while preserving every queued
            # frame for replay-style experiments.
            self.msg_queue.put(msg)
        else:
            self.process_msg(msg)

    def worker_loop(self):
        while not self.stop_worker.is_set() or not self.msg_queue.empty():
            try:
                msg = self.msg_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self.process_msg(msg)
            finally:
                self.msg_queue.task_done()

    def process_msg(self, msg: CustomMsg):
        try:
            t_prepare0 = time.perf_counter()
            if self.input_is_pointcloud2:
                body_xyzi = pointcloud_to_xyzi(msg)
                mid70_xyz = body_xyzi[:, :3] @ self.body_to_mid70_r.T + self.body_to_mid70_t
                salsa_xyzi = np.empty_like(body_xyzi)
                salsa_xyzi[:, :3] = mid70_xyz
                salsa_xyzi[:, 3] = body_xyzi[:, 3]
                rel_time = None
                ring = None
            else:
                body_xyzi, salsa_xyzi, rel_time, ring = self.livox_to_arrays(msg)
            prepare_ms = (time.perf_counter() - t_prepare0) * 1000.0
            if salsa_xyzi.shape[0] == 0:
                categories = np.zeros((0,), dtype=np.uint32)
                timings = {"project_ms": 0.0, "model_ms": 0.0, "post_ms": 0.0}
            else:
                categories, timings = self.backend.predict_hybrid_categories(salsa_xyzi)
            t_pack0 = time.perf_counter()
            if self.input_is_pointcloud2:
                out = add_hybrid_semantic_field(msg, categories)
                if self.args.frame_id:
                    out.header.frame_id = self.args.frame_id
            else:
                out = make_body_pointcloud(msg, body_xyzi, rel_time, ring, categories, self.args.frame_id)
            pack_ms = (time.perf_counter() - t_pack0) * 1000.0
            t_pub0 = time.perf_counter()
            self.publisher.publish(out)
            publish_ms = (time.perf_counter() - t_pub0) * 1000.0
        except Exception as exc:
            self.get_logger().error(f"SalsaNext MCD semantic inference failed: {exc}")
            return

        project_ms = timings.get("project_ms", 0.0)
        model_ms = timings.get("model_ms", 0.0)
        post_ms = timings.get("post_ms", 0.0)
        processing_ms = prepare_ms + project_ms + model_ms + post_ms + pack_ms + publish_ms
        self.frames += 1
        self.points += int(categories.shape[0])
        self.prepare_ms_sum += prepare_ms
        self.project_ms_sum += project_ms
        self.model_ms_sum += model_ms
        self.post_ms_sum += post_ms
        self.pack_ms_sum += pack_ms
        self.publish_ms_sum += publish_ms
        self.processing_ms_sum += processing_ms

        if self.frames <= 15 or self.frames % int(self.args.log_interval) == 0:
            counts = np.bincount(categories.astype(np.int64), minlength=4)
            self.get_logger().info(
                "SalsaNext MCD frame %d: points=%d unknown=%d plane=%d gaussian=%d object=%d "
                "prepare=%.2f project=%.2f model=%.2f post=%.2f pack=%.2f publish=%.2f processing=%.2f ms"
                % (
                    self.frames,
                    categories.shape[0],
                    counts[0],
                    counts[1],
                    counts[2],
                    counts[3],
                    prepare_ms,
                    project_ms,
                    model_ms,
                    post_ms,
                    pack_ms,
                    publish_ms,
                    processing_ms,
                )
            )

    def _write_efficiency_report(self):
        if self.efficiency_written:
            return
        self.efficiency_written = True
        if not self.args.efficiency_path:
            return

        frames = max(0, int(self.frames))
        denom = float(frames) if frames > 0 else 1.0
        avg_processing = self.processing_ms_sum / denom if frames > 0 else 0.0
        process_peak_rss_mb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
        gpu_alloc_mb = 0.0
        try:
            torch = self.backend.torch
            if self.backend.device.type == "cuda":
                torch.cuda.synchronize(self.backend.device)
                gpu_alloc_mb = torch.cuda.max_memory_allocated(self.backend.device) / (1024.0 * 1024.0)
        except Exception as exc:
            self.get_logger().warn(f"Failed to read CUDA memory stats for efficiency report: {exc}")

        try:
            path = Path(self.args.efficiency_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write("[Semantic]\n")
                f.write(f"frames={frames}\n")
                f.write(f"avg_processing_ms={avg_processing:.6f}\n")
                f.write(f"fps={1000.0 / avg_processing if avg_processing > 1e-9 else 0.0:.6f}\n")
                f.write(f"process_peak_rss_mb={process_peak_rss_mb:.6f}\n")
                f.write(f"gpu_peak_allocated_mb={gpu_alloc_mb:.6f}\n\n")
            self.get_logger().info(f"SalsaNext MCD efficiency report appended to: {path}")
        except Exception as exc:
            print(f"[SalsaNext MCD] failed to write efficiency report: {exc}", flush=True)

    def destroy_node(self):
        self.stop_worker.set()
        if hasattr(self, "worker") and self.worker.is_alive():
            self.worker.join(timeout=30.0)
        self._write_efficiency_report()
        return super().destroy_node()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-topic", default="/livox/lidar")
    parser.add_argument("--input-message-type", choices=("livox", "pointcloud2"), default="livox")
    parser.add_argument("--output-topic", default="/livox/lidar_semantic_salsanext")
    parser.add_argument("--queue-depth", type=int, default=4096)
    parser.add_argument("--async-pipeline", type=parse_bool, default=False)
    parser.add_argument("--wait-for-output-subscriber", type=parse_bool, default=True)
    parser.add_argument("--frame-id", default="camera_init")
    parser.add_argument("--scan-line", type=int, default=1)
    parser.add_argument("--point-filter-num", type=int, default=1)
    parser.add_argument(
        "--body-to-mid70-matrix",
        default=",".join(f"{v:.16g}" for v in DEFAULT_T_MID70_VN100),
    )
    parser.add_argument("--salsanext-repo", default=DEFAULT_SALSANEXT_REPO)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--deploy-engine", default=DEFAULT_ENGINE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--warmup-points", type=int, default=10000)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--efficiency-path", default="")
    parser.add_argument("--plane-labels", default=",".join(str(v) for v in DEFAULT_PLANE_LABELS))
    parser.add_argument("--object-labels", default=",".join(str(v) for v in DEFAULT_OBJECT_LABELS))
    return parser


def main():
    global _ACTIVE_NODE
    args = build_parser().parse_args()
    if str(args.efficiency_path).lower() in ("none", "off"):
        args.efficiency_path = ""
    args.async_pipeline = bool(args.async_pipeline)
    rclpy.init()
    node = SalsaNextMcdLivoxSemanticNode(args)
    _ACTIVE_NODE = node
    atexit.register(lambda: _ACTIVE_NODE._write_efficiency_report() if _ACTIVE_NODE is not None else None)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._write_efficiency_report()
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except KeyboardInterrupt:
                pass
        _ACTIVE_NODE = None


if __name__ == "__main__":
    main()
