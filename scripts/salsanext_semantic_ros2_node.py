#!/usr/bin/env python3
"""Stream online SalsaNext predictions into Hybrid-VoxelMap.

The node subscribes to KITTI-style PointCloud2, runs SalsaNext TensorRT
inference for every frame, then republishes the same cloud with a Hybrid-
VoxelMap `semantic` UINT32 field:

  0 = unknown
  1 = plane
  2 = Gaussian/NDT
  3 = object
"""

from __future__ import annotations

import argparse
import atexit
import os
import resource
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
import yaml


DEFAULT_SALSANEXT_REPO = os.environ.get("SALSANEXT_REPO", "")
DEFAULT_MODEL_DIR = os.environ.get("SALSANEXT_MODEL_DIR", "")
DEFAULT_ENGINE = os.environ.get("SALSANEXT_ENGINE", "")
DEFAULT_PLANE_LABELS = (40, 44, 48, 49, 50, 51, 52, 60, 72)
DEFAULT_OBJECT_LABELS = (
    10, 11, 13, 15, 16, 18, 20, 30, 31, 32,
    252, 253, 254, 255, 256, 257, 258, 259,
)
_ACTIVE_NODE = None


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {value}")


def parse_int_set(text: str | Iterable[int]) -> set[int]:
    if isinstance(text, str):
        if not text.strip():
            return set()
        return {int(tok.strip()) for tok in text.split(",") if tok.strip()}
    return {int(v) for v in text}


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def field_offset(msg: PointCloud2, name: str, datatype: int | None = None) -> int:
    for field in msg.fields:
        if field.name == name and (datatype is None or field.datatype == datatype):
            return int(field.offset)
    return -1


def strided_field(msg: PointCloud2, offset: int, dtype: np.dtype) -> np.ndarray:
    count = int(msg.width) * int(msg.height)
    if offset < 0 or offset + np.dtype(dtype).itemsize > int(msg.point_step):
        return np.zeros((count,), dtype=dtype)
    return np.ndarray(
        shape=(count,),
        dtype=dtype,
        buffer=memoryview(msg.data),
        offset=offset,
        strides=(int(msg.point_step),),
    )


def make_field(name: str, offset: int, datatype: int) -> PointField:
    field = PointField()
    field.name = name
    field.offset = offset
    field.datatype = datatype
    field.count = 1
    return field


def pointcloud_to_xyzi(msg: PointCloud2) -> np.ndarray:
    x_off = field_offset(msg, "x", PointField.FLOAT32)
    y_off = field_offset(msg, "y", PointField.FLOAT32)
    z_off = field_offset(msg, "z", PointField.FLOAT32)
    i_off = field_offset(msg, "intensity", PointField.FLOAT32)
    if x_off < 0 or y_off < 0 or z_off < 0:
        raise ValueError("input PointCloud2 must contain float32 x/y/z fields")

    count = int(msg.width) * int(msg.height)
    xyzi = np.empty((count, 4), dtype=np.float32)
    xyzi[:, 0] = strided_field(msg, x_off, np.dtype("<f4"))
    xyzi[:, 1] = strided_field(msg, y_off, np.dtype("<f4"))
    xyzi[:, 2] = strided_field(msg, z_off, np.dtype("<f4"))
    if i_off >= 0:
        xyzi[:, 3] = strided_field(msg, i_off, np.dtype("<f4"))
    else:
        xyzi[:, 3] = 0.0
    return xyzi


def add_hybrid_semantic_field(msg: PointCloud2, categories: np.ndarray) -> PointCloud2:
    count = int(msg.width) * int(msg.height)
    if categories.shape[0] != count:
        raise ValueError(f"semantic size mismatch: categories={categories.shape[0]} points={count}")

    semantic_off = field_offset(msg, "semantic", PointField.UINT32)
    if semantic_off < 0:
        semantic_off = 28

    if (not msg.is_bigendian) and semantic_off + 4 <= int(msg.point_step):
        try:
            data_view = memoryview(msg.data)
            if data_view.nbytes >= count * int(msg.point_step):
                writable = not data_view.readonly
                out = msg if writable else PointCloud2()
                if writable:
                    data_buffer = msg.data
                else:
                    data_buffer = bytearray(msg.data)
                    out.header = msg.header
                    out.height = msg.height
                    out.width = msg.width
                    out.fields = list(msg.fields)
                    out.is_bigendian = msg.is_bigendian
                    out.point_step = msg.point_step
                    out.row_step = msg.row_step
                    out.is_dense = msg.is_dense

                semantic = np.ndarray(
                    shape=(count,),
                    dtype=np.dtype("<u4"),
                    buffer=data_buffer,
                    offset=semantic_off,
                    strides=(int(msg.point_step),),
                )
                semantic[:] = np.ascontiguousarray(categories, dtype=np.dtype("<u4"))

                if field_offset(out, "semantic", PointField.UINT32) < 0:
                    out.fields = list(out.fields) + [make_field("semantic", semantic_off, PointField.UINT32)]
                if not writable:
                    out.data = bytes(data_buffer)
                return out
        except (BufferError, TypeError, ValueError):
            pass

    x = strided_field(msg, field_offset(msg, "x", PointField.FLOAT32), np.dtype("<f4"))
    y = strided_field(msg, field_offset(msg, "y", PointField.FLOAT32), np.dtype("<f4"))
    z = strided_field(msg, field_offset(msg, "z", PointField.FLOAT32), np.dtype("<f4"))
    intensity = strided_field(msg, field_offset(msg, "intensity", PointField.FLOAT32), np.dtype("<f4"))
    rel_time = strided_field(msg, field_offset(msg, "time", PointField.FLOAT32), np.dtype("<f4"))
    ring = strided_field(msg, field_offset(msg, "ring", PointField.UINT16), np.dtype("<u2"))

    def bytes_view(values: np.ndarray, dtype: str, width: int) -> np.ndarray:
        return np.ascontiguousarray(values, dtype=np.dtype(dtype)).view(np.uint8).reshape(count, width)

    raw = np.zeros((count * 32,), dtype=np.uint8)
    data = raw.reshape((count, 32))
    data[:, 0:4] = bytes_view(x, "<f4", 4)
    data[:, 4:8] = bytes_view(y, "<f4", 4)
    data[:, 8:12] = bytes_view(z, "<f4", 4)
    data[:, 16:20] = bytes_view(intensity, "<f4", 4)
    data[:, 20:24] = bytes_view(rel_time, "<f4", 4)
    data[:, 24:26] = bytes_view(ring, "<u2", 2)
    data[:, 28:32] = bytes_view(categories, "<u4", 4)

    out = PointCloud2()
    out.header = msg.header
    out.height = msg.height
    out.width = msg.width
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
    out.is_dense = msg.is_dense
    out.data = raw.tobytes()
    return out


class SalsaNextSemanticBackend:
    def __init__(self, args, logger=None):
        self.args = args
        self.logger = logger
        self.repo = Path(args.salsanext_repo).resolve()
        self.train_dir = self.repo / "train"
        if not args.salsanext_repo or not self.train_dir.is_dir():
            raise FileNotFoundError(
                "SalsaNext repository is not configured. Pass --salsanext-repo "
                "or set the SALSANEXT_REPO / SALSANEXT_MCD_REPO environment variable."
            )
        if not args.model_dir:
            raise FileNotFoundError(
                "SalsaNext model directory is not configured. Pass --model-dir "
                "or set SALSANEXT_MODEL_DIR / SALSANEXT_MCD_MODEL_DIR."
            )
        if not args.deploy_engine:
            raise FileNotFoundError(
                "SalsaNext TensorRT engine is not configured. Pass --deploy-engine "
                "or set SALSANEXT_ENGINE / SALSANEXT_MCD_ENGINE."
            )
        if str(self.train_dir) not in sys.path:
            sys.path.insert(0, str(self.train_dir))

        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

        import torch
        from common.laserscan import LaserScan
        from tasks.semantic.modules.SalsaNext import SalsaNext
        from tasks.semantic.modules.trt_runtime import TensorRTEngineModule
        from tasks.semantic.postproc.KNN import KNN

        self.torch = torch
        self.LaserScan = LaserScan
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        self.device = torch.device(args.device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available for SalsaNext inference")
            torch.cuda.set_device(args.gpu)

        self.arch = load_yaml(str(Path(args.model_dir) / "arch_cfg.yaml"))
        self.data = load_yaml(str(Path(args.model_dir) / "data_cfg.yaml"))
        sensor = self.arch["dataset"]["sensor"]
        img_prop = sensor["img_prop"]
        self.img_h = int(img_prop["height"])
        self.img_w = int(img_prop["width"])
        self.fov_up = float(sensor["fov_up"])
        self.fov_down = float(sensor["fov_down"])
        self.img_means = torch.tensor(sensor["img_means"], dtype=torch.float32, device=self.device)[:, None, None]
        self.img_stds = torch.tensor(sensor["img_stds"], dtype=torch.float32, device=self.device)[:, None, None]
        self.nclasses = len(self.data["learning_map_inv"])

        engine_path = Path(args.deploy_engine)
        if not engine_path.exists():
            raise FileNotFoundError(f"SalsaNext TensorRT engine not found: {engine_path}")
        self.model = TensorRTEngineModule(str(engine_path)).to(self.device).eval()
        self.post = None
        if self.arch.get("post", {}).get("KNN", {}).get("use", False):
            self.post = KNN(self.arch["post"]["KNN"]["params"], self.nclasses).to(self.device).eval()

        self.learning_map_inv = {int(k): int(v) for k, v in self.data["learning_map_inv"].items()}
        max_learning_id = max(self.learning_map_inv.keys())
        self.learning_to_original = torch.zeros((max_learning_id + 1,), dtype=torch.long, device=self.device)
        for learning_id, original_id in self.learning_map_inv.items():
            self.learning_to_original[learning_id] = int(original_id)

        self.plane_labels = parse_int_set(args.plane_labels)
        self.object_labels = parse_int_set(args.object_labels)
        lut_size = max(max(self.plane_labels | self.object_labels | {0}) + 1, 260)
        category_lut = torch.full((lut_size,), 2, dtype=torch.uint8, device=self.device)
        category_lut[0] = 0
        for label in self.plane_labels:
            if 0 <= label < lut_size:
                category_lut[label] = 1
        for label in self.object_labels:
            if 0 <= label < lut_size:
                category_lut[label] = 3
        self.category_lut = category_lut

        self._log(f"[SalsaNext] engine={engine_path}")
        self._log(f"[SalsaNext] plane_labels={sorted(self.plane_labels)} object_labels={sorted(self.object_labels)}")

    def _log(self, text: str):
        if self.logger is not None:
            self.logger.info(text)
        else:
            print(text, flush=True)

    def project_frame(self, xyzi: np.ndarray):
        scan = self.LaserScan(
            project=True,
            H=self.img_h,
            W=self.img_w,
            fov_up=self.fov_up,
            fov_down=self.fov_down,
        )
        scan.set_points(
            np.ascontiguousarray(xyzi[:, :3], dtype=np.float32),
            np.ascontiguousarray(xyzi[:, 3], dtype=np.float32),
        )
        torch = self.torch
        proj_range = torch.from_numpy(scan.proj_range).to(self.device, non_blocking=True)
        proj_xyz = torch.from_numpy(scan.proj_xyz).to(self.device, non_blocking=True)
        proj_remission = torch.from_numpy(scan.proj_remission).to(self.device, non_blocking=True)
        proj_mask = torch.from_numpy(scan.proj_mask.astype(np.float32)).to(self.device, non_blocking=True)
        proj_x = torch.from_numpy(scan.proj_x.astype(np.int64, copy=False)).to(self.device, non_blocking=True)
        proj_y = torch.from_numpy(scan.proj_y.astype(np.int64, copy=False)).to(self.device, non_blocking=True)
        unproj_range = torch.from_numpy(scan.unproj_range.astype(np.float32, copy=False)).to(self.device, non_blocking=True)

        proj = torch.cat([
            proj_range.unsqueeze(0),
            proj_xyz.permute(2, 0, 1),
            proj_remission.unsqueeze(0),
        ], dim=0)
        proj = (proj - self.img_means) / self.img_stds
        proj = proj * proj_mask
        return proj.unsqueeze(0), proj_range, unproj_range, proj_x, proj_y

    def predict_hybrid_categories(self, xyzi: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        torch = self.torch
        count = int(xyzi.shape[0])
        timings = {}
        t0 = time.perf_counter()
        proj_in, proj_range, unproj_range, proj_x, proj_y = self.project_frame(xyzi)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        timings["project_ms"] = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(proj_in)
            proj_argmax = logits[0].argmax(dim=0)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        timings["model_ms"] = (time.perf_counter() - t1) * 1000.0

        t2 = time.perf_counter()
        if self.post is not None:
            pred_learning = self.post(proj_range, unproj_range, proj_argmax, proj_x, proj_y)
        else:
            pred_learning = proj_argmax[proj_y, proj_x]
        original = self.learning_to_original[pred_learning.clamp_(0, self.learning_to_original.shape[0] - 1)]
        original = original.clamp_(0, self.category_lut.shape[0] - 1)
        categories = self.category_lut[original].detach().to("cpu").numpy().astype(np.uint32, copy=False)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        timings["post_ms"] = (time.perf_counter() - t2) * 1000.0
        if categories.shape[0] != count:
            raise RuntimeError(f"SalsaNext output size mismatch: {categories.shape[0]} vs {count}")
        return categories, timings

    def warmup(self):
        if self.args.warmup_frames <= 0:
            return
        rng = np.random.default_rng(0)
        points = max(1, int(self.args.warmup_points))
        self._log(f"[SalsaNext] warmup start: frames={self.args.warmup_frames} points={points}")
        for i in range(int(self.args.warmup_frames)):
            xyz = np.empty((points, 3), dtype=np.float32)
            xyz[:, 0] = rng.uniform(1.0, 80.0, size=points)
            xyz[:, 1] = rng.uniform(-40.0, 40.0, size=points)
            xyz[:, 2] = rng.uniform(-3.0, 2.0, size=points)
            intensity = rng.random((points, 1), dtype=np.float32)
            xyzi = np.concatenate([xyz, intensity], axis=1)
            t0 = time.perf_counter()
            _ = self.predict_hybrid_categories(xyzi)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self._log(f"[SalsaNext] warmup {i + 1}/{self.args.warmup_frames}: {dt_ms:.2f} ms")
        self._log("[SalsaNext] warmup finished")


class SalsaNextSemanticNode(Node):
    def __init__(self, args):
        super().__init__("salsanext_semantic_ros2")
        self.args = args
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=max(1, int(args.queue_depth)),
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.publisher = self.create_publisher(PointCloud2, args.output_topic, qos)
        self.backend = SalsaNextSemanticBackend(args, self.get_logger())
        self.backend.warmup()
        self._reset_runtime_memory_stats()

        self.frames = 0
        self.points = 0
        self.project_ms_sum = 0.0
        self.model_ms_sum = 0.0
        self.post_ms_sum = 0.0
        self.pack_ms_sum = 0.0
        self.processing_ms_sum = 0.0
        self.efficiency_written = False
        self.subscription = self.create_subscription(PointCloud2, args.input_topic, self.callback, qos)
        self.get_logger().info(f"SalsaNext semantic bridge ready: {args.input_topic} -> {args.output_topic}")
        if args.efficiency_path:
            self.get_logger().info(f"SalsaNext efficiency report target: {args.efficiency_path}")

    def _reset_runtime_memory_stats(self):
        try:
            torch = self.backend.torch
            if self.backend.device.type == "cuda":
                torch.cuda.synchronize(self.backend.device)
                torch.cuda.reset_peak_memory_stats(self.backend.device)
        except Exception as exc:
            self.get_logger().warn(f"Failed to reset CUDA memory stats after warmup: {exc}")

    def callback(self, msg: PointCloud2):
        try:
            t_parse0 = time.perf_counter()
            xyzi = pointcloud_to_xyzi(msg)
            parse_ms = (time.perf_counter() - t_parse0) * 1000.0
            categories, timings = self.backend.predict_hybrid_categories(xyzi)
            t_pack0 = time.perf_counter()
            out = add_hybrid_semantic_field(msg, categories)
            pack_ms = (time.perf_counter() - t_pack0) * 1000.0
            self.publisher.publish(out)
        except Exception as exc:
            self.get_logger().error(f"SalsaNext semantic inference failed: {exc}")
            return

        project_ms = timings.get("project_ms", 0.0)
        model_ms = timings.get("model_ms", 0.0)
        post_ms = timings.get("post_ms", 0.0)
        processing_ms = parse_ms + project_ms + model_ms + post_ms
        self.frames += 1
        self.points += int(categories.shape[0])
        self.project_ms_sum += project_ms
        self.model_ms_sum += model_ms
        self.post_ms_sum += post_ms
        self.pack_ms_sum += pack_ms
        self.processing_ms_sum += processing_ms

        if self.frames <= 5 or self.frames % int(self.args.log_interval) == 0:
            counts = np.bincount(categories.astype(np.int64), minlength=4)
            self.get_logger().info(
                "SalsaNext semantic frame %d: points=%d unknown=%d plane=%d gaussian=%d object=%d "
                "project=%.2f model=%.2f post=%.2f pack=%.2f processing=%.2f ms"
                % (
                    self.frames,
                    categories.shape[0],
                    counts[0],
                    counts[1],
                    counts[2],
                    counts[3],
                    project_ms,
                    model_ms,
                    post_ms,
                    pack_ms,
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

        gpu_alloc_mb = 0.0
        process_peak_rss_mb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
        try:
            torch = self.backend.torch
            if self.backend.device.type == "cuda":
                torch.cuda.synchronize(self.backend.device)
                gpu_alloc_mb = torch.cuda.max_memory_allocated(self.backend.device) / (1024.0 * 1024.0)
        except Exception as exc:
            self.get_logger().warn(f"Failed to read CUDA memory stats for efficiency report: {exc}")

        try:
            path = Path(self.args.efficiency_path)
            if path.parent:
                path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write("[Semantic]\n")
                f.write(f"frames={frames}\n")
                f.write(f"avg_processing_ms={avg_processing:.6f}\n")
                f.write(f"fps={1000.0 / avg_processing if avg_processing > 1e-9 else 0.0:.6f}\n")
                f.write(f"process_peak_rss_mb={process_peak_rss_mb:.6f}\n")
                f.write(f"gpu_peak_allocated_mb={gpu_alloc_mb:.6f}\n\n")
            self.get_logger().info(f"SalsaNext efficiency report appended to: {path}")
        except Exception as exc:
            print(f"[SalsaNext] failed to write efficiency report: {exc}", flush=True)

    def destroy_node(self):
        self._write_efficiency_report()
        return super().destroy_node()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-topic", default="/velodyne_points_raw")
    parser.add_argument("--output-topic", default="/velodyne_points")
    parser.add_argument("--queue-depth", type=int, default=512)
    parser.add_argument("--salsanext-repo", default=DEFAULT_SALSANEXT_REPO)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--deploy-engine", default=DEFAULT_ENGINE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--warmup-points", type=int, default=120000)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--efficiency-path", default="")
    parser.add_argument("--plane-labels", default=",".join(str(v) for v in DEFAULT_PLANE_LABELS))
    parser.add_argument("--object-labels", default=",".join(str(v) for v in DEFAULT_OBJECT_LABELS))
    parser.add_argument("--once-bin", default="")
    return parser


def run_once_bin(args):
    backend = SalsaNextSemanticBackend(args)
    backend.warmup()
    arr = np.fromfile(args.once_bin, dtype=np.float32).reshape((-1, 4))
    t0 = time.perf_counter()
    categories, timings = backend.predict_hybrid_categories(arr)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    counts = np.bincount(categories.astype(np.int64), minlength=4)
    print(
        f"once-bin points={categories.shape[0]} unknown={counts[0]} plane={counts[1]} "
        f"gaussian={counts[2]} object={counts[3]} latency_ms={dt_ms:.2f} "
        f"project={timings.get('project_ms', 0.0):.2f} model={timings.get('model_ms', 0.0):.2f} "
        f"post={timings.get('post_ms', 0.0):.2f}",
        flush=True,
    )


def main():
    global _ACTIVE_NODE
    args = build_parser().parse_args()
    if args.once_bin:
        run_once_bin(args)
        return

    rclpy.init()
    node = SalsaNextSemanticNode(args)
    _ACTIVE_NODE = node
    atexit.register(lambda: _ACTIVE_NODE._write_efficiency_report() if _ACTIVE_NODE is not None else None)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._write_efficiency_report()
        except KeyboardInterrupt:
            node._write_efficiency_report()
        try:
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
