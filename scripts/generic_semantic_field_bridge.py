#!/usr/bin/env python3
"""Generic semantic-label adapter for Hybrid-VoxelMap.

This node pairs a raw PointCloud2 frame with per-point semantic labels from an
external segmentation method, maps dataset labels to Hybrid-VoxelMap categories,
and republishes the original cloud with a UINT32 `semantic` field.

Hybrid-VoxelMap categories:
  0 = unknown / unused
  1 = planar region
  2 = non-planar Gaussian region
  3 = moving / ignored object region
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Int32MultiArray, UInt32MultiArray


def parse_int_set(text: str) -> set[int]:
    if not text.strip():
        return set()
    return {int(tok.strip()) for tok in text.split(",") if tok.strip()}


def stamp_to_ns(msg) -> int:
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


def field_offset(msg: PointCloud2, name: str, datatype: int | None = None) -> int:
    for field in msg.fields:
        if field.name == name and (datatype is None or field.datatype == datatype):
            return int(field.offset)
    return -1


def make_field(name: str, offset: int, datatype: int) -> PointField:
    field = PointField()
    field.name = name
    field.offset = offset
    field.datatype = datatype
    field.count = 1
    return field


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


def extract_pointcloud_labels(msg: PointCloud2, field_name: str) -> np.ndarray:
    candidates = [
        (PointField.UINT32, np.dtype("<u4")),
        (PointField.INT32, np.dtype("<i4")),
        (PointField.UINT16, np.dtype("<u2")),
        (PointField.INT16, np.dtype("<i2")),
        (PointField.UINT8, np.dtype("<u1")),
        (PointField.INT8, np.dtype("<i1")),
        (PointField.FLOAT32, np.dtype("<f4")),
    ]
    for datatype, dtype in candidates:
        offset = field_offset(msg, field_name, datatype)
        if offset >= 0:
            return strided_field(msg, offset, dtype).astype(np.int32, copy=True)

    names = ", ".join(field.name for field in msg.fields)
    raise ValueError(f"label field '{field_name}' not found. Available fields: {names}")


def add_hybrid_semantic_field(msg: PointCloud2, categories: np.ndarray) -> PointCloud2:
    count = int(msg.width) * int(msg.height)
    if categories.shape[0] != count:
        raise ValueError(f"semantic size mismatch: categories={categories.shape[0]} points={count}")

    semantic_offset = field_offset(msg, "semantic", PointField.UINT32)
    if semantic_offset < 0:
        semantic_offset = 28

    if (not msg.is_bigendian) and semantic_offset + 4 <= int(msg.point_step):
        try:
            data_buffer = bytearray(msg.data)
            semantic = np.ndarray(
                shape=(count,),
                dtype=np.dtype("<u4"),
                buffer=data_buffer,
                offset=semantic_offset,
                strides=(int(msg.point_step),),
            )
            semantic[:] = np.ascontiguousarray(categories, dtype=np.dtype("<u4"))

            out = PointCloud2()
            out.header = msg.header
            out.height = msg.height
            out.width = msg.width
            out.fields = list(msg.fields)
            if field_offset(out, "semantic", PointField.UINT32) < 0:
                out.fields.append(make_field("semantic", semantic_offset, PointField.UINT32))
            out.is_bigendian = msg.is_bigendian
            out.point_step = msg.point_step
            out.row_step = msg.row_step
            out.is_dense = msg.is_dense
            out.data = bytes(data_buffer)
            return out
        except (BufferError, TypeError, ValueError):
            pass

    # Fallback for compact PointCloud2 layouts without free room at byte 28.
    x = strided_field(msg, field_offset(msg, "x", PointField.FLOAT32), np.dtype("<f4"))
    y = strided_field(msg, field_offset(msg, "y", PointField.FLOAT32), np.dtype("<f4"))
    z = strided_field(msg, field_offset(msg, "z", PointField.FLOAT32), np.dtype("<f4"))
    intensity = strided_field(msg, field_offset(msg, "intensity", PointField.FLOAT32), np.dtype("<f4"))
    rel_time = strided_field(msg, field_offset(msg, "time", PointField.FLOAT32), np.dtype("<f4"))
    ring = strided_field(msg, field_offset(msg, "ring", PointField.UINT16), np.dtype("<u2"))

    raw = np.zeros((count * 32,), dtype=np.uint8)
    data = raw.reshape((count, 32))
    data[:, 0:4] = np.ascontiguousarray(x, dtype="<f4").view(np.uint8).reshape(count, 4)
    data[:, 4:8] = np.ascontiguousarray(y, dtype="<f4").view(np.uint8).reshape(count, 4)
    data[:, 8:12] = np.ascontiguousarray(z, dtype="<f4").view(np.uint8).reshape(count, 4)
    data[:, 16:20] = np.ascontiguousarray(intensity, dtype="<f4").view(np.uint8).reshape(count, 4)
    data[:, 20:24] = np.ascontiguousarray(rel_time, dtype="<f4").view(np.uint8).reshape(count, 4)
    data[:, 24:26] = np.ascontiguousarray(ring, dtype="<u2").view(np.uint8).reshape(count, 2)
    data[:, 28:32] = np.ascontiguousarray(categories, dtype="<u4").view(np.uint8).reshape(count, 4)

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


@dataclass
class LabelFrame:
    labels: np.ndarray
    stamp_ns: int | None


class GenericSemanticFieldBridge(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("generic_semantic_field_bridge")
        self.args = args
        self.plane_labels = parse_int_set(args.plane_labels)
        self.object_labels = parse_int_set(args.object_labels)
        self.ignore_labels = parse_int_set(args.ignore_labels)
        self.raw_queue: Deque[PointCloud2] = deque()
        self.label_queue: Deque[LabelFrame] = deque()
        self.frame_count = 0

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=args.queue_depth,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.pub = self.create_publisher(PointCloud2, args.output_topic, qos)
        self.create_subscription(PointCloud2, args.raw_topic, self.on_cloud, qos)

        if args.label_message_type == "pointcloud2":
            self.create_subscription(PointCloud2, args.label_topic, self.on_label_cloud, qos)
        elif args.label_message_type == "uint32_multi_array":
            self.create_subscription(UInt32MultiArray, args.label_topic, self.on_label_array, qos)
        elif args.label_message_type == "int32_multi_array":
            self.create_subscription(Int32MultiArray, args.label_topic, self.on_label_array, qos)
        else:
            raise ValueError(f"unsupported label message type: {args.label_message_type}")

        if args.label_message_type != "pointcloud2" and args.sync_policy == "timestamp":
            self.get_logger().warning("MultiArray labels have no header; falling back to sequential sync.")
            self.args.sync_policy = "sequential"

        self.get_logger().info(
            f"Generic semantic bridge: raw={args.raw_topic} "
            f"labels={args.label_topic} ({args.label_message_type}) -> {args.output_topic}, "
            f"plane={{{args.plane_labels}}}, object={{{args.object_labels}}}"
        )

    def labels_to_hybrid(self, labels: np.ndarray) -> np.ndarray:
        labels = labels.astype(np.int32, copy=False)
        if self.args.input_label_space == "hybrid":
            valid = (labels >= 0) & (labels <= 3)
            return np.where(valid, labels, 0).astype(np.uint32)

        categories = np.full(labels.shape, int(self.args.default_semantic), dtype=np.uint32)
        if self.ignore_labels:
            categories[np.isin(labels, list(self.ignore_labels))] = 0
        if self.plane_labels:
            categories[np.isin(labels, list(self.plane_labels))] = 1
        if self.object_labels:
            categories[np.isin(labels, list(self.object_labels))] = 3
        categories[labels < 0] = 0
        return categories

    def on_cloud(self, msg: PointCloud2) -> None:
        self.raw_queue.append(msg)
        self.trim_queues()
        self.try_publish()

    def on_label_cloud(self, msg: PointCloud2) -> None:
        try:
            labels = extract_pointcloud_labels(msg, self.args.label_field)
        except ValueError as exc:
            self.get_logger().error(str(exc))
            return
        self.label_queue.append(LabelFrame(labels=labels, stamp_ns=stamp_to_ns(msg)))
        self.trim_queues()
        self.try_publish()

    def on_label_array(self, msg) -> None:
        self.label_queue.append(LabelFrame(labels=np.asarray(msg.data, dtype=np.int32), stamp_ns=None))
        self.trim_queues()
        self.try_publish()

    def trim_queues(self) -> None:
        while len(self.raw_queue) > self.args.queue_depth:
            self.raw_queue.popleft()
        while len(self.label_queue) > self.args.queue_depth:
            self.label_queue.popleft()

    def try_publish(self) -> None:
        if self.args.sync_policy == "sequential":
            while self.raw_queue and self.label_queue:
                self.publish_pair(self.raw_queue.popleft(), self.label_queue.popleft())
            return

        max_diff_ns = int(float(self.args.max_stamp_diff) * 1_000_000_000)
        while self.raw_queue and self.label_queue:
            cloud = self.raw_queue[0]
            cloud_stamp = stamp_to_ns(cloud)
            best_idx = None
            best_diff = None
            for idx, label in enumerate(self.label_queue):
                if label.stamp_ns is None:
                    continue
                diff = abs(label.stamp_ns - cloud_stamp)
                if best_diff is None or diff < best_diff:
                    best_idx = idx
                    best_diff = diff
            if best_idx is None or best_diff is None:
                return
            if best_diff <= max_diff_ns:
                self.raw_queue.popleft()
                label = self.label_queue[best_idx]
                del self.label_queue[best_idx]
                self.publish_pair(cloud, label)
                continue

            earliest_label = self.label_queue[0]
            if earliest_label.stamp_ns is not None and earliest_label.stamp_ns < cloud_stamp - max_diff_ns:
                self.label_queue.popleft()
            elif cloud_stamp < earliest_label.stamp_ns - max_diff_ns:
                self.raw_queue.popleft()
            else:
                return

    def publish_pair(self, cloud: PointCloud2, label_frame: LabelFrame) -> None:
        count = int(cloud.width) * int(cloud.height)
        if label_frame.labels.shape[0] != count:
            self.get_logger().error(
                f"label count mismatch: labels={label_frame.labels.shape[0]} "
                f"points={count}. Frame dropped."
            )
            return

        categories = self.labels_to_hybrid(label_frame.labels)
        try:
            out = add_hybrid_semantic_field(cloud, categories)
        except ValueError as exc:
            self.get_logger().error(str(exc))
            return
        self.pub.publish(out)
        self.frame_count += 1

        if self.frame_count == 1 or self.frame_count % self.args.log_interval == 0:
            counts = np.bincount(categories.astype(np.int32), minlength=4)
            self.get_logger().info(
                f"semantic bridge frame {self.frame_count}: "
                f"unknown={int(counts[0])} plane={int(counts[1])} "
                f"gaussian={int(counts[2])} object={int(counts[3])}"
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-topic", default="/points_raw")
    parser.add_argument("--label-topic", default="/semantic_labels")
    parser.add_argument("--output-topic", default="/points_semantic")
    parser.add_argument(
        "--label-message-type",
        choices=("pointcloud2", "uint32_multi_array", "int32_multi_array"),
        default="pointcloud2",
    )
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--input-label-space", choices=("dataset", "hybrid"), default="dataset")
    parser.add_argument("--plane-labels", default="")
    parser.add_argument("--object-labels", default="")
    parser.add_argument("--ignore-labels", default="")
    parser.add_argument("--default-semantic", type=int, default=2)
    parser.add_argument("--sync-policy", choices=("timestamp", "sequential"), default="timestamp")
    parser.add_argument("--max-stamp-diff", type=float, default=0.02)
    parser.add_argument("--queue-depth", type=int, default=64)
    parser.add_argument("--log-interval", type=int, default=100)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rclpy.init()
    node = GenericSemanticFieldBridge(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
