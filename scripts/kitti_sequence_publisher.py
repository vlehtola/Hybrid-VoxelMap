#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import PointCloud2, PointField


class KittiSequencePublisher(Node):
    def __init__(self):
        super().__init__("kitti_sequence_publisher")
        self.declare_parameter("sequence_path", "data/KITTI/dataset/sequences/08")
        self.declare_parameter("topic", "/velodyne_points")
        self.declare_parameter("frame_id", "camera_init")
        self.declare_parameter("rate", 10.0)
        self.declare_parameter("loop", False)
        self.declare_parameter("start_index", 0)
        self.declare_parameter("end_index", -1)

        self.sequence_path = Path(self.get_parameter("sequence_path").value).expanduser()
        self.topic = self.get_parameter("topic").value
        self.frame_id = self.get_parameter("frame_id").value
        self.rate = float(self.get_parameter("rate").value)
        self.loop = bool(self.get_parameter("loop").value)
        self.start_index = int(self.get_parameter("start_index").value)
        self.end_index = int(self.get_parameter("end_index").value)

        self.velodyne_dir = self.sequence_path / "velodyne"
        if not self.velodyne_dir.is_dir():
            raise RuntimeError(f"Missing velodyne directory: {self.velodyne_dir}")

        scans = sorted(self.velodyne_dir.glob("*.bin"))
        if self.end_index >= 0:
            scans = scans[: self.end_index + 1]
        self.scans = scans[self.start_index :]
        if not self.scans:
            raise RuntimeError(f"No KITTI .bin scans found under {self.velodyne_dir}")

        self.times = self._load_times()
        self.publisher = self.create_publisher(PointCloud2, self.topic, QoSProfile(depth=4))
        self.index = 0
        period = 1.0 / max(self.rate, 1e-6)
        self.timer = self.create_timer(period, self.publish_next)

        self.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name="time", offset=20, datatype=PointField.FLOAT32, count=1),
            PointField(name="ring", offset=24, datatype=PointField.UINT16, count=1),
        ]
        self.point_dtype = np.dtype(
            {
                "names": ["x", "y", "z", "intensity", "time", "ring"],
                "formats": ["<f4", "<f4", "<f4", "<f4", "<f4", "<u2"],
                "offsets": [0, 4, 8, 16, 20, 24],
                "itemsize": 32,
            }
        )
        self.get_logger().info(
            f"Publishing {len(self.scans)} KITTI scans from {self.sequence_path} to {self.topic}"
        )

    def _load_times(self):
        times_path = self.sequence_path / "times.txt"
        if not times_path.is_file():
            return None
        values = []
        with times_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    values.append(float(line))
        return values

    def publish_next(self):
        if not rclpy.ok():
            return
        if self.index >= len(self.scans):
            if self.loop:
                self.index = 0
            else:
                self.get_logger().info("KITTI sequence finished")
                rclpy.shutdown()
                return

        scan_path = self.scans[self.index]
        raw = np.fromfile(scan_path, dtype=np.float32)
        if raw.size % 4 != 0:
            self.get_logger().warn(f"Skipping malformed scan: {scan_path}")
            self.index += 1
            return
        points = raw.reshape((-1, 4))
        cloud = np.zeros(points.shape[0], dtype=self.point_dtype)
        cloud["x"] = points[:, 0]
        cloud["y"] = points[:, 1]
        cloud["z"] = points[:, 2]
        cloud["intensity"] = points[:, 3]

        if points.shape[0] > 1:
            cloud["time"] = np.linspace(0.0, 0.1, points.shape[0], endpoint=False, dtype=np.float32)

        msg = PointCloud2()
        stamp = self.get_clock().now().to_msg()
        if self.times is not None and self.start_index + self.index < len(self.times):
            t = self.times[self.start_index + self.index]
            msg.header.stamp.sec = int(t)
            msg.header.stamp.nanosec = int((t - int(t)) * 1e9)
        else:
            msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.height = 1
        msg.width = points.shape[0]
        msg.fields = self.fields
        msg.is_bigendian = False
        msg.point_step = 32
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = cloud.tobytes()
        try:
            self.publisher.publish(msg)
        except Exception as exc:
            self.get_logger().debug(f"Publish skipped during shutdown: {exc}")
            return

        if self.index == 0 or (self.index + 1) % 100 == 0:
            self.get_logger().info(f"Published KITTI frame {scan_path.stem} ({self.index + 1}/{len(self.scans)})")
        self.index += 1


def main():
    rclpy.init()
    node = None
    try:
        node = KittiSequencePublisher()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
