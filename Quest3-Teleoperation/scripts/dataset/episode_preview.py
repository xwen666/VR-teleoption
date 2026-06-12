# filepath: /home/lyh/RealMan-Quest3-teleoperation/scripts/dataset/image_preview.py
#!/usr/bin/env python3
import os
import argparse

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import Image

import numpy as np
import cv2

SENSOR_QOS = QoSPresetProfiles.SENSOR_DATA.value

def yuyv_to_bgr(data_bytes: bytes, w: int, h: int):
    try:
        arr = np.frombuffer(data_bytes, dtype=np.uint8).reshape(h, w, 2)
        return cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_YUY2)
    except Exception:
        return None

def msg_to_bgr(msg: Image):
    enc = (msg.encoding or "").lower()
    if enc in ("yuv422_yuy2", "yuyv"):
        return yuyv_to_bgr(msg.data, msg.width, msg.height)
    if enc == "bgr8":
        try:
            return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        except Exception:
            return None
    # Other encodings are not converted, return None
    return None

def make_grid(images, cols=2, scale=0.75):
    # No scaling, concatenate at original resolution; fill missing images with black padding of same size
    if not images:
        return None
    # Find the first non-None image to determine uniform H, W
    base = next((im for im in images if im is not None), None)
    if base is None:
        return None
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    H, W = int(base.shape[0]*scale), int(base.shape[1]*scale)

    fixed = []
    for im in images:
        if im is None:
            fixed.append(np.zeros((H, W, 3), dtype=np.uint8))
            continue
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        h, w = im.shape[:2]
        # No scaling; if size is inconsistent, crop to base size
        if h >= H and w >= W:
            im = cv2.resize(im, (W, H))
            fixed.append(im)
        else:
            canvas = np.zeros((H, W, 3), dtype=np.uint8)
            h2, w2 = min(h, H), min(w, W)
            canvas[:h2, :w2] = im[:h2, :w2]
            fixed.append(canvas)
    # Pad to complete rows
    rows = (len(fixed) + cols - 1) // cols
    while len(fixed) < rows * cols:
        fixed.append(np.zeros((H, W, 3), dtype=np.uint8))
    # Concatenate
    grid_rows = []
    k = 0
    for _ in range(rows):
        grid_rows.append(cv2.hconcat(fixed[k:k+cols]))
        k += cols
    return np.vstack(grid_rows)

class ImagePreviewNode(Node):
    def __init__(self, topics, preview_fps=15.0, window_name="Camera Preview"):
        super().__init__("image_preview")
        self.window_name = window_name
        self.preview_period = 1.0 / max(preview_fps, 1.0)
        self.last_preview_ts = 0.0

        self.topic_list = list(topics)
        self.latest = {t: None for t in self.topic_list}  # topic -> np.ndarray(BGR)
        self.subs = []
        for t in self.topic_list:
            self.subs.append(self.create_subscription(Image, t, self._make_cb(t), SENSOR_QOS))

        # Auto-adjust window size based on image dimensions, no resize
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

        self.timer = self.create_timer(self.preview_period, self._on_timer)
        self.get_logger().info(f"Preview topics: {', '.join(self.topic_list)} @ {preview_fps} FPS")

    def _make_cb(self, topic_name):
        def _cb(msg: Image):
            bgr = msg_to_bgr(msg)
            if bgr is not None:
                # Overlay topic name and timestamp
                txt = f"{topic_name}  t={msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9:.3f}"
                cv2.putText(bgr, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                self.latest[topic_name] = bgr
        return _cb

    def _on_timer(self):
        imgs = [self.latest[t] for t in self.topic_list]
        mosaic = make_grid(imgs, cols=2)
        if mosaic is not None:
            cv2.imshow(self.window_name, mosaic)
            cv2.waitKey(1)

    def destroy_node(self):
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass
        super().destroy_node()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", nargs="+",
                        default=["/cam_head", "/cam_side", "/cam_wrist"],
                        help="camera image topics to preview")
    parser.add_argument("--fps", type=float, default=float(os.getenv("PREVIEW_FPS", "10")))
    parser.add_argument("--cols", type=int, default=int(os.getenv("PREVIEW_COLS", "2")), help="grid columns (no resize)")
    parser.add_argument("--window", default="Camera Preview")
    args = parser.parse_args()

    rclpy.init()
    node = ImagePreviewNode(args.topics, preview_fps=args.fps, window_name=args.window)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
