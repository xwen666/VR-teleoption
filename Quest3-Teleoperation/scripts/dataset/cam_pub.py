#!/usr/bin/env python3
import os
import argparse
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import Image
import pyrealsense2 as rs

SENSOR_QOS = QoSPresetProfiles.SENSOR_DATA.value
ENCODING = 'yuv422_yuy2'   # ROS encoding name for YUYV
BYTES_PER_PIXEL = 2

CAM_WRIST_SERIAL = 243122076407
CAM_HEAD_SERIAL = 335222074941
CAM_SIDE_SERIAL = 327122074011

CAM_FPS = 60

class SingleCameraNode(Node):
    def __init__(self, name: str, serial: str, topic: str, frame_id: str,
                 width: int, height: int, fps: int):
        super().__init__(name)
        self.serial = serial
        self.topic = topic
        self.frame_id = frame_id
        self.width = width
        self.height = height
        self.fps = fps

        self.pub = self.create_publisher(Image, self.topic, SENSOR_QOS)

        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.yuyv, CAM_FPS)
        profile = self.pipeline.start(cfg)

        # Drop frames - publish one frame as it arrives
        try:
            dev = profile.get_device()
            color = dev.first_color_sensor()
            color.set_option(rs.option.frames_queue_size, 1)
        except Exception:
            pass

        self.timer = self.create_timer(1.0 / self.fps, self.publish_once)
        self.get_logger().info(
            f"[{self.get_name()}] serial={self.serial}, topic={self.topic}, "
            f"res={self.width}x{self.height}@{self.fps}Hz format=YUYV"
        )

    # def cmd_callback(self, msg: String):
    #     cmd = (msg.data or "").strip().lower()
    #     if cmd == "start" and not self.enabled:
    #         self.enabled = True
    #         self.get_logger().info("Collect START -> enable publishing")
    #     elif cmd == "stop" and self.enabled:
    #         self.enabled = False
    #         self.get_logger().info("Collect STOP -> disable publishing")

    def publish_once(self):
        # if not self.enabled:
        #     return
        frames = self.pipeline.wait_for_frames()
        if not frames or frames.size() == 0:
            return
        color = frames.get_color_frame()
        if not color:
            return

        yuyv = memoryview(color.get_data())
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.height = self.height
        msg.width = self.width
        msg.encoding = ENCODING
        msg.is_bigendian = 0
        msg.step = self.width * BYTES_PER_PIXEL
        msg.data = yuyv.tobytes()
        self.pub.publish(msg)

    def destroy_node(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass
        super().destroy_node()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", required=True, help="RealSense camera name")
    parser.add_argument("--frame-id", default="camera", help="frame_id")
    parser.add_argument("--width", type=int, default=int(os.getenv('IMAGE_WIDTH', '640')))
    parser.add_argument("--height", type=int, default=int(os.getenv('IMAGE_HEIGHT', '480')))
    parser.add_argument("--fps", type=int, default=int(os.getenv('FRAME_RATE', '15')))
    parser.add_argument("--node-name", default=None)
    args = parser.parse_args()

    rclpy.init()
    node_name = args.node_name or f"cam_{args.camera}"
    if args.camera == "wrist":
        camera_serial = str(CAM_WRIST_SERIAL)
        camera_topic = '/cam_wrist'
    elif args.camera == "head":
        camera_serial = str(CAM_HEAD_SERIAL)
        camera_topic = '/cam_head'
    elif args.camera == "side":
        camera_serial = str(CAM_SIDE_SERIAL)
        camera_topic = '/cam_side'
    else:
        raise ValueError(f"Unknown camera name: {args.camera}")
    
    node = SingleCameraNode(node_name, camera_serial, camera_topic, args.frame_id,
                            args.width, args.height, args.fps)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if node:
                node.destroy_node()
        finally:
            print("Shutting down rclpy...")
            if rclpy.ok():
                try:
                    rclpy.shutdown()
                except Exception:
                    pass

if __name__ == "__main__":
    main()