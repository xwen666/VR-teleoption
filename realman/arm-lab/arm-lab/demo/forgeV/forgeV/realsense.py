import threading
import time
import numpy as np
import pyrealsense2 as rs

from dataclasses import dataclass
from typing import Any
from numpy.typing import NDArray

from .models import Pos
from .utils.singleton import singleton
from .utils.log import get_logger_spdlog

logger = get_logger_spdlog()


@dataclass
class CameraInfo:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    scale: float


@dataclass
class CamFrame:
    timestamp: float = 0
    color: Any = None  # bgr8
    depth: Any = None  # z16
    pos: Pos | None = None  # current pos on fetch frame


@singleton
class RealsenseCamera:
    def __init__(self, width=640, height=480, fps=30):
        self.pipeline = rs.pipeline()

        self.watch_enabled = False
        self.frame_latest = CamFrame()
        self.fetch_pos_method = None

        config = rs.config()
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        logger.info('start realsense cam')
        self.profile = self.pipeline.start(config)

        device = self.profile.get_device()
        device_product_line = device.get_info(rs.camera_info.product_line)

        self.depth_scale = device.first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)

        logger.info(f'device_product_line: {device_product_line}')
        logger.info(f'depth scale: {self.depth_scale}')

    def get_info(self, stream='color') -> CameraInfo:
        stream = getattr(rs.stream, stream)
        intrinsics = (
            self.profile.get_stream(stream).as_video_stream_profile().get_intrinsics()
        )
        logger.debug(f'{stream} cam intrinsics: {intrinsics}')

        return CameraInfo(
            width=intrinsics.width,
            height=intrinsics.height,
            fx=intrinsics.fx,
            fy=intrinsics.fy,
            cx=intrinsics.ppx,
            cy=intrinsics.ppy,
            scale=self.depth_scale,
        )

    def set_fetch_pos_method(self, fetch_pose):
        self.fetch_pos_method = fetch_pose

    def cosume_frames(self, size: int = 30):
        for _ in range(size):
            self.pipeline.wait_for_frames()

    def get_depth_scale(self):
        return self.depth_scale

    def get_latest_frames(self, delta_ms=100) -> CamFrame | None:
        delta_latest = time.time() - self.frame_latest.timestamp
        if delta_latest > delta_ms / 1e3:
            logger.warn(f'latest frame is too old, delta_t: {delta_latest:.6f}s')
            return None

        return self.frame_latest

    def get_aligned_frames(self) -> CamFrame:
        frames = self.pipeline.wait_for_frames()

        aligned_frames = self.align.process(frames)

        return CamFrame(
            timestamp=time.time(),
            color=np.asanyarray(aligned_frames.get_color_frame().get_data()).copy(),
            depth=np.asanyarray(aligned_frames.get_depth_frame().get_data()).copy(),
            pos=self.fetch_pos_method() if self.fetch_pos_method else None,
        )

    def print_sth(self):
        frame = self.get_aligned_frames()

        images = {'color': frame.color, 'depth': frame.depth}
        logger.info(f'{images}')

    def start_watching(self):
        def watching():
            logger.info('cam: start watch frames ..')
            self.watch_enabled = True
            while self.watch_enabled:
                self.frame_latest = self.get_aligned_frames()

            logger.info('cam: exit frame watch loop')

        threading.Thread(target=watching).start()

    def stop_watching(self):
        self.watch_enabled = False
