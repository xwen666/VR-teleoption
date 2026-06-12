import time
import cv2
import numpy as np
from functools import wraps

from .realsense import RealsenseCamera, CameraInfo
from .utils.log import get_logger_spdlog

logger = get_logger_spdlog()


def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        logger.info(f'time_it[{func.__name__}]: {(end_time - start_time) * 1000:.3f}ms')
        return result

    return wrapper


def point_from_single_depth(u, v, depth, cam_info: CameraInfo):
    """将单个像素 (u, v) 及其深度值转换为相机坐标系下的 3D 点 (X, Y, Z)"""
    if np.isnan(depth) or depth <= 0:
        return None

    Z = depth * cam_info.scale
    X = (u - cam_info.cx) * Z / cam_info.fx
    Y = (v - cam_info.cy) * Z / cam_info.fy

    return (X, Y, Z)


def point_cloud_from_depth_image(depth, camera: CameraInfo):
    assert depth.shape[0] == camera.height and depth.shape[1] == camera.width
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth * camera.scale
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    return cloud


def get_depth_in_bbox(depth_frame, bbox, depth_scale, max_search_radius=5):
    """
    Given depth frame (numpy array in raw units), and bbox (x1,y1,x2,y2),
    try to return a robust depth (meters) near the bbox center by searching
    a small neighborhood for non-zero depth.
    """
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    h, w = depth_frame.shape

    # bound center
    cx = min(max(cx, 0), w - 1)
    cy = min(max(cy, 0), h - 1)

    # search increasing radius for first valid depth
    for r in range(max_search_radius + 1):
        ys, xs = np.ogrid[
            max(cy - r, 0) : min(cy + r + 1, h), max(cx - r, 0) : min(cx + r + 1, w)
        ]
        patch = depth_frame[ys, xs]
        valid = patch > 0
        if np.any(valid):
            # take median of valid depths to reduce noise
            median = np.median(patch[valid])
            return float(median) * depth_scale
    return None


def debug_sth(cam: RealsenseCamera):
    while True:
        time.sleep(0.01)

        frame = cam.get_latest_frames()
        if frame is None:
            continue

        # 有效测量距离: 0.3m - 3m, 实测 15cm 内的物体深度值已经是负的了
        clipping_distance_min = 0.35 / cam.get_depth_scale()
        clipping_distance_max = 1 / cam.get_depth_scale()

        # Remove background - Set pixels further than clipping_distance to gre
        grey_color = 153
        # depth image is 1 channel, color is 3 channels
        depth_image_3d = np.dstack((frame.depth, frame.depth, frame.depth))
        bg_removed = np.where(
            (depth_image_3d > clipping_distance_max)
            | (depth_image_3d < clipping_distance_min),
            grey_color,
            frame.color,
        )

        # Render images:
        #   depth align to color on left
        #   depth on right
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(frame.depth, alpha=0.03), cv2.COLORMAP_JET
        )
        images = np.hstack((bg_removed, depth_colormap))

        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('Align Example', images)
        key = cv2.waitKey(1)

        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
