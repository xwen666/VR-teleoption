import time
import cv2
import numpy as np
import open3d as o3d

from typing import Any
from collections.abc import Callable

from ultralytics.models import SAM, YOLO

from .misc import (
    get_depth_in_bbox,
    point_cloud_from_depth_image,
    point_from_single_depth,
    time_it,
)
from .realsense import CamFrame, CameraInfo, RealsenseCamera
from .utils.singleton import singleton
from .utils.log import get_logger_spdlog

logger = get_logger_spdlog()

DETECT_CONF_THRESHOLD = 0.4


@singleton
class YoloManager:
    m_yolo: Any = None
    m_sam: Any = None
    video_writer: Any = None

    detect_targets: list = []
    on_detected: Callable | None = None
    on_segmented_pos: Callable | None = None

    def load_models(self):
        self.m_yolo, self.video_writer = self.load_yolo_model()
        self.m_sam = self.load_sam_model()

    def load_yolo_model(
        self,
        model_path='model/yolo/yolo11m.pt',
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        logger.info(f'load yolo model: {model_path} ..')

        model = YOLO(model=model_path)
        logger.debug(f'yolo model info: {model.info()}')

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(
            'out/yolo_realsense_out.mp4', fourcc, fps, (width, height)
        )

        return model, out_writer

    def load_sam_model(self, model_path='model/sam2/sam2.1_t.pt'):
        logger.info(f'load sam2 model: {model_path} ..')

        model = SAM(model=model_path)
        logger.debug(f'sam2 model info: {model.info()}')

        return model

    def yolo_predict(self, source, imgsz, conf=0.3):
        if not self.m_yolo:
            logger.warn('load yolo model before predict')
            self.m_yolo, self.out_writer = self.load_yolo_model()

        ts_start = time.time()
        result = self.m_yolo.predict(
            source=source,
            device='cpu',
            imgsz=imgsz,
            verbose=False,
            conf=conf,
        )[0]
        ts_delta = time.time() - ts_start
        logger.debug(f'yolo predict time: {ts_delta:.6f}s')
        return result

    def sam_mask_with_bbox(self, color_np, xyxy):
        """sam 分割目标检测 bbox"""
        if not self.m_sam:
            logger.warn('load sam model before seg')
            self.m_sam = self.load_sam_model()

        ts_start = time.time()
        result = self.m_sam.predict(source=color_np, bboxes=xyxy)[0]
        ts_delta = time.time() - ts_start

        logger.debug(f'sam seg time: {ts_delta}s')
        return result

    def colored_mask(self, mask_bool):
        return (mask_bool > 0).astype(np.uint8) * 255

    def find_mask_center(self, mask_colored):
        contours, _ = cv2.findContours(
            mask_colored, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        M = cv2.moments(contours[0])
        if M['m00'] == 0:
            return None

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return (cx, cy)

    @time_it
    def to_3d_cloud_o3d(self, frame: CamFrame, mask_np, cam_info: CameraInfo):
        """从深度图反向建立 3d 点云，比下面哪个自己计算快多了，至少两倍速"""
        if mask_np is not None:
            masked_depth_np = frame.depth.copy()
            target_mask = mask_np & (frame.depth > 0)
            masked_depth_np[~target_mask] = 0
        else:
            masked_depth_np = frame.depth

        depth_img = o3d.geometry.Image(masked_depth_np)
        color_img = o3d.geometry.Image(frame.color)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img,
            depth_img,
            depth_scale=1 / cam_info.scale,  # 缩放有点不太一样
            # convert_rgb_to_intensity=True,
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
        )
        intrinsic.set_intrinsics(
            cam_info.width,
            cam_info.height,
            cam_info.fx,
            cam_info.fy,
            cam_info.cx,
            cam_info.cy,
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        return pcd

    @time_it
    def to_3d_cloud(self, frame: CamFrame, mask_np, cam_info: CameraInfo):
        cloud_np = point_cloud_from_depth_image(frame.depth, cam_info)
        mask = (mask_np > 0) & (frame.depth > 0)

        cloud_masked = cloud_np[mask]
        color_masked = frame.color[mask]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        pcd.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
        return pcd

    def set_detect_targets(self, targets):
        self.detect_targets = targets

    def set_on_detected(self, on_detected):
        self.on_detected = on_detected

    def set_on_segmented_pos(self, on_segmented_pos):
        self.on_segmented_pos = on_segmented_pos

    def watch_on_frame_for_target(self, cam: RealsenseCamera):
        last_ts = 0

        while True:
            frame = cam.get_latest_frames()
            if frame is None:
                time.sleep(0.1)
                continue

            if last_ts != frame.timestamp:
                last_ts = frame.timestamp
            else:
                time.sleep(0.01)
                continue

            width, height = 640, 480
            result = self.yolo_predict(source=frame.color, imgsz=max(width, height))

            if not result.boxes:
                # TODO: may popup with click selection
                logger.debug('no detected object, skip')
                time.sleep(0.1)
                continue

            vis_img = result.plot()
            cv2.imwrite('out/detect.jpg', vis_img)

            logger.info('check object before seg')
            xyxy = None

            for box in result.boxes:
                conf = float(box.conf)
                xyxy_v = box.xyxy.numpy().astype(int)[0].tolist()
                cls_id = int(box.cls.item())
                label = self.m_yolo.names[cls_id]
                logger.warn(f'detected: {label}, conf: {conf:.3f}, xyxy: {xyxy_v}')

                if (
                    self.detect_targets
                    and label in self.detect_targets
                    and float(box.conf) > DETECT_CONF_THRESHOLD
                ):
                    xyxy = xyxy_v
                    break

            if xyxy:
                return frame, xyxy

    def get_frame_center_pos_in_segment_of_detected_target(
        self, frame: CamFrame, xyxy, cam: RealsenseCamera
    ):
        logger.info(f'seg with sam2 for detected object, bbox: {xyxy}')
        result = self.sam_mask_with_bbox(color_np=frame.color, xyxy=xyxy)
        mask_np = result.masks.data[0].numpy()

        colored_mask = self.colored_mask(mask_np)
        cv2.imwrite('out/mask.png', colored_mask, [cv2.IMWRITE_PNG_BILEVEL, 1])

        res = self.find_mask_center(colored_mask)
        if not res:
            logger.warn('lost center in mask')
            return None

        cx, cy = res
        depth = frame.depth[cx, cy]
        logger.debug(
            f'mask center({cx}, {cy}) distance to cam: {depth * cam.depth_scale:.6f}m'
        )

        cam_info = cam.get_info(stream='depth')
        target_pos = point_from_single_depth(cx, cy, depth, cam_info)
        target_pos = [float(x) for x in target_pos] if target_pos else None
        return frame, target_pos

    def debug_sam2(self, cam: RealsenseCamera, targets=None):
        self.load_models()

        if targets:
            self.detect_targets = targets

        cam_info = cam.get_info(stream='depth')

        last_ts = 0

        while True:
            frame = cam.get_latest_frames()
            if frame is None:
                time.sleep(0.1)
                continue

            if last_ts != frame.timestamp:
                last_ts = frame.timestamp
            else:
                time.sleep(0.01)
                continue

            width, height = 640, 480
            result = self.yolo_predict(source=frame.color, imgsz=max(width, height))

            if not result.boxes:
                # TODO: may popup with click selection
                logger.debug('no detected object, skip')
                continue

            vis_img = result.plot()
            cv2.imwrite('out/detect.jpg', vis_img)
            # cv2.imshow('selec obj', color_np.copy())

            logger.info('check object before seg')
            xyxy = None

            for box in result.boxes:
                conf = float(box.conf)
                xyxy_v = box.xyxy.numpy().astype(int)[0].tolist()
                cls_id = int(box.cls.item())
                label = self.m_yolo.names[cls_id]
                logger.warn(f'detected: {label}, conf: {conf:.3f}, xyxy: {xyxy_v}')

                if (
                    self.detect_targets
                    and label in self.detect_targets
                    and float(box.conf) > DETECT_CONF_THRESHOLD
                ):
                    xyxy = xyxy_v
                    break

            # no valid target
            if not xyxy:
                continue

            logger.info('seg with sam2 for detected object bbox: {xyxy}')
            result = self.sam_mask_with_bbox(color_np=frame.color, xyxy=xyxy)
            mask_np = result.masks.data[0].numpy()

            colored_mask = self.colored_mask(mask_np)
            cv2.imwrite('out/mask.png', colored_mask, [cv2.IMWRITE_PNG_BILEVEL, 1])

            pcd = self.to_3d_cloud_o3d(frame, mask_np, cam_info)
            o3d.io.write_point_cloud('out/pcd_o3d_auto.ply', pcd)
            # pcd = self.to_3d_cloud(frame, mask_np, cam_info)
            # o3d.io.write_point_cloud('out/pcd_o3d_manual.ply', pcd)

    def debug_yolo(self, cam: RealsenseCamera):
        width, height, fps = 640, 480, 30
        self.m_yolo, self.video_writer = self.load_yolo_model(
            width=width,
            height=height,
            fps=fps,
        )

        last_ts = 0
        predict_time = 0
        ct = 0

        while True:
            frame = cam.get_latest_frames()
            if frame is None:
                time.sleep(0.1)
                continue

            if last_ts != frame.timestamp:
                last_ts = frame.timestamp
            else:
                time.sleep(0.02)
                continue

            ts_start = time.time()
            result = self.m_yolo.predict(
                source=frame.color, device='cpu', imgsz=max(width, height), verbose=False
            )[0]
            ts_delta = time.time() - ts_start

            predict_time += ts_delta
            ct += 1

            logger.debug(
                f'predict time: {ts_delta}s, '
                f'fps(cur, mean): {1 / ts_delta:.2f}/{ct / predict_time:.2f}'
            )

            if result is None or not result.boxes:
                logger.debug('no result')
                continue

            # Draw results
            annotated = frame.color.copy()
            for box in result.boxes:
                # logger.debug(f'box: {box}')

                # box.xyxy, box.conf, box.cls
                x1, y1, x2, y2 = box.xyxy.numpy().astype(int)[0]
                conf = float(box.conf)
                cls_id = int(box.cls)
                label = self.m_yolo.names[cls_id]

                # get robust depth in bbox center
                depth_m = get_depth_in_bbox(frame.depth, (x1, y1, x2, y2), cam.depth_scale)
                depth_text = f'{depth_m:.2f}m' if depth_m is not None else 'N/A'

                logger.debug(f'detected: {label}, depth: {depth_text}')

                # draw rectangle and text
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f'{label} {conf:.2f} {depth_text}'
                # ensure text background visibility
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(
                    annotated, (x1, y1 - th - 6), (x1 + tw + 6, y1), (0, 255, 0), -1
                )
                cv2.putText(
                    annotated,
                    text,
                    (x1 + 3, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                )

                # draw a small circle at center
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                cv2.circle(annotated, (cx, cy), 3, (0, 0, 255), -1)

            if self.video_writer:
                self.video_writer.write(annotated)

            cv2.imshow('YOLO RealSense', annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

        cam.pipeline.stop()
        self.video_writer.release()
        logger.warn('clean exit')

    def debug_yolo_mini(self, cam: RealsenseCamera):
        # obb 检测的效果差很多，只有各种角度的 ship ???
        # yoloe 这个分割 fps=4
        # model_path = 'model/yolo-world/yolov8m-worldv2.pt'

        width, height, fps = 640, 480, 30
        self.m_yolo, self.video_writer = self.load_yolo_model(
            # model_path='model/yoloe/yoloe-11m-seg-pf.pt',
            width=width,
            height=height,
            fps=fps,
        )

        last_ts = 0
        predict_time = 0
        ct = 0

        while True:
            frame = cam.get_latest_frames()
            if frame is None:
                time.sleep(0.1)
                continue

            if last_ts != frame.timestamp:
                last_ts = frame.timestamp
            else:
                time.sleep(0.02)
                continue

            last_ts = frame.timestamp

            ts_start = time.time()
            results = self.m_yolo.predict(
                source=frame.color,
                device='cpu',
                imgsz=max(width, height),
                verbose=False,
                conf=DETECT_CONF_THRESHOLD,
            )
            ts_delta = time.time() - ts_start

            predict_time += ts_delta
            ct += 1

            logger.debug(
                f'predict time: {ts_delta}s, '
                f'fps(cur, mean): {1 / ts_delta:.2f}/{ct / predict_time:.2f}'
            )

            # 仅计算帧率，20 次打印一次
            if ct % 20 > 0:
                continue

            for result in results:
                boxes = result.boxes  # Boxes object for bounding box outputs
                masks = result.masks  # Masks object for segmentation masks outputs
                keypoints = result.keypoints  # Keypoints object for pose outputs
                probs = result.probs  # Probs object for classification outputs
                obb = result.obb  # Oriented boxes object for OBB outputs
                logger.info(f'm: {masks}, p: {probs}, o: {obb}')
                result.show()  # display to screen
