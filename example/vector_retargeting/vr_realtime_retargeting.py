import json
import multiprocessing
import os
import socket
import sys
import time
from pathlib import Path
from queue import Empty
from typing import Literal, Optional


def _ensure_vulkan_can_load_system_libstdcxx():
    """Conda's old libstdc++ can hide symbols required by Mesa Vulkan drivers."""
    if sys.platform != "linux" or os.environ.get("DEX_RETARGETING_LIBSTDCXX_FIXED"):
        return

    required_symbol = b"GLIBCXX_3.4.29"
    conda_libstdcxx = Path(sys.prefix) / "lib" / "libstdc++.so.6"
    system_libstdcxx = Path("/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
    if not conda_libstdcxx.exists() or not system_libstdcxx.exists():
        return

    try:
        conda_is_old = required_symbol not in conda_libstdcxx.read_bytes()
        system_is_new_enough = required_symbol in system_libstdcxx.read_bytes()
    except OSError:
        return
    if not conda_is_old or not system_is_new_enough:
        return

    ld_preload = os.environ.get("LD_PRELOAD", "")
    system_libstdcxx_str = str(system_libstdcxx)
    if system_libstdcxx_str not in ld_preload.split(":"):
        os.environ["LD_PRELOAD"] = (
            f"{system_libstdcxx_str}:{ld_preload}" if ld_preload else system_libstdcxx_str
        )
    os.environ["DEX_RETARGETING_LIBSTDCXX_FIXED"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])


_ensure_vulkan_can_load_system_libstdcxx()

import cv2
import numpy as np
import sapien
import tyro
from loguru import logger
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from revo2_geometric_retargeting import Revo2GeometricRetargeting
from vr_hand_detector import VRHandDetector


def append_debug_line(log_file: Optional[str], source: str, message: str):
    if not log_file:
        return
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as file:
        file.write(f"[{timestamp}] [{source}] {message}\n")


QUEST_TO_SAPIEN = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)


def quest_position_to_sapien(position: np.ndarray) -> np.ndarray:
    # Quest/Unity uses x-right, y-up, z-forward. SAPIEN uses z-up.
    return QUEST_TO_SAPIEN @ position


def get_revo2_controllable_joint_names(hand_type: str) -> list[str]:
    prefix = hand_type.lower()
    return [
        f"{prefix}_thumb_metacarpal_joint",
        f"{prefix}_thumb_proximal_joint",
        f"{prefix}_index_proximal_joint",
        f"{prefix}_middle_proximal_joint",
        f"{prefix}_ring_proximal_joint",
        f"{prefix}_pinky_proximal_joint",
    ]


def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = quat / np.linalg.norm(quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    return quat_xyzw_to_matrix(np.array([quat[1], quat[2], quat[3], quat[0]]))


def matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    trace = np.trace(matrix)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        
        z = (matrix[1, 0] - matrix[0, 1]) / s
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / s
            x = 0.25 * s
            y = (matrix[0, 1] + matrix[1, 0]) / s
            z = (matrix[0, 2] + matrix[2, 0]) / s
        elif axis == 1:
            s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / s
            x = (matrix[0, 1] + matrix[1, 0]) / s
            y = 0.25 * s
            z = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / s
            x = (matrix[0, 2] + matrix[2, 0]) / s
            y = (matrix[1, 2] + matrix[2, 1]) / s
            z = 0.25 * s
    quat = np.array([w, x, y, z])
    return quat / np.linalg.norm(quat)


def start_retargeting(
    queue: Optional[multiprocessing.Queue],
    robot_dir: str,
    config_path: str,
    transport: str,
    host: Optional[str],
    port: int,
    follow_wrist: bool,
    follow_wrist_rotation: bool,
    wrist_position_scale: float,
    wrist_smoothing_alpha: float,
    revo2_retargeting_mode: str,
    revo2_smoothing_alpha: float,
    revo2_finger_gain: float,
    revo2_thumb_gain: float,
    revo2_thumb_metacarpal_gain: float,
    revo2_thumb_metacarpal_mode: str,
    publish_ros_hand_qpos: bool,
    ros_hand_qpos_host: str,
    ros_hand_qpos_port: int,
    forward_ros_wrist_pose: bool,
    ros_wrist_pose_host: str,
    ros_wrist_pose_port: int,
    debug_wrist_stream: bool,
    debug_wrist_period: float,
    wrist_debug_log_file: Optional[str],
    stream_poll_hz: float,
    viewer_render_hz: float,
):
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Start retargeting with config {config_path}")
    append_debug_line(
        wrist_debug_log_file,
        "META",
        f"start retargeting transport={transport} bind={host}:{port}",
    )
    config = RetargetingConfig.load_from_file(config_path)
    filepath = Path(config.urdf_path)
    robot_name = filepath.stem
    use_revo2_geometric = "revo2" in robot_name and revo2_retargeting_mode == "brainco"
    retargeting = None if use_revo2_geometric else config.build()
    hand_type = "Right" if "right" in config_path.lower() else "Left"

    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")

    # Setup
    scene = sapien.Scene()
    render_mat = sapien.render.RenderMaterial()
    render_mat.base_color = [0.06, 0.08, 0.12, 1]
    render_mat.metallic = 0.0
    render_mat.roughness = 0.9
    render_mat.specular = 0.8
    scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])

    # Lighting
    scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
    scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.set_environment_map(
        create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
    )
    scene.add_area_light_for_ray_tracing(
        sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5
    )

    # Camera
    cam = scene.add_camera(
        name="Cheese!", width=600, height=600, fovy=1, near=0.1, far=10
    )
    cam.set_local_pose(sapien.Pose([0.50, 0, 0.0], [0, 0, 0, -1]))

    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.control_window.show_origin_frame = False
    viewer.control_window.move_speed = 0.01
    viewer.control_window.toggle_camera_lines(False)
    viewer.set_camera_pose(cam.get_local_pose())
    stream_poll_dt = 1.0 / max(float(stream_poll_hz), 1.0)
    viewer_render_dt = 1.0 / max(float(viewer_render_hz), 1.0)

    # Load robot and set it to a good pose to take picture
    loader = scene.create_urdf_loader()
    # Initialize VR detector with robot name for robot-specific adaptations
    detector = VRHandDetector(
        hand_type=hand_type,
        transport=transport,
        host=host,
        port=port,
        robot_name=robot_name,
    )
    logger.info(
        f"Listening for {detector.landmark_prefix} via "
        f"{transport.upper()} on {detector.host}:{detector.port}"
    )

    loader.load_multiple_collisions_from_file = True
    if "ability" in robot_name:
        loader.scale = 1.5
    elif "dclaw" in robot_name:
        loader.scale = 1.25
    elif "allegro" in robot_name:
        loader.scale = 1.4
    elif "shadow" in robot_name:
        loader.scale = 0.9
    elif "bhand" in robot_name:
        loader.scale = 1.5
    elif "leap" in robot_name:
        loader.scale = 1.4
    elif "svh" in robot_name:
        loader.scale = 1.5
    elif "xhand" in robot_name:
        loader.scale = 1.1
    elif "bidexhand" in robot_name:
        loader.scale = 1.0
    elif "revo2" in robot_name:
        loader.scale = 1.0

    if "glb" not in robot_name:
        glb_filepath = filepath.with_name(f"{filepath.stem}_glb.urdf")
        filepath = str(glb_filepath if glb_filepath.exists() else filepath)
    else:
        filepath = str(filepath)

    robot = loader.load(filepath)
    revo2_retargeting = None
    if use_revo2_geometric:
        revo2_retargeting = Revo2GeometricRetargeting(
            filepath,
            hand_side=hand_type.lower(),
            smoothing_alpha=revo2_smoothing_alpha,
            finger_gain=revo2_finger_gain,
            thumb_gain=revo2_thumb_gain,
            thumb_metacarpal_gain=revo2_thumb_metacarpal_gain,
            thumb_metacarpal_mode=revo2_thumb_metacarpal_mode,
        )
        logger.info(
            "Using BrainCo geometric retargeting for Revo2 "
            f"(6 controllable DOF + URDF mimic joints, "
            f"finger_gain={revo2_finger_gain}, thumb_gain={revo2_thumb_gain}, "
            f"thumb_metacarpal_gain={revo2_thumb_metacarpal_gain}, "
            f"thumb_metacarpal_mode={revo2_thumb_metacarpal_mode})."
        )

    robot_base_pose = sapien.Pose([0, 0, -0.15])
    if "shadow" in robot_name or "bhand" in robot_name:
        robot_base_pose = sapien.Pose([0, 0, -0.2])
    elif "allegro" in robot_name:
        robot_base_pose = sapien.Pose([0, 0, -0.05])
    elif "svh" in robot_name:
        robot_base_pose = sapien.Pose([0, 0, -0.13])
    elif "revo2" in robot_name:
        robot_base_pose = sapien.Pose([0, 0, -0.15])
    robot.set_pose(robot_base_pose)

    # Different robot loader may have different orders for joints
    sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    retargeting_to_sapien = None
    if retargeting is not None:
        retargeting_joint_names = retargeting.joint_names
        retargeting_to_sapien = np.array(
            [retargeting_joint_names.index(name) for name in sapien_joint_names]
        ).astype(int)

    last_status_log_time = 0.0
    last_wrist_debug_log_time = 0.0
    last_viewer_render_time = 0.0
    last_landmark_packets = 0
    last_wrist_packets = 0
    wrist_origin_position = None
    wrist_origin_rotation = None
    wrist_filtered_position = None
    wrist_filtered_quat = None
    wrist_smoothing_alpha = float(np.clip(wrist_smoothing_alpha, 0.0, 1.0))
    robot_base_rotation = quat_wxyz_to_matrix(robot_base_pose.q)
    ros_udp_socket = None
    if publish_ros_hand_qpos or forward_ros_wrist_pose:
        ros_udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    revo2_controllable_joint_names = get_revo2_controllable_joint_names(hand_type)
    if publish_ros_hand_qpos:
        logger.info(
            f"Publishing hand_qpos UDP to {ros_hand_qpos_host}:{ros_hand_qpos_port}"
        )
    if forward_ros_wrist_pose:
        logger.info(
            f"Forwarding wrist_pose UDP to {ros_wrist_pose_host}:{ros_wrist_pose_port}"
        )
    while not viewer.closed:
        if queue is None:
            time.sleep(stream_poll_dt)
        else:
            try:
                queue.get(timeout=5)
            except Empty:
                logger.error(
                    "Fail to fetch image from camera in 5 secs. Please check your web camera device."
                )
                return

        _, joint_pos, keypoint_2d, _ = detector.detect()
        now = time.monotonic()

        if joint_pos is None:
            if now - last_status_log_time > 1.0:
                logger.warning(
                    f"{hand_type} hand is not detected. Waiting for "
                    f"{detector.landmark_prefix} on {detector.host}:{detector.port}. "
                    f"Packets so far: landmarks={detector.landmark_packets}, "
                    f"wrist={detector.wrist_packets}."
                )
                last_status_log_time = now
        else:
            if (
                now - last_status_log_time > 1.0
                or detector.landmark_packets == 1
            ):
                logger.info(
                    f"Receiving {hand_type} hand stream: "
                    f"landmarks={detector.landmark_packets} "
                    f"(+{detector.landmark_packets - last_landmark_packets}/s), "
                    f"wrist={detector.wrist_packets} "
                    f"(+{detector.wrist_packets - last_wrist_packets}/s)."
                )
                last_landmark_packets = detector.landmark_packets
                last_wrist_packets = detector.wrist_packets
                last_status_log_time = now
            if use_revo2_geometric:
                landmarks = detector.get_vr_hand_landmarks()
                qpos = revo2_retargeting.qpos_from_landmarks(
                    landmarks, sapien_joint_names
                )
            else:
                retargeting_type = retargeting.optimizer.retargeting_type
                indices = retargeting.optimizer.target_link_human_indices
                if retargeting_type == "POSITION":
                    indices = indices
                    ref_value = joint_pos[indices, :]
                else:
                    origin_indices = indices[0, :]
                    task_indices = indices[1, :]
                    ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
                qpos = retargeting.retarget(ref_value)[retargeting_to_sapien]
            robot.set_qpos(qpos)
            if publish_ros_hand_qpos:
                qpos_by_name = dict(zip(sapien_joint_names, qpos))
                if use_revo2_geometric:
                    ros_hand_qpos = [
                        qpos_by_name.get(joint_name, 0.0)
                        for joint_name in revo2_controllable_joint_names
                    ]
                else:
                    ros_hand_qpos = qpos.tolist()
                ros_udp_socket.sendto(
                    json.dumps({"hand_qpos": ros_hand_qpos}).encode("utf-8"),
                    (ros_hand_qpos_host, ros_hand_qpos_port),
                )
            if forward_ros_wrist_pose and detector.latest_wrist_pose is not None:
                ros_wrist_pose = detector.latest_wrist_pose.astype(float).tolist()
                ros_wrist_packet = {"wrist_pose": ros_wrist_pose}
                if detector.latest_landmarks is not None:
                    ros_wrist_packet["landmarks"] = (
                        detector.latest_landmarks.astype(float).tolist()
                    )
                ros_udp_socket.sendto(
                    json.dumps(ros_wrist_packet).encode("utf-8"),
                    (ros_wrist_pose_host, ros_wrist_pose_port),
                )
                if (
                    debug_wrist_stream
                    and now - last_wrist_debug_log_time > debug_wrist_period
                ):
                    message = (
                        f"landmark_packets={detector.landmark_packets}, "
                        f"wrist_packets={detector.wrist_packets}, "
                        f"wrist_pose={np.round(np.array(ros_wrist_pose), 4).tolist()}, "
                        f"landmarks_forwarded={detector.latest_landmarks is not None}, "
                        f"udp_target={ros_wrist_pose_host}:{ros_wrist_pose_port}"
                    )
                    logger.info(f"META wrist input -> ROS UDP: {message}")
                    append_debug_line(wrist_debug_log_file, "META", message)
                    last_wrist_debug_log_time = now
            elif debug_wrist_stream and now - last_wrist_debug_log_time > debug_wrist_period:
                message = (
                    f"landmark_packets={detector.landmark_packets}, "
                    f"wrist_packets={detector.wrist_packets}, "
                    f"latest_wrist_pose={detector.latest_wrist_pose}"
                )
                logger.warning(f"META wrist input missing: {message}")
                append_debug_line(wrist_debug_log_file, "META", f"missing {message}")
                last_wrist_debug_log_time = now
            if follow_wrist and detector.latest_wrist_pose is not None:
                wrist = detector.latest_wrist_pose
                wrist_position = quest_position_to_sapien(wrist[:3])
                wrist_rotation = (
                    QUEST_TO_SAPIEN
                    @ quat_xyzw_to_matrix(wrist[3:7])
                    @ QUEST_TO_SAPIEN.T
                )
                if wrist_origin_position is None:
                    wrist_origin_position = wrist_position
                    wrist_origin_rotation = wrist_rotation
                    wrist_filtered_position = robot_base_pose.p.copy()
                    wrist_filtered_quat = robot_base_pose.q.copy()
                    logger.info(
                        "Calibrated wrist origin. Restart the script to recalibrate."
                    )

                target_position = robot_base_pose.p + (
                    wrist_position - wrist_origin_position
                ) * wrist_position_scale
                wrist_filtered_position = (
                    (1.0 - wrist_smoothing_alpha) * wrist_filtered_position
                    + wrist_smoothing_alpha * target_position
                )

                target_quat = robot_base_pose.q
                if follow_wrist_rotation:
                    relative_rotation = wrist_rotation @ wrist_origin_rotation.T
                    target_rotation = relative_rotation @ robot_base_rotation
                    target_quat = matrix_to_quat_wxyz(target_rotation)
                    if np.dot(wrist_filtered_quat, target_quat) < 0:
                        target_quat *= -1
                    wrist_filtered_quat = (
                        (1.0 - wrist_smoothing_alpha) * wrist_filtered_quat
                        + wrist_smoothing_alpha * target_quat
                    )
                    wrist_filtered_quat /= np.linalg.norm(wrist_filtered_quat)
                else:
                    wrist_filtered_quat = robot_base_pose.q

                robot.set_root_pose(
                    sapien.Pose(wrist_filtered_position, wrist_filtered_quat)
                )

        if now - last_viewer_render_time >= viewer_render_dt:
            viewer.render()
            last_viewer_render_time = now


def produce_frame(queue: multiprocessing.Queue, camera_path: Optional[str] = None):
    if camera_path is None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(camera_path)

    while cap.isOpened():
        success, image = cap.read()
        time.sleep(1 / 30.0)
        if not success:
            continue
        queue.put(image)


def main(
    robot_name: RobotName,
    retargeting_type: RetargetingType,
    hand_type: HandType,
    camera_path: Optional[str] = None,
    transport: Literal["udp", "tcp"] = "udp",
    host: Optional[str] = None,
    port: Optional[int] = None,
    udp_port: int = 9000,
    follow_wrist: bool = False,
    follow_wrist_rotation: bool = False,
    wrist_position_scale: float = 1.0,
    wrist_smoothing_alpha: float = 0.35,
    revo2_retargeting_mode: Literal["brainco", "dexpilot"] = "brainco",
    revo2_smoothing_alpha: float = 0.35,
    revo2_finger_gain: float = 1.0,
    revo2_thumb_gain: float = 1.0,
    revo2_thumb_metacarpal_gain: float = 1.0,
    revo2_thumb_metacarpal_mode: Literal["index_angle", "wrist_angle"] = "index_angle",
    publish_ros_hand_qpos: bool = False,
    ros_hand_qpos_host: str = "127.0.0.1",
    ros_hand_qpos_port: int = 5010,
    forward_ros_wrist_pose: bool = False,
    ros_wrist_pose_host: str = "127.0.0.1",
    ros_wrist_pose_port: int = 5005,
    debug_wrist_stream: bool = False,
    debug_wrist_period: float = 1.0,
    wrist_debug_log_file: Optional[str] = None,
    stream_poll_hz: float = 90.0,
    viewer_render_hz: float = 60.0,
):
    """
    Detects the human hand pose from a video and translates the human pose trajectory into a robot pose trajectory.

    Args:
        robot_name: The identifier for the robot. This should match one of the default supported robots.
        retargeting_type: The type of retargeting, each type corresponds to a different retargeting algorithm.
        hand_type: Specifies which hand is being tracked, either left or right.
            Please note that retargeting is specific to the same type of hand: a left robot hand can only be retargeted
            to another left robot hand, and the same applies for the right hand.
        camera_path: optional camera path. VR retargeting does not need a local camera by default.
        transport: socket transport for Quest hand landmarks.
        host: local host/IP to bind. Defaults to 0.0.0.0 for UDP and localhost for TCP.
        port: local port to bind. Defaults to 9000 for UDP and 8000 for TCP.
        udp_port: legacy UDP port option. Use --port for new commands.
        follow_wrist: also move the robot base with the streamed wrist pose.
        follow_wrist_rotation: also rotate the robot base with wrist orientation.
        wrist_position_scale: multiplier for wrist translation after calibration.
        wrist_smoothing_alpha: root follow smoothing. Larger values follow faster.
        revo2_retargeting_mode: Revo2 backend. "brainco" uses direct 6-DOF geometric retargeting;
            "dexpilot" uses the optimizer config.
        revo2_smoothing_alpha: smoothing for Revo2 joint commands. Larger values follow faster.
        revo2_finger_gain: gain for index/middle/ring/pinky curl in Revo2 brainco mode.
        revo2_thumb_gain: gain for thumb joints in Revo2 brainco mode.
        revo2_thumb_metacarpal_gain: additional gain for thumb abduction only.
        revo2_thumb_metacarpal_mode: thumb abduction mapping. "index_angle" avoids the wrist landmark;
            "wrist_angle" is the original BrainCo formula.
        publish_ros_hand_qpos: publish retargeted hand qpos as {"hand_qpos": [...]} over UDP.
        ros_hand_qpos_host: UDP host for dex_hand_control hand_qpos_node.
        ros_hand_qpos_port: UDP port for dex_hand_control hand_qpos_node.
        forward_ros_wrist_pose: forward latest Quest wrist pose as {"wrist_pose": [...]} over UDP.
        ros_wrist_pose_host: UDP host for quest_bridge wrist_twist_bridge.
        ros_wrist_pose_port: UDP port for quest_bridge wrist_twist_bridge.
        debug_wrist_stream: print Quest wrist input and forwarded ROS UDP wrist payload periodically.
        debug_wrist_period: seconds between wrist debug log lines.
        wrist_debug_log_file: optional text file path that receives wrist input/output debug lines.
        stream_poll_hz: processing/publish rate when VR input does not use a local camera.
        viewer_render_hz: maximum SAPIEN viewer refresh rate.
    """
    stream_port = port if port is not None else (udp_port if transport == "udp" else 8000)
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )

    queue = multiprocessing.Queue(maxsize=1000) if camera_path is not None else None
    producer_process = (
        multiprocessing.Process(target=produce_frame, args=(queue, camera_path))
        if queue is not None
        else None
    )
    consumer_process = multiprocessing.Process(
        target=start_retargeting,
        args=(
            queue,
            str(robot_dir),
            str(config_path),
            transport,
            host,
            stream_port,
            follow_wrist,
            follow_wrist_rotation,
            wrist_position_scale,
            wrist_smoothing_alpha,
            revo2_retargeting_mode,
            revo2_smoothing_alpha,
            revo2_finger_gain,
            revo2_thumb_gain,
            revo2_thumb_metacarpal_gain,
            revo2_thumb_metacarpal_mode,
            publish_ros_hand_qpos,
            ros_hand_qpos_host,
            ros_hand_qpos_port,
            forward_ros_wrist_pose,
            ros_wrist_pose_host,
            ros_wrist_pose_port,
            debug_wrist_stream,
            debug_wrist_period,
            wrist_debug_log_file,
            stream_poll_hz,
            viewer_render_hz,
        ),
    )

    if producer_process is not None:
        producer_process.start()
    consumer_process.start()

    if producer_process is not None:
        producer_process.join()
    consumer_process.join()
    time.sleep(5)

    print("done")


if __name__ == "__main__":
    tyro.cli(main)
