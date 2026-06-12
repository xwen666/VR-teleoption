# coding=utf-8

"""
眼在手上 hand-eye calibration

本脚本输出的是:
    相机坐标系 -> 机械臂末端坐标系
即:
    gT_c  或  cam -> gripper

不是 base -> camera

如果你想计算某一时刻的 base -> camera，需要再乘上该时刻的 base -> gripper:
    bT_c = bT_g @ gT_c
"""

import os
import re
import logging
from pathlib import Path

import yaml
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as SciRot

from libs.auxiliary import find_latest_data_folder
from libs.log_setting import CommonLog
from save_poses import poses_main

np.set_printoptions(precision=8, suppress=True)

logger_ = logging.getLogger(__name__)
logger_ = CommonLog(logger_)

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
EYE_HAND_DATA_DIR = BASE_DIR / "eye_hand_data"

with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as file:
    CONFIG = yaml.safe_load(file)

XX = CONFIG.get("checkerboard_args", {}).get("XX")
YY = CONFIG.get("checkerboard_args", {}).get("YY")
L = CONFIG.get("checkerboard_args", {}).get("L")

HAND_EYE_METHOD_MAP = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def get_latest_dataset_dir():
    latest_folder = find_latest_data_folder(str(EYE_HAND_DATA_DIR))
    if latest_folder is None:
        raise FileNotFoundError(f"在 {EYE_HAND_DATA_DIR} 下没有找到采集数据目录")
    return EYE_HAND_DATA_DIR / latest_folder


def numeric_stem(path_obj: Path):
    """
    取文件名中的数字部分用于排序和索引
    例如 1.jpg -> 1
         12.jpg -> 12
    """
    stem = path_obj.stem
    if stem.isdigit():
        return int(stem)

    match = re.search(r"(\d+)", stem)
    if match:
        return int(match.group(1))

    raise ValueError(f"图片文件名不包含数字，无法与位姿对应: {path_obj.name}")


def get_sorted_image_files(images_path: Path):
    image_files = [p for p in images_path.iterdir() if p.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp"]]
    image_files = sorted(image_files, key=numeric_stem)
    return image_files


def build_checkerboard_points(xx, yy, square_size):
    """
    棋盘格角点世界坐标
    """
    objp = np.zeros((xx * yy, 3), np.float32)
    objp[:, :2] = np.mgrid[0:xx, 0:yy].T.reshape(-1, 2)
    objp *= square_size
    return objp


def load_robot_tool_poses(csv_file: Path):
    """
    读取 poses_main 生成的 RobotToolPose.csv

    兼容以下几种格式：

    1) 3 x (4N)
       每 4 列是一个 3x4 位姿块 [R | t]
       自动补最后一行 [0, 0, 0, 1]

    2) 4 x (4N)
       每 4 列是一个完整 4x4 齐次矩阵

    3) (4N) x 4
       每 4 行是一个完整 4x4 齐次矩阵
    """
    tool_pose = np.loadtxt(str(csv_file), delimiter=",")

    if tool_pose.ndim == 1:
        raise ValueError(f"RobotToolPose.csv 读取结果是一维，当前 shape={tool_pose.shape}，请检查文件内容")

    rows, cols = tool_pose.shape
    poses = []

    # 情况 1: 3 x (4N)
    if rows == 3:
        if cols % 4 != 0:
            raise ValueError(f"RobotToolPose.csv 为 3 行时，列数应为 4 的整数倍，当前为 {cols}")

        num_poses = cols // 4
        for i in range(num_poses):
            block = tool_pose[:, 4 * i: 4 * i + 4]   # 3x4
            T = np.eye(4, dtype=np.float64)
            T[:3, :4] = block
            poses.append(T)

        return poses

    # 情况 2: 4 x (4N)
    if rows == 4 and cols % 4 == 0:
        num_poses = cols // 4
        for i in range(num_poses):
            block = tool_pose[:, 4 * i: 4 * i + 4].astype(np.float64)   # 4x4
            poses.append(block)

        return poses

    # 情况 3: (4N) x 4
    if cols == 4 and rows % 4 == 0:
        num_poses = rows // 4
        for i in range(num_poses):
            block = tool_pose[4 * i: 4 * i + 4, :].astype(np.float64)   # 4x4
            poses.append(block)

        return poses

    raise ValueError(
        f"无法识别 RobotToolPose.csv 的形状: {tool_pose.shape}。"
        f"目前仅支持 3x(4N)、4x(4N)、(4N)x4 这三种格式。"
    )


def load_precalibrated_intrinsics_from_config():
    """
    可选
    如果 config.yaml 中提前写好了相机内参和畸变，可以直接读出来
    例如:
    camera_intrinsics:
      use_precalibrated: true
      camera_matrix:
        - [fx, 0, cx]
        - [0, fy, cy]
        - [0, 0, 1]
      dist_coeffs: [k1, k2, p1, p2, k3]
    """
    cam_cfg = CONFIG.get("camera_intrinsics", {})
    use_pre = cam_cfg.get("use_precalibrated", False)

    if not use_pre:
        return None, None

    camera_matrix = np.array(cam_cfg.get("camera_matrix"), dtype=np.float64)
    dist_coeffs = np.array(cam_cfg.get("dist_coeffs", []), dtype=np.float64).reshape(-1, 1)

    if camera_matrix.shape != (3, 3):
        raise ValueError("config.yaml 中的 camera_matrix 形状必须是 3x3")

    return camera_matrix, dist_coeffs


def detect_corners_for_all_images(images_path: Path, xx, yy, objp):
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    object_points = []
    image_points = []
    valid_pose_indices = []
    used_image_files = []
    image_size = None

    image_files = get_sorted_image_files(images_path)
    if len(image_files) == 0:
        raise FileNotFoundError(f"目录下没有找到图片: {images_path}")

    logger_.info(f"共找到 {len(image_files)} 张图片")

    for img_path in image_files:
        logger_.info(f"读取图片: {img_path}")

        img = cv2.imread(str(img_path))
        if img is None:
            logger_.warning(f"图片读取失败，跳过: {img_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, (xx, yy), flags)

        if not ret:
            logger_.warning(f"棋盘角点检测失败，跳过: {img_path.name}")
            continue

        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        if corners2 is None:
            corners2 = corners

        object_points.append(objp.copy())
        image_points.append(corners2)

        img_index = numeric_stem(img_path) - 1
        valid_pose_indices.append(img_index)
        used_image_files.append(img_path.name)

        logger_.info(f"角点检测成功: {img_path.name}, 对应位姿索引: {img_index}")

    if image_size is None:
        raise RuntimeError("没有成功读取任何图片")

    return object_points, image_points, valid_pose_indices, used_image_files, image_size


def estimate_intrinsics(object_points, image_points, image_size):
    ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )

    logger_.info(f"相机重投影误差: {ret}")
    logger_.info(f"内参矩阵:\n{camera_matrix}")
    logger_.info(f"畸变系数:\n{dist_coeffs.ravel()}")

    return camera_matrix, dist_coeffs


def estimate_target_to_camera_poses(object_points, image_points, camera_matrix, dist_coeffs):
    """
    对每一张成功图片，单独求 target -> camera
    """
    rvecs = []
    tvecs = []
    R_target2cam = []
    t_target2cam = []

    for i, (objp, corners) in enumerate(zip(object_points, image_points)):
        ok, rvec, tvec = cv2.solvePnP(
            objp,
            corners,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not ok:
            raise RuntimeError(f"第 {i} 张图 solvePnP 失败")

        R_tc, _ = cv2.Rodrigues(rvec)

        rvecs.append(rvec)
        tvecs.append(tvec)
        R_target2cam.append(R_tc.astype(np.float64))
        t_target2cam.append(tvec.reshape(3, 1).astype(np.float64))

    return R_target2cam, t_target2cam, rvecs, tvecs


def sanity_check_rotation_matrix(R_mat):
    should_be_I = R_mat.T @ R_mat
    det_R = np.linalg.det(R_mat)
    ortho_err = np.linalg.norm(should_be_I - np.eye(3))

    logger_.info(f"R^T R 与 I 的误差: {ortho_err:.12f}")
    logger_.info(f"det(R): {det_R:.12f}")


def save_result_yaml(save_path: Path, R_cam2gripper, t_cam2gripper, quaternion_xyzw):
    data = {
        "hand_eye_result": {
            "meaning": "cam_to_gripper",
            "rotation_matrix": R_cam2gripper.tolist(),
            "translation_vector_m": t_cam2gripper.reshape(3).tolist(),
            "quaternion_xyzw": quaternion_xyzw.tolist(),
        }
    }

    with open(save_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def func():
    if XX is None or YY is None or L is None:
        raise ValueError("config.yaml 中 checkerboard_args 的 XX、YY、L 不能为空")

    images_path = get_latest_dataset_dir()
    file_path = images_path / "poses.txt"

    logger_.info(f"当前使用数据目录: {images_path}")
    logger_.info(f"poses.txt 路径: {file_path}")

    objp = build_checkerboard_points(XX, YY, L)

    # 1. 角点检测
    object_points, image_points, valid_pose_indices, used_image_files, image_size = detect_corners_for_all_images(
        images_path, XX, YY, objp
    )

    N = len(image_points)
    logger_.info(f"最终参与标定的图片数量: {N}")
    logger_.info(f"参与标定的图片: {used_image_files}")
    logger_.info(f"参与标定的位姿索引: {valid_pose_indices}")

    if N < 6:
        raise RuntimeError(f"有效图片太少，当前只有 {N} 张，建议至少 10 到 15 张")

    # 2. 读取机器人位姿并转换成 RobotToolPose.csv
    poses_main(str(file_path))
    csv_file = BASE_DIR / "RobotToolPose.csv"
    robot_poses_all = load_robot_tool_poses(csv_file)

    logger_.info(f"RobotToolPose.csv 中共有 {len(robot_poses_all)} 组位姿")

    if max(valid_pose_indices) >= len(robot_poses_all):
        raise IndexError(
            f"图片索引超出位姿数量范围: 最大图片索引={max(valid_pose_indices)}, 位姿总数={len(robot_poses_all)}"
        )

    # 3. 只提取成功图片对应的机器人位姿
    R_gripper2base = []
    t_gripper2base = []

    for idx in valid_pose_indices:
        T_bg = robot_poses_all[idx]

        R_gripper2base.append(T_bg[:3, :3].astype(np.float64))
        t_gripper2base.append(T_bg[:3, 3].reshape(3, 1).astype(np.float64))

    # 4. 获取相机内参
    camera_matrix, dist_coeffs = load_precalibrated_intrinsics_from_config()
    if camera_matrix is None:
        logger_.info("config.yaml 中未启用预标定内参，将使用当前图片估计内参")
        camera_matrix, dist_coeffs = estimate_intrinsics(object_points, image_points, image_size)
    else:
        logger_.info("使用 config.yaml 中的预标定相机内参")
        logger_.info(f"内参矩阵:\n{camera_matrix}")
        logger_.info(f"畸变系数:\n{dist_coeffs.ravel()}")

    # 5. 对每一张成功图片求 target -> camera
    R_target2cam, t_target2cam, _, _ = estimate_target_to_camera_poses(
        object_points, image_points, camera_matrix, dist_coeffs
    )

    # 6. 手眼标定
    method_name = CONFIG.get("handeye_args", {}).get("method", "TSAI").upper()
    method_flag = HAND_EYE_METHOD_MAP.get(method_name, cv2.CALIB_HAND_EYE_TSAI)

    logger_.info(f"使用手眼标定方法: {method_name}")

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base,
        t_gripper2base,
        R_target2cam,
        t_target2cam,
        method=method_flag
    )

    return R_cam2gripper, t_cam2gripper, camera_matrix, dist_coeffs, used_image_files, valid_pose_indices


if __name__ == "__main__":
    R_cam2gripper, t_cam2gripper, camera_matrix, dist_coeffs, used_image_files, valid_pose_indices = func()

    rotation = SciRot.from_matrix(R_cam2gripper)
    quaternion_xyzw = rotation.as_quat()

    logger_.info(f"旋转矩阵 R_cam2gripper:\n{R_cam2gripper}")
    logger_.info(f"平移向量 t_cam2gripper:\n{t_cam2gripper}")
    logger_.info(f"四元数 xyzw:\n{quaternion_xyzw}")

    sanity_check_rotation_matrix(R_cam2gripper)

    result_yaml_path = BASE_DIR / "handeye_result.yaml"
    save_result_yaml(result_yaml_path, R_cam2gripper, t_cam2gripper, quaternion_xyzw)
    logger_.info(f"结果已保存到: {result_yaml_path}")

    logger_.info("注意，这个结果是 cam -> gripper，不是 base -> camera")
    logger_.info(f"本次参与标定的图片: {used_image_files}")
    logger_.info(f"本次参与标定的位姿索引: {valid_pose_indices}")