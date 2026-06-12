from __future__ import annotations

import ctypes
import math
import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


ARM_DOF = 7
RM_MODEL_RM_65_E = 0
RM_MODEL_RM_75_E = 1
RM_MODEL_RM_B_E = 0


class RmQuat(ctypes.Structure):
    _fields_ = [
        ("w", ctypes.c_float),
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
    ]


class RmPosition(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
    ]


class RmEuler(ctypes.Structure):
    _fields_ = [
        ("rx", ctypes.c_float),
        ("ry", ctypes.c_float),
        ("rz", ctypes.c_float),
    ]


class RmPose(ctypes.Structure):
    _fields_ = [
        ("position", RmPosition),
        ("quaternion", RmQuat),
        ("euler", RmEuler),
    ]


class RmInverseKinematicsParams(ctypes.Structure):
    _fields_ = [
        ("q_in", ctypes.c_float * ARM_DOF),
        ("q_pose", RmPose),
        ("flag", ctypes.c_uint8),
    ]


class RealManIK:
    """Small ctypes wrapper around RealMan's official offline IK library."""

    MODEL_NAMES = {
        "rm65": RM_MODEL_RM_65_E,
        "rm_65": RM_MODEL_RM_65_E,
        "rm75": RM_MODEL_RM_75_E,
        "rm_75": RM_MODEL_RM_75_E,
    }

    def __init__(
        self,
        lib_path: str = "",
        robot_model: str = "rm65",
        force_type: int = RM_MODEL_RM_B_E,
        traversal_mode: bool = False,
    ):
        self.lib_path = self.resolve_lib_path(lib_path)
        self.lib = ctypes.CDLL(str(self.lib_path))
        self.configure_symbols()

        model = self.MODEL_NAMES.get(robot_model.lower())
        if model is None:
            raise ValueError(f"Unsupported RealMan robot_model: {robot_model}")

        self.lib.rm_algo_init_sys_data(model, int(force_type))
        self.lib.rm_algo_set_redundant_parameter_traversal_mode(bool(traversal_mode))

    @staticmethod
    def resolve_lib_path(lib_path: str) -> Path:
        candidates = []
        if lib_path:
            candidates.append(Path(lib_path))
        env_path = os.environ.get("REALMAN_API_LIB")
        if env_path:
            candidates.append(Path(env_path))

        package_dir = Path(__file__).resolve().parent
        candidates.extend(
            [
                package_dir / "libapi_cpp.so",
                Path("/realman/ros_ruierman/rm_driver/lib/linux_x86_c++_v1.1.3/libapi_cpp.so"),
                Path("/realman/ros_ruierman/rm_driver/lib/libapi_cpp.so"),
                Path("/home/xwen/vr/realman/ros_ruierman/rm_driver/lib/linux_x86_c++_v1.1.3/libapi_cpp.so"),
                Path("/home/xwen/vr/realman/ros_ruierman/rm_driver/lib/libapi_cpp.so"),
            ]
        )

        for candidate in candidates:
            if candidate.exists():
                return candidate

        tried = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Cannot find RealMan libapi_cpp.so. Tried: {tried}")

    def configure_symbols(self):
        self.lib.rm_algo_version.restype = ctypes.c_char_p

        self.lib.rm_algo_init_sys_data.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.rm_algo_init_sys_data.restype = None

        self.lib.rm_algo_set_redundant_parameter_traversal_mode.argtypes = [ctypes.c_bool]
        self.lib.rm_algo_set_redundant_parameter_traversal_mode.restype = None

        self.lib.rm_algo_inverse_kinematics.argtypes = [
            ctypes.c_void_p,
            RmInverseKinematicsParams,
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.rm_algo_inverse_kinematics.restype = ctypes.c_int

        self.lib.rm_algo_forward_kinematics.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.rm_algo_forward_kinematics.restype = RmPose

        self.lib.rm_algo_ikine_check_joint_position_limit.argtypes = [
            ctypes.POINTER(ctypes.c_float)
        ]
        self.lib.rm_algo_ikine_check_joint_position_limit.restype = ctypes.c_int

        self.lib.rm_algo_ikine_check_joint_velocity_limit.argtypes = [
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.rm_algo_ikine_check_joint_velocity_limit.restype = ctypes.c_int

        self.lib.rm_algo_kin_robot_singularity_analyse.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.rm_algo_kin_robot_singularity_analyse.restype = ctypes.c_int

    def version(self) -> str:
        raw = self.lib.rm_algo_version()
        return raw.decode("utf-8", errors="replace") if raw else "unknown"

    def inverse(
        self,
        q_seed_rad: Iterable[float],
        target_position: np.ndarray,
        target_quat_xyzw: np.ndarray,
    ) -> tuple[int, np.ndarray]:
        q_seed_deg = self.rad_to_realman_deg(q_seed_rad)
        quat = self.normalized_quat_xyzw(target_quat_xyzw)

        params = RmInverseKinematicsParams()
        for index in range(ARM_DOF):
            params.q_in[index] = float(q_seed_deg[index])
        params.q_pose.position = RmPosition(
            float(target_position[0]),
            float(target_position[1]),
            float(target_position[2]),
        )
        params.q_pose.quaternion = RmQuat(
            float(quat[3]),
            float(quat[0]),
            float(quat[1]),
            float(quat[2]),
        )
        params.q_pose.euler = RmEuler(0.0, 0.0, 0.0)
        params.flag = 0

        q_out = (ctypes.c_float * ARM_DOF)()
        ret = self.lib.rm_algo_inverse_kinematics(None, params, q_out)
        q_deg = np.array([q_out[i] for i in range(6)], dtype=np.float32)
        return int(ret), np.deg2rad(q_deg)

    def forward(self, q_rad: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
        q_deg = self.rad_to_realman_deg(q_rad)
        q_array = (ctypes.c_float * ARM_DOF)(*q_deg)
        pose = self.lib.rm_algo_forward_kinematics(None, q_array)
        position = np.array(
            [pose.position.x, pose.position.y, pose.position.z],
            dtype=np.float32,
        )
        quat_xyzw = np.array(
            [pose.quaternion.x, pose.quaternion.y, pose.quaternion.z, pose.quaternion.w],
            dtype=np.float32,
        )
        return position, self.normalized_quat_xyzw(quat_xyzw)

    def check_position_limit(self, q_rad: Iterable[float]) -> int:
        q_deg = self.rad_to_realman_deg(q_rad)
        q_array = (ctypes.c_float * ARM_DOF)(*q_deg)
        return int(self.lib.rm_algo_ikine_check_joint_position_limit(q_array))

    def check_velocity_limit(
        self,
        dt: float,
        q_ref_rad: Iterable[float],
        q_solve_rad: Iterable[float],
    ) -> int:
        q_ref_deg = self.rad_to_realman_deg(q_ref_rad)
        q_solve_deg = self.rad_to_realman_deg(q_solve_rad)
        q_ref = (ctypes.c_float * ARM_DOF)(*q_ref_deg)
        q_solve = (ctypes.c_float * ARM_DOF)(*q_solve_deg)
        return int(self.lib.rm_algo_ikine_check_joint_velocity_limit(float(dt), q_ref, q_solve))

    def singularity(self, q_rad: Iterable[float]) -> tuple[int, Optional[float]]:
        q_deg = self.rad_to_realman_deg(q_rad)
        q_array = (ctypes.c_float * ARM_DOF)(*q_deg)
        distance = ctypes.c_float()
        ret = int(self.lib.rm_algo_kin_robot_singularity_analyse(q_array, ctypes.byref(distance)))
        return ret, float(distance.value)

    @staticmethod
    def rad_to_realman_deg(q_rad: Iterable[float]) -> list[float]:
        values = list(q_rad)
        if len(values) < 6:
            raise ValueError("RealMan IK needs at least 6 joint values")
        degrees = [math.degrees(float(value)) for value in values[:6]]
        degrees.append(0.0)
        return degrees

    @staticmethod
    def normalized_quat_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
        quat = np.array(quat_xyzw, dtype=np.float32)
        norm = float(np.linalg.norm(quat))
        if norm < 1e-9:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return quat / norm
