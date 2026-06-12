import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


class Revo2GeometricRetargeting:
    """BrainCo-style direct landmark-to-Revo2 retargeting.

    Revo2 only has six controllable DOF: thumb metacarpal, thumb proximal,
    and one proximal curl joint for each other finger. Distal joints follow
    those commands through URDF mimic rules.
    """

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    FINGER_LANDMARKS = {
        "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
        "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
        "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
        "pinky": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
    }

    def __init__(
        self,
        urdf_path: str | Path,
        hand_side: str,
        smoothing_alpha: float = 0.35,
        finger_gain: float = 1.0,
        thumb_gain: float = 1.0,
        thumb_metacarpal_gain: float = 1.0,
        thumb_metacarpal_mode: str = "index_angle",
    ):
        self.urdf_path = Path(urdf_path)
        self.hand_side = hand_side.lower()
        self.smoothing_alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
        self.finger_gain = finger_gain
        self.thumb_gain = thumb_gain
        self.thumb_metacarpal_gain = thumb_metacarpal_gain
        if thumb_metacarpal_mode not in {"index_angle", "wrist_angle"}:
            raise ValueError(
                "thumb_metacarpal_mode must be 'index_angle' or 'wrist_angle'"
            )
        self.thumb_metacarpal_mode = thumb_metacarpal_mode
        self.joint_limits = self._parse_urdf_joint_limits()
        self.mimic_joints = self._parse_mimic_joints()
        self.filtered_qpos = None

        self.joints = {
            "thumb_metacarpal": f"{self.hand_side}_thumb_metacarpal_joint",
            "thumb_proximal": f"{self.hand_side}_thumb_proximal_joint",
            "index_proximal": f"{self.hand_side}_index_proximal_joint",
            "middle_proximal": f"{self.hand_side}_middle_proximal_joint",
            "ring_proximal": f"{self.hand_side}_ring_proximal_joint",
            "pinky_proximal": f"{self.hand_side}_pinky_proximal_joint",
        }

    def _parse_urdf_joint_limits(self) -> Dict[str, Tuple[float, float]]:
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()
        limits = {}
        for joint in root.findall('.//joint[@type="revolute"]'):
            limit = joint.find("limit")
            if limit is None:
                continue
            limits[joint.get("name")] = (
                float(limit.get("lower", 0.0)),
                float(limit.get("upper", 0.0)),
            )
        return limits

    def _parse_mimic_joints(self) -> Dict[str, Dict[str, float | str]]:
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()
        mimic_joints = {}
        for joint in root.findall('.//joint[@type="revolute"]'):
            mimic = joint.find("mimic")
            if mimic is None:
                continue
            mimic_joints[joint.get("name")] = {
                "parent": mimic.get("joint"),
                "multiplier": float(mimic.get("multiplier", 1.0)),
                "offset": float(mimic.get("offset", 0.0)),
            }
        return mimic_joints

    @staticmethod
    def _angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        v1 = p1 - p2
        v2 = p3 - p2
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom < 1e-8:
            return 0.0
        cosine = np.dot(v1, v2) / denom
        return float(np.arccos(np.clip(cosine, -1.0, 1.0)))

    @staticmethod
    def _bend_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        return float(np.pi - Revo2GeometricRetargeting._angle(p1, p2, p3))

    def _apply_joint_limit(self, joint_name: str, angle: float) -> float:
        if joint_name not in self.joint_limits:
            return float(angle)
        lower, upper = self.joint_limits[joint_name]
        return float(np.clip(angle, lower, upper))

    def _apply_mimic_joints(self, joint_angles: Dict[str, float]) -> Dict[str, float]:
        full_angles = dict(joint_angles)
        for joint_name, mimic_info in self.mimic_joints.items():
            parent = mimic_info["parent"]
            if parent not in full_angles:
                continue
            angle = (
                full_angles[parent] * mimic_info["multiplier"]
                + mimic_info["offset"]
            )
            full_angles[joint_name] = self._apply_joint_limit(joint_name, angle)
        return full_angles

    def _thumb_metacarpal_angle(self, landmarks: np.ndarray) -> float:
        if self.thumb_metacarpal_mode == "wrist_angle":
            return (
                self._angle(
                    landmarks[self.WRIST],
                    landmarks[self.THUMB_CMC],
                    landmarks[self.THUMB_MCP],
                )
                - np.pi / 2
            )

        return self._angle(
            landmarks[self.INDEX_MCP],
            landmarks[self.THUMB_CMC],
            landmarks[self.THUMB_MCP],
        )

    def retarget(self, landmarks: np.ndarray) -> Dict[str, float]:
        if landmarks.shape != (21, 3):
            raise ValueError(f"Expected landmarks with shape (21, 3), got {landmarks.shape}")

        joint_angles = {}

        thumb_meta = self._thumb_metacarpal_angle(landmarks)
        thumb_prox = self._bend_angle(
            landmarks[self.THUMB_CMC],
            landmarks[self.THUMB_MCP],
            landmarks[self.THUMB_IP],
        )

        joint_angles[self.joints["thumb_metacarpal"]] = self._apply_joint_limit(
            self.joints["thumb_metacarpal"],
            thumb_meta * self.thumb_gain * self.thumb_metacarpal_gain,
        )
        joint_angles[self.joints["thumb_proximal"]] = self._apply_joint_limit(
            self.joints["thumb_proximal"], thumb_prox * self.thumb_gain
        )

        for finger_name, indices in self.FINGER_LANDMARKS.items():
            mcp, pip, dip, _ = indices
            joint_name = self.joints[f"{finger_name}_proximal"]
            curl = self._bend_angle(landmarks[mcp], landmarks[pip], landmarks[dip])
            joint_angles[joint_name] = self._apply_joint_limit(
                joint_name, curl * self.finger_gain
            )

        return self._apply_mimic_joints(joint_angles)

    def qpos_from_landmarks(
        self, landmarks: np.ndarray, sapien_joint_names: Iterable[str]
    ) -> np.ndarray:
        joint_angles = self.retarget(landmarks)
        qpos = np.array(
            [joint_angles.get(joint_name, 0.0) for joint_name in sapien_joint_names],
            dtype=float,
        )

        if self.filtered_qpos is None:
            self.filtered_qpos = qpos
        else:
            self.filtered_qpos = (
                (1.0 - self.smoothing_alpha) * self.filtered_qpos
                + self.smoothing_alpha * qpos
            )
        return self.filtered_qpos.copy()
