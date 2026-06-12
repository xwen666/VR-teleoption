from __future__ import annotations

from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent.parent
RM65_DOCKER_ROOT = SCRIPTS_DIR.parent
WS_SRC = RM65_DOCKER_ROOT / "workspace/rm65_dex_ws/src"
QUEST_BRIDGE_SRC = WS_SRC / "quest_bridge"

CUBE_SIDE_MOUNT_CONFIG_PATH = WS_SRC / "quest_bridge/config/wrist_cube_side_mount.yaml"
DEFAULT_CONFIG_PATH = CUBE_SIDE_MOUNT_CONFIG_PATH

ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]
HAND_ACTUATED_JOINT_NAMES = [
    "hand_thumb_metacarpal_joint",
    "hand_thumb_proximal_joint",
    "hand_index_proximal_joint",
    "hand_middle_proximal_joint",
    "hand_ring_proximal_joint",
    "hand_pinky_proximal_joint",
]
HAND_MIMIC_RULES = {
    "hand_thumb_distal_joint": ("hand_thumb_proximal_joint", 1.0, 0.0),
    "hand_index_distal_joint": ("hand_index_proximal_joint", 1.155, 0.0),
    "hand_middle_distal_joint": ("hand_middle_proximal_joint", 1.155, 0.0),
    "hand_ring_distal_joint": ("hand_ring_proximal_joint", 1.155, 0.0),
    "hand_pinky_distal_joint": ("hand_pinky_proximal_joint", 1.155, 0.0),
}

RM65_JOINT_LIMIT_MIN = [-3.1, -2.268, -2.355, -3.1, -2.233, -6.28]
RM65_JOINT_LIMIT_MAX = [3.1, 2.268, 2.355, 3.1, 2.233, 6.28]
DEFAULT_WRIST_REGULARIZATION_JOINTS = [3, 4, 5]

