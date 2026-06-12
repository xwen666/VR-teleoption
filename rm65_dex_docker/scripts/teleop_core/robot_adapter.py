from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class ArmState:
    qpos: np.ndarray
    qvel: np.ndarray | None = None


@dataclass
class Pose:
    position: np.ndarray
    quat_xyzw: np.ndarray


@dataclass
class JointCommand:
    qpos: np.ndarray


class RobotAdapter(Protocol):
    """Minimal robot boundary for future MuJoCo/SDK backends.

    Existing v1 runners still use their current code paths. New robot ports can
    implement this protocol so the teleop controller does not depend on a
    specific vendor SDK.
    """

    def get_joint_state(self) -> ArmState:
        ...

    def get_ee_pose(self) -> Pose:
        ...

    def send_joint_command(self, command: JointCommand) -> None:
        ...

