from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import sys
from typing import Optional

import numpy as np


def _is_local_ruckig_source_path(path_entry: str) -> bool:
    root = Path(path_entry or os.getcwd())
    candidate = root / "ruckig"
    return (
        (candidate / "pyproject.toml").is_file()
        and (candidate / "src" / "wrapper" / "python.cpp").is_file()
    )


def _import_ruckig_symbols():
    try:
        module = importlib.import_module("ruckig")
        return (
            module.InputParameter,
            module.OutputParameter,
            module.Result,
            module.Ruckig,
        )
    except Exception:
        pass

    original_path = list(sys.path)
    removed_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if name == "ruckig" or name.startswith("ruckig.")
    }
    for name in removed_modules:
        sys.modules.pop(name, None)

    try:
        sys.path = [
            path_entry
            for path_entry in original_path
            if not _is_local_ruckig_source_path(path_entry)
        ]
        module = importlib.import_module("ruckig")
        return (
            module.InputParameter,
            module.OutputParameter,
            module.Result,
            module.Ruckig,
        )
    except Exception:
        for name, module in removed_modules.items():
            sys.modules.setdefault(name, module)
        raise
    finally:
        sys.path = original_path


try:  # pragma: no cover - optional dependency
    InputParameter, OutputParameter, Result, Ruckig = _import_ruckig_symbols()

    HAS_RUCKIG = True
except Exception:  # pragma: no cover - optional dependency
    InputParameter = None  # type: ignore
    OutputParameter = None  # type: ignore
    Result = None  # type: ignore
    Ruckig = None  # type: ignore
    HAS_RUCKIG = False


@dataclass
class SmoothingResult:
    q_cmd: np.ndarray
    dq_cmd: np.ndarray
    mode_used: str


class JointTrajectorySmoother:
    def __init__(
        self,
        dof: int,
        dt: float,
        mode: str = "auto",
        max_velocity: Optional[np.ndarray] = None,
        max_acceleration: Optional[np.ndarray] = None,
        max_jerk: Optional[np.ndarray] = None,
    ) -> None:
        self.dof = int(dof)
        self.dt = float(max(dt, 1e-4))
        self.requested_mode = mode
        self.max_velocity = (
            np.array(max_velocity, dtype=np.float32)
            if max_velocity is not None
            else np.full(self.dof, 0.3, dtype=np.float32)
        )
        self.max_acceleration = (
            np.array(max_acceleration, dtype=np.float32)
            if max_acceleration is not None
            else np.full(self.dof, 0.6, dtype=np.float32)
        )
        self.max_jerk = (
            np.array(max_jerk, dtype=np.float32)
            if max_jerk is not None
            else np.full(self.dof, 2.5, dtype=np.float32)
        )

        if self.requested_mode == "ruckig" and not HAS_RUCKIG:
            self.mode = "accel_limited"
        elif self.requested_mode == "auto":
            self.mode = "ruckig" if HAS_RUCKIG else "accel_limited"
        else:
            self.mode = self.requested_mode

        self.otg = Ruckig(self.dof, self.dt) if self.mode == "ruckig" else None
        self.input_param = InputParameter(self.dof) if self.mode == "ruckig" else None
        self.output_param = OutputParameter(self.dof) if self.mode == "ruckig" else None

        self.last_q_cmd: Optional[np.ndarray] = None
        self.last_dq_cmd: Optional[np.ndarray] = None
        self.last_ddq_cmd: Optional[np.ndarray] = None

    def reset(self, q: np.ndarray, dq: Optional[np.ndarray] = None, ddq: Optional[np.ndarray] = None) -> None:
        self.last_q_cmd = np.array(q, dtype=np.float32).copy()
        self.last_dq_cmd = (
            np.zeros(self.dof, dtype=np.float32)
            if dq is None
            else np.array(dq, dtype=np.float32).copy()
        )
        self.last_ddq_cmd = (
            np.zeros(self.dof, dtype=np.float32)
            if ddq is None
            else np.array(ddq, dtype=np.float32).copy()
        )

    def _fallback_step(
        self,
        current_q: np.ndarray,
        current_dq: np.ndarray,
        target_q: np.ndarray,
    ) -> SmoothingResult:
        max_dq = self.max_velocity * self.dt
        desired_delta = target_q - current_q
        velocity_limited_delta = np.clip(desired_delta, -max_dq, max_dq)
        desired_dq = velocity_limited_delta / self.dt

        max_ddq = self.max_acceleration * self.dt
        dq_delta = desired_dq - current_dq
        dq_cmd = current_dq + np.clip(dq_delta, -max_ddq, max_ddq)
        q_cmd = current_q + dq_cmd * self.dt

        self.last_q_cmd = q_cmd.astype(np.float32)
        self.last_dq_cmd = dq_cmd.astype(np.float32)
        self.last_ddq_cmd = ((dq_cmd - current_dq) / self.dt).astype(np.float32)
        return SmoothingResult(
            q_cmd=self.last_q_cmd.copy(),
            dq_cmd=self.last_dq_cmd.copy(),
            mode_used="accel_limited",
        )

    def step(
        self,
        current_q: np.ndarray,
        current_dq: np.ndarray,
        target_q: np.ndarray,
    ) -> SmoothingResult:
        current_q = np.array(current_q, dtype=np.float32)
        current_dq = np.array(current_dq, dtype=np.float32)
        target_q = np.array(target_q, dtype=np.float32)

        if self.mode == "none":
            self.last_q_cmd = target_q.copy()
            self.last_dq_cmd = np.zeros(self.dof, dtype=np.float32)
            self.last_ddq_cmd = np.zeros(self.dof, dtype=np.float32)
            return SmoothingResult(
                q_cmd=target_q.copy(),
                dq_cmd=np.zeros(self.dof, dtype=np.float32),
                mode_used="none",
            )

        if self.mode != "ruckig" or self.otg is None or self.input_param is None or self.output_param is None:
            return self._fallback_step(current_q, current_dq, target_q)

        self.input_param.current_position = current_q.tolist()
        self.input_param.current_velocity = current_dq.tolist()
        self.input_param.current_acceleration = (
            np.zeros(self.dof, dtype=np.float32)
            if self.last_ddq_cmd is None
            else self.last_ddq_cmd.astype(np.float32)
        ).tolist()
        self.input_param.target_position = target_q.tolist()
        self.input_param.target_velocity = [0.0] * self.dof
        self.input_param.target_acceleration = [0.0] * self.dof
        self.input_param.max_velocity = self.max_velocity.astype(np.float32).tolist()
        self.input_param.max_acceleration = self.max_acceleration.astype(np.float32).tolist()
        self.input_param.max_jerk = self.max_jerk.astype(np.float32).tolist()

        result = self.otg.update(self.input_param, self.output_param)
        if Result is not None and int(result) < 0:  # pragma: no cover - rare error path
            return self._fallback_step(current_q, current_dq, target_q)

        q_cmd = np.array(self.output_param.new_position, dtype=np.float32)
        dq_cmd = np.array(self.output_param.new_velocity, dtype=np.float32)
        ddq_cmd = np.array(self.output_param.new_acceleration, dtype=np.float32)

        self.last_q_cmd = q_cmd.copy()
        self.last_dq_cmd = dq_cmd.copy()
        self.last_ddq_cmd = ddq_cmd.copy()
        return SmoothingResult(
            q_cmd=q_cmd,
            dq_cmd=dq_cmd,
            mode_used="ruckig",
        )
