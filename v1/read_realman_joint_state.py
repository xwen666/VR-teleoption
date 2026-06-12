#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
VR_ROOT = SCRIPT_PATH.parent.parent
RM65_DOCKER_ROOT = VR_ROOT / "rm65_dex_docker"
RM65_SCRIPTS = RM65_DOCKER_ROOT / "scripts"
if str(RM65_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(RM65_SCRIPTS))

from realman_sdk_common import (  # noqa: E402
    DEFAULT_ARM_IP,
    DEFAULT_ARM_PORT,
    RealmanSdkClient,
    can_import_sdk,
    deg_to_rad,
    wait_for_joint_state_sdk,
)


ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]
DEFAULT_SNAPSHOT_PATH = Path("/tmp/rm65_real_sdk_arm_snapshot.json")
DEFAULT_SDK_PYTHON = Path("/home/xwen/anaconda3/envs/sdk/bin/python")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read the current RealMan arm joint state through the Python SDK and write "
            "a v1 MuJoCo-compatible snapshot JSON."
        )
    )
    parser.add_argument("--arm-ip", type=str, default=DEFAULT_ARM_IP, help="RealMan controller IP.")
    parser.add_argument("--arm-port", type=int, default=DEFAULT_ARM_PORT, help="RealMan controller TCP port.")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=DEFAULT_SNAPSHOT_PATH,
        help="Output JSON snapshot path for v1 MuJoCo initialization.",
    )
    parser.add_argument(
        "--wait-for-state-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for the first valid SDK arm state.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Also print the captured snapshot JSON to stdout.",
    )
    return parser.parse_args()


def build_snapshot_payload(arm_qpos_rad: list[float], arm_ip: str, arm_port: int) -> dict:
    now_wall = time.time()
    now_monotonic = time.monotonic()
    return {
        "timestamp": now_wall,
        "monotonic_time": now_monotonic,
        "arm_joint_names": list(ARM_JOINT_NAMES),
        "arm_qpos": [float(value) for value in arm_qpos_rad[:6]],
        "hand_qpos": [0.0] * 6,
        "source": "v1_read_realman_joint_state",
        "backend": "sdk",
        "arm_ip": arm_ip,
        "arm_port": int(arm_port),
    }


def write_snapshot(snapshot_path: Path, payload: dict) -> None:
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = snapshot_path.with_suffix(snapshot_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(snapshot_path)


def maybe_reexec_into_sdk_python() -> int | None:
    if can_import_sdk():
        return None

    configured_python = os.environ.get("REALMAN_SDK_PYTHON", "").strip()
    candidate = Path(configured_python) if configured_python else DEFAULT_SDK_PYTHON
    try:
        same_binary = candidate.resolve() == Path(sys.executable).resolve()
    except Exception:
        same_binary = str(candidate) == sys.executable

    if same_binary or not candidate.is_file():
        return None

    command = [str(candidate), __file__, *sys.argv[1:]]
    return subprocess.run(command, check=False).returncode


def main() -> int:
    reexec_code = maybe_reexec_into_sdk_python()
    if reexec_code is not None:
        return int(reexec_code)

    args = parse_args()
    if not can_import_sdk():
        print(
            "RealMan SDK is not importable in this Python environment. "
            "Run this script with the SDK conda env or set REALMAN_SDK_PYTHON accordingly.",
            file=sys.stderr,
        )
        return 2

    client = RealmanSdkClient(args.arm_ip, args.arm_port)
    client.connect()
    try:
        current_deg = wait_for_joint_state_sdk(client, args.wait_for_state_timeout)
        current_rad = deg_to_rad(current_deg)
        payload = build_snapshot_payload(current_rad, args.arm_ip, args.arm_port)
        write_snapshot(args.snapshot, payload)

        print("Captured current RealMan joint state for v1:")
        print("  arm rad: " + ", ".join(f"{value:.4f}" for value in payload["arm_qpos"]))
        print("  arm deg: " + ", ".join(f"{value:.2f}" for value in current_deg[:6]))
        print(f"  snapshot: {args.snapshot}")
        if args.print_json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"Failed to capture current RealMan joint state: {exc}", file=sys.stderr)
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
