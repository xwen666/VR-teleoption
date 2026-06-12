#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use the same RealMan SDK transport style as vr_teleop's RM65 real path:
# 50 Hz loop + rm_movej_canfd(..., follow=True).
export REALMAN_SDK_TELEOP_CONFIG="${REALMAN_SDK_TELEOP_CONFIG:-$ROOT_DIR/config/wrist_realman_arm_only.yaml}"
export REALMAN_ARM_COMMAND_MODE="${REALMAN_ARM_COMMAND_MODE:-movej_canfd_vrteleop}"
export REALMAN_CONTROL_DT="${REALMAN_CONTROL_DT:-0.02}"
export REALMAN_FEEDBACK_SOURCE="${REALMAN_FEEDBACK_SOURCE:-command}"
export REALMAN_SEED_CURRENT_AS_NOMINAL="${REALMAN_SEED_CURRENT_AS_NOMINAL:-1}"
export REALMAN_AUTO_PRESYNC="${REALMAN_AUTO_PRESYNC:-0}"

# Keep v1's DLS/Ruckig safety by default. Set REALMAN_TRAJECTORY_SMOOTHER=none
# if you want the transport layer to be even closer to vr_teleop's direct CANFD loop.
export REALMAN_TRAJECTORY_SMOOTHER="${REALMAN_TRAJECTORY_SMOOTHER:-config}"

# Hand following remains opt-in for this v1 arm-first test path.
export V1_ENABLE_HAND_FOLLOW="${V1_ENABLE_HAND_FOLLOW:-0}"
export V1_LAUNCH_MUJOCO_WITH_REAL="${V1_LAUNCH_MUJOCO_WITH_REAL:-0}"
export V1_ENABLE_UDP_FANOUT="${V1_ENABLE_UDP_FANOUT:-1}"

exec bash "$ROOT_DIR/run_realman_desktop_follow.sh" "$@"
