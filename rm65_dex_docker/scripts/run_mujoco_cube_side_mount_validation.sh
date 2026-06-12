#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${MUJOCO_PYTHON:-${MUJOCO_PYTHON_BIN:-/home/xwen/anaconda3/envs/VR/bin/python}}"
CONFIG_PATH="${MUJOCO_CONFIG:-$ROOT_DIR/workspace/rm65_dex_ws/src/quest_bridge/config/wrist_cube_side_mount.yaml}"
WRIST_PORT="${MUJOCO_WRIST_PORT:-5005}"
HAND_PORT="${MUJOCO_HAND_PORT:-5010}"
RUCKIG_SOURCE_DIR="${RUCKIG_SOURCE_DIR:-$ROOT_DIR/../ruckig}"
JOINT_SNAPSHOT_PATH="${MUJOCO_JOINT_SNAPSHOT_PATH:-/tmp/rm65_mujoco_arm_snapshot.json}"
INITIAL_SNAPSHOT_PATH="${MUJOCO_INITIAL_SNAPSHOT_PATH:-}"
REALMAN_SYNC_SCRIPT="${REALMAN_SYNC_SCRIPT:-$ROOT_DIR/scripts/sync_realman_to_mujoco_pose.py}"
REALMAN_SYNC_ENABLED="${REALMAN_AUTO_PRESYNC:-1}"
REALMAN_SYNC_BACKEND="${REALMAN_SYNC_BACKEND:-sdk}"
REALMAN_SYNC_SPEED="${REALMAN_SYNC_SPEED:-5}"
REALMAN_SYNC_SNAPSHOT_TIMEOUT="${REALMAN_SYNC_SNAPSHOT_TIMEOUT:-30}"
REALMAN_SYNC_PYTHON="${REALMAN_SYNC_PYTHON:-}"
REALMAN_SDK_PYTHON="${REALMAN_SDK_PYTHON:-/home/xwen/anaconda3/envs/sdk/bin/python}"
REALMAN_ROS_SETUP="${REALMAN_ROS_SETUP:-}"
REALMAN_ROS2_SETUP="${REALMAN_ROS2_SETUP:-}"
REALMAN_MOVEJ_TOPIC="${REALMAN_MOVEJ_TOPIC:-/rm_driver/movej_cmd}"
REALMAN_MOVE_RESULT_TOPIC="${REALMAN_MOVE_RESULT_TOPIC:-/rm_driver/movej_result}"
REALMAN_JOINT_STATE_TOPIC="${REALMAN_JOINT_STATE_TOPIC:-/joint_states}"
REALMAN_ARM_IP="${REALMAN_ARM_IP:-192.168.1.18}"
REALMAN_ARM_PORT="${REALMAN_ARM_PORT:-8080}"
REALMAN_WAIT_FOR_STATE_TIMEOUT="${REALMAN_WAIT_FOR_STATE_TIMEOUT:-10}"
REALMAN_MOTION_TIMEOUT="${REALMAN_MOTION_TIMEOUT:-90}"
REALMAN_SYNC_MAX_ABS_ERROR="${REALMAN_SYNC_MAX_ABS_ERROR:-0.02}"
REALMAN_SYNC_L2_ERROR="${REALMAN_SYNC_L2_ERROR:-0.04}"
REALMAN_SYNC_SETTLE_TIME="${REALMAN_SYNC_SETTLE_TIME:-1.0}"

PYTHON_BIN_DIR="$(dirname "$PYTHON_BIN")"
export PATH="$PYTHON_BIN_DIR:$PATH"

export MUJOCO_GL="${MUJOCO_GL:-glfw}"
export GLFW_PLATFORM="${GLFW_PLATFORM:-x11}"
export XDG_SESSION_TYPE="${XDG_SESSION_TYPE:-x11}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export GDK_BACKEND="${GDK_BACKEND:-x11}"
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-x11}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-glx}"
export DISPLAY="${DISPLAY:-:0}"
unset WAYLAND_DISPLAY

# Prefer the system C++ runtime to avoid Conda/driver mismatches when MuJoCo creates
# a GLFW window on desktop Linux.
SYSTEM_LIBSTDCPP="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
if [[ -f "$SYSTEM_LIBSTDCPP" ]]; then
  if [[ -n "${LD_PRELOAD:-}" ]]; then
    export LD_PRELOAD="$SYSTEM_LIBSTDCPP:$LD_PRELOAD"
  else
    export LD_PRELOAD="$SYSTEM_LIBSTDCPP"
  fi
fi

ensure_ruckig_python() {
  if (cd /tmp && "$PYTHON_BIN" - <<'PY') >/dev/null 2>&1
from ruckig import InputParameter, OutputParameter, Ruckig
PY
  then
    echo "Python ruckig is already importable."
    return
  fi

  if [[ ! -f "$RUCKIG_SOURCE_DIR/pyproject.toml" ]]; then
    echo "Ruckig source not found at $RUCKIG_SOURCE_DIR; MuJoCo will use fallback smoothing."
    return
  fi

  echo "Installing Python ruckig from $RUCKIG_SOURCE_DIR ..."
  "$PYTHON_BIN" -m pip install --upgrade --no-cache-dir scikit-build-core nanobind cmake ninja
  local conda_prefix
  local conda_gcc
  local conda_gxx
  conda_prefix="$(dirname "$PYTHON_BIN_DIR")"
  conda_gcc="$PYTHON_BIN_DIR/x86_64-conda-linux-gnu-gcc"
  conda_gxx="$PYTHON_BIN_DIR/x86_64-conda-linux-gnu-g++"
  if [[ -x "$conda_gcc" && -x "$conda_gxx" ]]; then
    CC="$conda_gcc" CXX="$conda_gxx" CMAKE_PREFIX_PATH="$conda_prefix" \
      "$PYTHON_BIN" -m pip install --upgrade --no-cache-dir "$RUCKIG_SOURCE_DIR"
  else
    "$PYTHON_BIN" -m pip install --upgrade --no-cache-dir "$RUCKIG_SOURCE_DIR"
  fi
}

want_validator_help() {
  local arg
  for arg in "$@"; do
    case "$arg" in
      -h|--help)
        return 0
        ;;
    esac
  done
  return 1
}

realman_sync_is_enabled() {
  case "${REALMAN_SYNC_ENABLED,,}" in
    0|false|no|off)
      return 1
      ;;
  esac
  return 0
}

wait_for_joint_snapshot() {
  local deadline
  deadline=$((SECONDS + REALMAN_SYNC_SNAPSHOT_TIMEOUT))
  while (( SECONDS < deadline )); do
    if [[ -f "$JOINT_SNAPSHOT_PATH" ]]; then
      if "$PYTHON_BIN" - "$JOINT_SNAPSHOT_PATH" <<'PY' >/dev/null 2>&1
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
arm_qpos = data.get("arm_qpos")
raise SystemExit(0 if isinstance(arm_qpos, list) and len(arm_qpos) >= 6 else 1)
PY
      then
        return 0
      fi
    fi

    if ! kill -0 "$MUJOCO_PID" 2>/dev/null; then
      echo "MuJoCo validator exited before writing a usable joint snapshot." >&2
      return 1
    fi
    sleep 0.2
  done

  echo "Timed out waiting for MuJoCo joint snapshot at $JOINT_SNAPSHOT_PATH." >&2
  return 1
}

source_realman_ros_env() {
  local ros2_setup="${REALMAN_ROS2_SETUP:-/opt/ros/humble/setup.bash}"
  local realman_setup="${REALMAN_ROS_SETUP:-$ROOT_DIR/../realman/ros_ruierman/rm_install/scripts/install/setup.bash}"

  if [[ -n "${REALMAN_ROS2_SETUP:-}" && ! -f "$ros2_setup" ]]; then
    echo "Configured ROS 2 setup script not found: $ros2_setup" >&2
    return 1
  fi
  if [[ -f "$ros2_setup" ]]; then
    # shellcheck disable=SC1090
    source "$ros2_setup"
  fi

  if [[ -n "${REALMAN_ROS_SETUP:-}" && ! -f "$realman_setup" ]]; then
    echo "Configured RealMan setup script not found: $realman_setup" >&2
    return 1
  fi
  if [[ -f "$realman_setup" ]]; then
    # shellcheck disable=SC1090
    source "$realman_setup"
  fi

  if ! command -v ros2 >/dev/null 2>&1; then
    echo "ros2 command is unavailable after sourcing the ROS environment." >&2
    return 1
  fi
}

resolve_realman_sync_python() {
  local candidate
  local backend="${REALMAN_SYNC_BACKEND,,}"
  local import_snippet

  case "$backend" in
    sdk)
      import_snippet=$'from Robotic_Arm.rm_robot_interface import RoboticArm\n'
      ;;
    ros)
      import_snippet=$'import rclpy\nfrom sensor_msgs.msg import JointState\nfrom std_msgs.msg import Bool\n'
      ;;
    auto)
      if [[ -n "$REALMAN_SYNC_PYTHON" ]]; then
        candidate="$REALMAN_SYNC_PYTHON"
        if "$candidate" - <<'PY' >/dev/null 2>&1
from Robotic_Arm.rm_robot_interface import RoboticArm
PY
        then
          printf '%s\n' "$candidate"
          return 0
        fi
        if "$candidate" - <<'PY' >/dev/null 2>&1
import rclpy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
PY
        then
          printf '%s\n' "$candidate"
          return 0
        fi
        echo "Configured REALMAN_SYNC_PYTHON cannot import SDK or ROS 2 packages: $candidate" >&2
        return 1
      fi

      for candidate in "$REALMAN_SDK_PYTHON" /usr/bin/python3 python3 "$PYTHON_BIN"; do
        if [[ -x "$candidate" ]] || command -v "$candidate" >/dev/null 2>&1; then
          if "$candidate" - <<'PY' >/dev/null 2>&1
from Robotic_Arm.rm_robot_interface import RoboticArm
PY
          then
            printf '%s\n' "$candidate"
            return 0
          fi
        fi
      done

      for candidate in /usr/bin/python3 python3 "$PYTHON_BIN"; do
        if [[ -x "$candidate" ]] || command -v "$candidate" >/dev/null 2>&1; then
          if "$candidate" - <<'PY' >/dev/null 2>&1
import rclpy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
PY
          then
            printf '%s\n' "$candidate"
            return 0
          fi
        fi
      done

      echo "Could not find a Python interpreter with either RealMan SDK or ROS 2 packages." >&2
      return 1
      ;;
    *)
      echo "Unsupported REALMAN_SYNC_BACKEND=$REALMAN_SYNC_BACKEND" >&2
      return 1
      ;;
  esac

  if [[ -n "$REALMAN_SYNC_PYTHON" ]]; then
    candidate="$REALMAN_SYNC_PYTHON"
    if ! "$candidate" - <<'PY' >/dev/null 2>&1
from Robotic_Arm.rm_robot_interface import RoboticArm
PY
    then
      if ! "$candidate" - <<PY >/dev/null 2>&1
$import_snippet
PY
      then
        case "$backend" in
          sdk)
            echo "Configured REALMAN_SYNC_PYTHON cannot import RealMan SDK packages: $candidate" >&2
            ;;
          ros)
            echo "Configured REALMAN_SYNC_PYTHON cannot import ROS 2 Python packages: $candidate" >&2
            ;;
        esac
        return 1
      fi
    fi
    if ! "$candidate" - <<PY >/dev/null 2>&1
$import_snippet
PY
    then
      case "$backend" in
        sdk)
          echo "Configured REALMAN_SYNC_PYTHON cannot import RealMan SDK packages: $candidate" >&2
          ;;
        ros)
          echo "Configured REALMAN_SYNC_PYTHON cannot import ROS 2 Python packages: $candidate" >&2
          ;;
      esac
      return 1
    fi
    printf '%s\n' "$candidate"
    return 0
  fi

  if [[ "$backend" == "sdk" ]]; then
    for candidate in "$REALMAN_SDK_PYTHON" /usr/bin/python3 python3 "$PYTHON_BIN"; do
      if [[ -x "$candidate" ]] || command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" - <<'PY' >/dev/null 2>&1
from Robotic_Arm.rm_robot_interface import RoboticArm
PY
        then
          printf '%s\n' "$candidate"
          return 0
        fi
      fi
    done

    echo "Could not find a Python interpreter with RealMan SDK packages for RealMan pre-sync." >&2
    return 1
  fi

  for candidate in /usr/bin/python3 python3 "$PYTHON_BIN"; do
    if [[ -x "$candidate" ]] || command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" - <<'PY' >/dev/null 2>&1
import rclpy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
PY
      then
        printf '%s\n' "$candidate"
        return 0
      fi
    fi
  done

  echo "Could not find a Python interpreter with ROS 2 packages for RealMan pre-sync." >&2
  return 1
}

run_realman_presync() {
  local sync_python

  if [[ ! -f "$REALMAN_SYNC_SCRIPT" ]]; then
    echo "RealMan pre-sync script not found: $REALMAN_SYNC_SCRIPT" >&2
    return 1
  fi

  case "${REALMAN_SYNC_BACKEND,,}" in
    ros)
      source_realman_ros_env
      ;;
    auto)
      if ! command -v ros2 >/dev/null 2>&1; then
        source_realman_ros_env || true
      fi
      ;;
  esac
  sync_python="$(resolve_realman_sync_python)"

  echo
  echo "Starting automatic RealMan pre-sync..."
  echo "  sync script: $REALMAN_SYNC_SCRIPT"
  echo "  sync python: $sync_python"
  echo "  backend: $REALMAN_SYNC_BACKEND"
  echo "  snapshot: $JOINT_SNAPSHOT_PATH"
  echo "  speed: $REALMAN_SYNC_SPEED"
  echo

  "$sync_python" "$REALMAN_SYNC_SCRIPT" \
    --backend "$REALMAN_SYNC_BACKEND" \
    --snapshot "$JOINT_SNAPSHOT_PATH" \
    --speed "$REALMAN_SYNC_SPEED" \
    --movej-topic "$REALMAN_MOVEJ_TOPIC" \
    --move-result-topic "$REALMAN_MOVE_RESULT_TOPIC" \
    --joint-state-topic "$REALMAN_JOINT_STATE_TOPIC" \
    --arm-ip "$REALMAN_ARM_IP" \
    --arm-port "$REALMAN_ARM_PORT" \
    --wait-for-state-timeout "$REALMAN_WAIT_FOR_STATE_TIMEOUT" \
    --motion-timeout "$REALMAN_MOTION_TIMEOUT" \
    --max-abs-error "$REALMAN_SYNC_MAX_ABS_ERROR" \
    --l2-error "$REALMAN_SYNC_L2_ERROR" \
    --settle-time "$REALMAN_SYNC_SETTLE_TIME"
}

ensure_ruckig_python

echo "MuJoCo cube-side-mount DLS validator"
echo "  python: $PYTHON_BIN"
echo "  config: $CONFIG_PATH"
echo "  ruckig source: $RUCKIG_SOURCE_DIR"
echo "  wrist UDP port: $WRIST_PORT"
echo "  hand UDP port: $HAND_PORT"
echo "  joint snapshot: $JOINT_SNAPSHOT_PATH"
if [[ -n "$INITIAL_SNAPSHOT_PATH" ]]; then
  echo "  initial snapshot: $INITIAL_SNAPSHOT_PATH"
fi
echo "  render backend: $MUJOCO_GL"
echo "  glfw platform: $GLFW_PLATFORM"
echo "  display: $DISPLAY"
echo
echo "Cube scene convention:"
echo "  +X: cube front"
echo "  +Y: cube left"
echo "  +Z: up"
echo "Robot mount convention:"
echo "  base z -> cube left normal (+Y)"
echo "  base x -> cube front (+X)"
echo
echo "Reminder:"
echo "  1. Keep Quest -> host retargeting running so wrist_pose/hand_qpos continue forwarding."
echo "  2. Stop ROS wrist bridge first if it is still listening on UDP 5005/5010."
echo

if want_validator_help "$@"; then
  EXTRA_ARGS=()
  if [[ -n "$INITIAL_SNAPSHOT_PATH" ]]; then
    EXTRA_ARGS+=(--initial-snapshot-path "$INITIAL_SNAPSHOT_PATH")
  fi
  exec "$PYTHON_BIN" \
    "$ROOT_DIR/scripts/mujoco_cube_side_mount_validate.py" \
    --config "$CONFIG_PATH" \
    --wrist-port "$WRIST_PORT" \
    --hand-port "$HAND_PORT" \
    --joint-snapshot-path "$JOINT_SNAPSHOT_PATH" \
    "${EXTRA_ARGS[@]}" \
    "$@"
fi

rm -f "$JOINT_SNAPSHOT_PATH"

EXTRA_ARGS=()
if [[ -n "$INITIAL_SNAPSHOT_PATH" ]]; then
  EXTRA_ARGS+=(--initial-snapshot-path "$INITIAL_SNAPSHOT_PATH")
fi

"$PYTHON_BIN" \
  "$ROOT_DIR/scripts/mujoco_cube_side_mount_validate.py" \
  --config "$CONFIG_PATH" \
  --wrist-port "$WRIST_PORT" \
  --hand-port "$HAND_PORT" \
  --joint-snapshot-path "$JOINT_SNAPSHOT_PATH" \
  "${EXTRA_ARGS[@]}" \
  "$@" &
MUJOCO_PID=$!

cleanup() {
  if [[ -n "${MUJOCO_PID:-}" ]] && kill -0 "$MUJOCO_PID" 2>/dev/null; then
    kill "$MUJOCO_PID" 2>/dev/null || true
    wait "$MUJOCO_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

if realman_sync_is_enabled; then
  wait_for_joint_snapshot
  run_realman_presync
else
  echo "Automatic RealMan pre-sync disabled via REALMAN_AUTO_PRESYNC=$REALMAN_SYNC_ENABLED."
fi

trap - EXIT
wait "$MUJOCO_PID"
