#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SDK_PYTHON="${REALMAN_SDK_PYTHON:-/home/xwen/anaconda3/envs/sdk/bin/python}"
CONFIG_PATH="${REALMAN_SDK_TELEOP_CONFIG:-$ROOT_DIR/workspace/rm65_dex_ws/src/quest_bridge/config/wrist_cube_side_mount.yaml}"
HAND_CONFIG_PATH="${REALMAN_SDK_HAND_CONFIG:-$ROOT_DIR/workspace/rm65_dex_ws/src/dex_hand_control/config/revo2_left_qpos.yaml}"
WRIST_PORT="${REALMAN_SDK_WRIST_PORT:-5005}"
HAND_PORT="${REALMAN_SDK_HAND_PORT:-5010}"
DISABLE_HAND_FOLLOW="${REALMAN_DISABLE_HAND_FOLLOW:-0}"
HAND_BACKEND="${REALMAN_HAND_BACKEND:-auto}"
BRAINCO_ECAT_MASTER_POS="${BRAINCO_ECAT_MASTER_POS:-0}"
BRAINCO_ECAT_SLAVE_POS="${BRAINCO_ECAT_SLAVE_POS:-0}"
BRAINCO_ECAT_CYCLE_NS="${BRAINCO_ECAT_CYCLE_NS:-1000000}"
JOINT_SNAPSHOT_PATH="${REALMAN_SDK_JOINT_SNAPSHOT_PATH:-/tmp/rm65_real_sdk_arm_snapshot.json}"
MUJOCO_SNAPSHOT_PATH="${MUJOCO_JOINT_SNAPSHOT_PATH:-/tmp/rm65_mujoco_arm_snapshot.json}"
PRESYNC_SNAPSHOT_PATH="${REALMAN_PRESYNC_SNAPSHOT_PATH:-$MUJOCO_SNAPSHOT_PATH}"
RUCKIG_SOURCE_DIR="${RUCKIG_SOURCE_DIR:-$ROOT_DIR/../ruckig}"
REALMAN_ARM_IP="${REALMAN_ARM_IP:-192.168.1.18}"
REALMAN_ARM_PORT="${REALMAN_ARM_PORT:-8080}"
REALMAN_ARM_COMMAND_MODE="${REALMAN_ARM_COMMAND_MODE:-movej_follow}"
REALMAN_CONTROL_DT="${REALMAN_CONTROL_DT:-0}"
REALMAN_FEEDBACK_SOURCE="${REALMAN_FEEDBACK_SOURCE:-sdk}"
REALMAN_TRAJECTORY_SMOOTHER="${REALMAN_TRAJECTORY_SMOOTHER:-config}"
REALMAN_SEED_CURRENT_AS_NOMINAL="${REALMAN_SEED_CURRENT_AS_NOMINAL:-0}"
REALMAN_AUTO_PRESYNC="${REALMAN_AUTO_PRESYNC:-auto}"
REALMAN_PRESYNC_SPEED="${REALMAN_PRESYNC_SPEED:-5}"
REALMAN_HAND_CONTROL_DT="${REALMAN_HAND_CONTROL_DT:-0.02}"
REALMAN_HAND_SPEED="${REALMAN_HAND_SPEED:--1}"
REALMAN_HAND_FORCE="${REALMAN_HAND_FORCE:--1}"

SDK_PYTHON_DIR="$(dirname "$SDK_PYTHON")"
export PATH="$SDK_PYTHON_DIR:$PATH"

SYSTEM_LIBSTDCPP="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
if [[ -f "$SYSTEM_LIBSTDCPP" ]]; then
  if [[ -n "${LD_PRELOAD:-}" ]]; then
    export LD_PRELOAD="$SYSTEM_LIBSTDCPP:$LD_PRELOAD"
  else
    export LD_PRELOAD="$SYSTEM_LIBSTDCPP"
  fi
fi

ensure_sdk_python_deps() {
  if (cd /tmp && "$SDK_PYTHON" - <<'PY') >/dev/null 2>&1
from ruckig import InputParameter, OutputParameter, Ruckig
import yaml
import numpy
PY
  then
    return
  fi

  if [[ ! -f "$RUCKIG_SOURCE_DIR/pyproject.toml" ]]; then
    echo "Ruckig source not found at $RUCKIG_SOURCE_DIR; cannot prepare SDK teleop dependencies." >&2
    return 1
  fi

  echo "Installing Python deps into SDK env ..."
  "$SDK_PYTHON" -m pip install --upgrade --no-cache-dir pyyaml numpy scikit-build-core nanobind cmake ninja
  local conda_prefix
  local conda_gcc
  local conda_gxx
  conda_prefix="$(dirname "$SDK_PYTHON_DIR")"
  conda_gcc="$SDK_PYTHON_DIR/x86_64-conda-linux-gnu-gcc"
  conda_gxx="$SDK_PYTHON_DIR/x86_64-conda-linux-gnu-g++"
  if [[ -x "$conda_gcc" && -x "$conda_gxx" ]]; then
    CC="$conda_gcc" CXX="$conda_gxx" CMAKE_PREFIX_PATH="$conda_prefix" \
      "$SDK_PYTHON" -m pip install --upgrade --no-cache-dir "$RUCKIG_SOURCE_DIR"
  else
    "$SDK_PYTHON" -m pip install --upgrade --no-cache-dir "$RUCKIG_SOURCE_DIR"
  fi
}

want_help() {
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

presync_mode() {
  printf '%s' "${REALMAN_AUTO_PRESYNC,,}"
}

HAND_ARGS=()
case "${DISABLE_HAND_FOLLOW,,}" in
  1|true|yes|on)
    HAND_ARGS+=(--disable-hand-follow)
    ;;
esac
if [[ "$HAND_PORT" == "0" ]]; then
  HAND_ARGS+=(--disable-hand-follow)
fi

SEED_ARGS=()
case "${REALMAN_SEED_CURRENT_AS_NOMINAL,,}" in
  1|true|yes|on)
    SEED_ARGS+=(--seed-current-as-nominal)
    ;;
esac

maybe_run_presync() {
  local mode
  mode="$(presync_mode)"

  case "$mode" in
    0|false|no|off)
      echo "Automatic pre-sync disabled via REALMAN_AUTO_PRESYNC=$REALMAN_AUTO_PRESYNC."
      return 0
      ;;
  esac

  if [[ ! -f "$PRESYNC_SNAPSHOT_PATH" ]]; then
    if [[ "$mode" == "auto" ]]; then
      echo "No pre-sync snapshot at $PRESYNC_SNAPSHOT_PATH; skip automatic pre-sync."
      return 0
    fi
    echo "Requested pre-sync, but snapshot is missing: $PRESYNC_SNAPSHOT_PATH" >&2
    return 1
  fi

  echo
  echo "Running RealMan pre-sync from startup snapshot ..."
  echo "  snapshot: $PRESYNC_SNAPSHOT_PATH"
  echo "  speed: $REALMAN_PRESYNC_SPEED"
  echo

  "$SDK_PYTHON" "$ROOT_DIR/scripts/sync_realman_to_mujoco_pose.py" \
    --backend sdk \
    --snapshot "$PRESYNC_SNAPSHOT_PATH" \
    --arm-ip "$REALMAN_ARM_IP" \
    --arm-port "$REALMAN_ARM_PORT" \
    --speed "$REALMAN_PRESYNC_SPEED"
}

maybe_warn_brainco_ethercat_env() {
  if [[ "$HAND_BACKEND" != "brainco_ethercat" ]]; then
    return 0
  fi
  if ! command -v ethercat >/dev/null 2>&1; then
    echo "Warning: 'ethercat' CLI not found in PATH. BrainCo EtherCAT backend usually expects the IgH EtherCAT master environment."
  fi
  if [[ ! -e "/dev/EtherCAT${BRAINCO_ECAT_MASTER_POS}" && ! -e "/dev/ethercat${BRAINCO_ECAT_MASTER_POS}" ]]; then
    echo "Warning: no local EtherCAT master device found at /dev/EtherCAT${BRAINCO_ECAT_MASTER_POS} or /dev/ethercat${BRAINCO_ECAT_MASTER_POS}."
  fi
}

echo "RealMan SDK hybrid teleop"
echo "  python: $SDK_PYTHON"
echo "  config: $CONFIG_PATH"
echo "  hand config: $HAND_CONFIG_PATH"
echo "  wrist UDP port: $WRIST_PORT"
echo "  hand UDP port: $HAND_PORT"
echo "  hand backend: $HAND_BACKEND"
if [[ "$HAND_BACKEND" == "brainco_ethercat" ]]; then
  echo "  brainco EtherCAT master/slave: ${BRAINCO_ECAT_MASTER_POS}/${BRAINCO_ECAT_SLAVE_POS}"
fi
echo "  real joint snapshot: $JOINT_SNAPSHOT_PATH"
echo "  mujoco snapshot: $MUJOCO_SNAPSHOT_PATH"
echo "  pre-sync snapshot: $PRESYNC_SNAPSHOT_PATH"
echo "  arm mode: $REALMAN_ARM_COMMAND_MODE"
echo "  control dt: $REALMAN_CONTROL_DT"
echo "  feedback source: $REALMAN_FEEDBACK_SOURCE"
echo "  trajectory smoother: $REALMAN_TRAJECTORY_SMOOTHER"
echo "  seed current as nominal: $REALMAN_SEED_CURRENT_AS_NOMINAL"
echo "  hand control dt: $REALMAN_HAND_CONTROL_DT"
echo
echo "Reminder:"
echo "  1. Keep Quest -> host retargeting running so wrist_pose/hand_qpos continue forwarding."
echo "  2. Do not run ROS wrist bridge or another SDK teleop process on the same ports."
echo

if want_help "$@"; then
  exec "$SDK_PYTHON" \
    "$ROOT_DIR/scripts/realman_sdk_hybrid_teleop.py" \
    --config "$CONFIG_PATH" \
    --hand-config "$HAND_CONFIG_PATH" \
    --wrist-port "$WRIST_PORT" \
    --hand-port "$HAND_PORT" \
    --joint-snapshot-path "$JOINT_SNAPSHOT_PATH" \
    --startup-snapshot-path "$PRESYNC_SNAPSHOT_PATH" \
    --arm-ip "$REALMAN_ARM_IP" \
    --arm-port "$REALMAN_ARM_PORT" \
    --arm-command-mode "$REALMAN_ARM_COMMAND_MODE" \
    --control-dt "$REALMAN_CONTROL_DT" \
    --feedback-source "$REALMAN_FEEDBACK_SOURCE" \
    --trajectory-smoother "$REALMAN_TRAJECTORY_SMOOTHER" \
    --hand-control-dt "$REALMAN_HAND_CONTROL_DT" \
    --hand-backend "$HAND_BACKEND" \
    --brainco-master-pos "$BRAINCO_ECAT_MASTER_POS" \
    --brainco-slave-pos "$BRAINCO_ECAT_SLAVE_POS" \
    --brainco-cycle-ns "$BRAINCO_ECAT_CYCLE_NS" \
    --hand-speed "$REALMAN_HAND_SPEED" \
    --hand-force "$REALMAN_HAND_FORCE" \
    "${SEED_ARGS[@]}" \
    "${HAND_ARGS[@]}" \
    "$@"
fi

ensure_sdk_python_deps

maybe_warn_brainco_ethercat_env

maybe_run_presync

exec "$SDK_PYTHON" \
  "$ROOT_DIR/scripts/realman_sdk_hybrid_teleop.py" \
  --config "$CONFIG_PATH" \
  --hand-config "$HAND_CONFIG_PATH" \
  --wrist-port "$WRIST_PORT" \
  --hand-port "$HAND_PORT" \
  --joint-snapshot-path "$JOINT_SNAPSHOT_PATH" \
  --startup-snapshot-path "$PRESYNC_SNAPSHOT_PATH" \
  --arm-ip "$REALMAN_ARM_IP" \
  --arm-port "$REALMAN_ARM_PORT" \
  --arm-command-mode "$REALMAN_ARM_COMMAND_MODE" \
  --control-dt "$REALMAN_CONTROL_DT" \
  --feedback-source "$REALMAN_FEEDBACK_SOURCE" \
  --trajectory-smoother "$REALMAN_TRAJECTORY_SMOOTHER" \
  --hand-control-dt "$REALMAN_HAND_CONTROL_DT" \
  --hand-backend "$HAND_BACKEND" \
  --brainco-master-pos "$BRAINCO_ECAT_MASTER_POS" \
  --brainco-slave-pos "$BRAINCO_ECAT_SLAVE_POS" \
  --brainco-cycle-ns "$BRAINCO_ECAT_CYCLE_NS" \
  --hand-speed "$REALMAN_HAND_SPEED" \
  --hand-force "$REALMAN_HAND_FORCE" \
  "${SEED_ARGS[@]}" \
  "${HAND_ARGS[@]}" \
  "$@"
