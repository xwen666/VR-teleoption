#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RM65_DOCKER_ROOT="$(cd "${ROOT_DIR}/../rm65_dex_docker" && pwd)"
V1_ENABLE_HAND_FOLLOW="${V1_ENABLE_HAND_FOLLOW:-0}"
V1_LAUNCH_MUJOCO_WITH_REAL="${V1_LAUNCH_MUJOCO_WITH_REAL:-1}"
V1_ENABLE_UDP_FANOUT="${V1_ENABLE_UDP_FANOUT:-1}"
V1_RELAY_PYTHON="${V1_RELAY_PYTHON:-/home/xwen/anaconda3/envs/VR/bin/python}"
V1_RELAY_LISTEN_WRIST_PORT="${V1_RELAY_LISTEN_WRIST_PORT:-5005}"
V1_RELAY_LISTEN_HAND_PORT="${V1_RELAY_LISTEN_HAND_PORT:-5010}"
V1_REAL_WRIST_PORT="${REALMAN_SDK_WRIST_PORT:-${V1_REAL_WRIST_PORT:-5105}}"
V1_REAL_HAND_PORT="${REALMAN_SDK_HAND_PORT:-${V1_REAL_HAND_PORT:-5110}}"
V1_MUJOCO_WRIST_PORT="${V1_MUJOCO_WRIST_PORT:-5205}"
V1_MUJOCO_HAND_PORT="${V1_MUJOCO_HAND_PORT:-5210}"

export REALMAN_SDK_TELEOP_CONFIG="${REALMAN_SDK_TELEOP_CONFIG:-$ROOT_DIR/config/wrist_realman_arm_only.yaml}"
export REALMAN_SDK_JOINT_SNAPSHOT_PATH="${REALMAN_SDK_JOINT_SNAPSHOT_PATH:-/tmp/v1_real_sdk_arm_snapshot.json}"
export MUJOCO_JOINT_SNAPSHOT_PATH="${MUJOCO_JOINT_SNAPSHOT_PATH:-/tmp/v1_mujoco_realman_arm_snapshot.json}"
export REALMAN_PRESYNC_SNAPSHOT_PATH="${REALMAN_PRESYNC_SNAPSHOT_PATH:-/tmp/rm65_real_sdk_arm_snapshot.json}"
export REALMAN_ARM_COMMAND_MODE="${REALMAN_ARM_COMMAND_MODE:-movej_follow}"
export REALMAN_SDK_WRIST_PORT="${V1_REAL_WRIST_PORT}"

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

case "${V1_ENABLE_HAND_FOLLOW,,}" in
  1|true|yes|on)
    export REALMAN_DISABLE_HAND_FOLLOW="${REALMAN_DISABLE_HAND_FOLLOW:-0}"
    export REALMAN_SDK_HAND_PORT="${V1_REAL_HAND_PORT}"
    export REALMAN_HAND_BACKEND="${REALMAN_HAND_BACKEND:-realman_modbus}"
    export REALMAN_HAND_CONTROL_DT="${REALMAN_HAND_CONTROL_DT:-0.05}"
    ;;
  *)
    export REALMAN_DISABLE_HAND_FOLLOW="${REALMAN_DISABLE_HAND_FOLLOW:-1}"
    export REALMAN_SDK_HAND_PORT="0"
    ;;
esac

launch_udp_fanout() {
  if [[ "${V1_ENABLE_UDP_FANOUT,,}" =~ ^(0|false|no|off)$ ]]; then
    return 0
  fi

  local real_hand_port
  real_hand_port="${REALMAN_SDK_HAND_PORT:-0}"

  echo "Starting v1 UDP fanout ..."
  echo "  relay wrist listen: ${V1_RELAY_LISTEN_WRIST_PORT}"
  echo "  relay hand listen: ${V1_RELAY_LISTEN_HAND_PORT}"
  echo "  real wrist downstream: ${REALMAN_SDK_WRIST_PORT}"
  echo "  real hand downstream: ${real_hand_port}"
  echo "  mujoco wrist downstream: ${V1_MUJOCO_WRIST_PORT}"
  echo "  mujoco hand downstream: ${V1_MUJOCO_HAND_PORT}"

  "${V1_RELAY_PYTHON}" "${ROOT_DIR}/udp_port_fanout.py" \
    --listen-wrist-port "${V1_RELAY_LISTEN_WRIST_PORT}" \
    --listen-hand-port "${V1_RELAY_LISTEN_HAND_PORT}" \
    --real-wrist-port "${REALMAN_SDK_WRIST_PORT}" \
    --real-hand-port "${real_hand_port}" \
    --mujoco-wrist-port "${V1_MUJOCO_WRIST_PORT}" \
    --mujoco-hand-port "${V1_MUJOCO_HAND_PORT}" &
  RELAY_PID=$!
}

launch_mujoco() {
  if [[ "${V1_LAUNCH_MUJOCO_WITH_REAL,,}" =~ ^(0|false|no|off)$ ]]; then
    return 0
  fi

  echo "Starting v1 MuJoCo side-by-side validator ..."
  V1_MUJOCO_ARM_CONFIG="${V1_MUJOCO_ARM_CONFIG:-${REALMAN_SDK_TELEOP_CONFIG}}" \
  V1_MUJOCO_WRIST_PORT="${V1_MUJOCO_WRIST_PORT}" \
  V1_MUJOCO_HAND_PORT="${V1_MUJOCO_HAND_PORT}" \
  bash "${ROOT_DIR}/run_mujoco_realman_arm_only_validation.sh" &
  MUJOCO_PID=$!
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "${MUJOCO_PID:-}" ]]; then
    kill "${MUJOCO_PID}" >/dev/null 2>&1 || true
    wait "${MUJOCO_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${RELAY_PID:-}" ]]; then
    kill "${RELAY_PID}" >/dev/null 2>&1 || true
    wait "${RELAY_PID}" >/dev/null 2>&1 || true
  fi
  exit "${status}"
}

if want_help "$@"; then
  exec "${RM65_DOCKER_ROOT}/scripts/run_realman_sdk_follow_teleop.sh" "$@"
fi

trap cleanup EXIT INT TERM

launch_udp_fanout
launch_mujoco

"${RM65_DOCKER_ROOT}/scripts/run_realman_sdk_follow_teleop.sh" "$@"
