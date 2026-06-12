#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

QUEST_TCP_HOST="${REVO2_TCP_HOST:-0.0.0.0}"
QUEST_TCP_PORT="${REVO2_TCP_PORT:-8010}"
ROS_HAND_QPOS_HOST="${ROS_HAND_QPOS_HOST:-127.0.0.1}"
ROS_HAND_QPOS_PORT="${ROS_HAND_QPOS_PORT:-5010}"
ROS_WRIST_POSE_HOST="${ROS_WRIST_POSE_HOST:-127.0.0.1}"
ROS_WRIST_POSE_PORT="${ROS_WRIST_POSE_PORT:-5005}"
WRIST_DEBUG_LOG_FILE="${WRIST_DEBUG_LOG_FILE:-${REPO_ROOT}/rm65_dex_docker/logs/wrist_io_debug.txt}"

mkdir -p "$(dirname "${WRIST_DEBUG_LOG_FILE}")"
echo "==== host retargeting start $(date '+%F %T') ====" >> "${WRIST_DEBUG_LOG_FILE}"

cd "${REPO_ROOT}/example/vector_retargeting"

echo "Quest TCP input: ${QUEST_TCP_HOST}:${QUEST_TCP_PORT}"
echo "ROS hand qpos UDP output: ${ROS_HAND_QPOS_HOST}:${ROS_HAND_QPOS_PORT}"
echo "ROS wrist pose UDP output: ${ROS_WRIST_POSE_HOST}:${ROS_WRIST_POSE_PORT}"
echo "Wrist debug log file: ${WRIST_DEBUG_LOG_FILE}"

python3 vr_realtime_retargeting.py \
  --robot-name revo2 \
  --retargeting-type dexpilot \
  --hand-type left \
  --transport tcp \
  --host "${QUEST_TCP_HOST}" \
  --port "${QUEST_TCP_PORT}" \
  --publish-ros-hand-qpos \
  --ros-hand-qpos-host "${ROS_HAND_QPOS_HOST}" \
  --ros-hand-qpos-port "${ROS_HAND_QPOS_PORT}" \
  --forward-ros-wrist-pose \
  --ros-wrist-pose-host "${ROS_WRIST_POSE_HOST}" \
  --ros-wrist-pose-port "${ROS_WRIST_POSE_PORT}" \
  --debug-wrist-stream \
  --debug-wrist-period 1.0 \
  --wrist-debug-log-file "${WRIST_DEBUG_LOG_FILE}"
