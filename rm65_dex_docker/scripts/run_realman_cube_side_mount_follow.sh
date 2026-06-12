#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REALMAN_SDK_TELEOP_CONFIG="${REALMAN_SDK_TELEOP_CONFIG:-$ROOT_DIR/workspace/rm65_dex_ws/src/quest_bridge/config/wrist_cube_side_mount.yaml}"

exec "$ROOT_DIR/scripts/run_realman_sdk_follow_teleop.sh" "$@"
