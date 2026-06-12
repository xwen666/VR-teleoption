#!/usr/bin/env bash
set -euo pipefail

PORT="${REVO2_TCP_PORT:-8010}"

adb devices
adb reverse "tcp:${PORT}" "tcp:${PORT}"
adb reverse --list
