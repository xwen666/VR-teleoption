#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RM65_DOCKER_ROOT="$(cd "${ROOT_DIR}/../rm65_dex_docker" && pwd)"
RETARGET_PYTHON="${V1_RETARGET_PYTHON:-/home/xwen/anaconda3/envs/VR/bin/python3}"
RETARGET_PYTHON_DIR="$(dirname "${RETARGET_PYTHON}")"

if [[ -x "${RETARGET_PYTHON}" ]]; then
  export PATH="${RETARGET_PYTHON_DIR}:${PATH}"
fi

exec "${RM65_DOCKER_ROOT}/scripts/run_revo2_retargeting_to_ros.sh" "$@"
