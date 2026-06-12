#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${V1_MUJOCO_PYTHON:-${MUJOCO_PYTHON_BIN:-/home/xwen/anaconda3/envs/VR/bin/python}}"
CONFIG_PATH="${V1_MUJOCO_ARM_CONFIG:-$ROOT_DIR/config/wrist_realman_arm_only.yaml}"
WRIST_PORT="${V1_MUJOCO_WRIST_PORT:-5005}"
HAND_PORT="${V1_MUJOCO_HAND_PORT:-5010}"
JOINT_SNAPSHOT_PATH="${V1_MUJOCO_JOINT_SNAPSHOT_PATH:-/tmp/v1_mujoco_realman_arm_snapshot.json}"
INITIAL_SNAPSHOT_PATH="${V1_REALMAN_INITIAL_SNAPSHOT_PATH:-/tmp/rm65_real_sdk_arm_snapshot.json}"
MUJOCO_GL_BACKEND="${MUJOCO_GL:-glfw}"
SYSTEM_LIBSTDCXX="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
PY_ARGS=(
  --config "${CONFIG_PATH}"
  --wrist-port "${WRIST_PORT}"
  --hand-port "${HAND_PORT}"
  --joint-snapshot-path "${JOINT_SNAPSHOT_PATH}"
)

echo "v1 MuJoCo RealMan + dex-hand validator"
echo "  python: ${PYTHON_BIN}"
echo "  config: ${CONFIG_PATH}"
echo "  wrist UDP port: ${WRIST_PORT}"
echo "  hand UDP port: ${HAND_PORT}"
echo "  joint snapshot: ${JOINT_SNAPSHOT_PATH}"
if [[ -f "${INITIAL_SNAPSHOT_PATH}" ]]; then
  echo "  initial snapshot: ${INITIAL_SNAPSHOT_PATH}"
  PY_ARGS+=(--initial-snapshot-path "${INITIAL_SNAPSHOT_PATH}")
else
  echo "  initial snapshot: none; using config initial_joint_positions"
fi
echo "  render backend: ${MUJOCO_GL_BACKEND}"
echo
echo "Control path:"
echo "  Quest wrist/hand UDP -> shared DLS IK + regularization + Ruckig -> MuJoCo arm + dex-hand"
echo
echo "Reminder:"
echo "  1. Keep Quest -> host retargeting running so wrist_pose/hand_qpos continue forwarding."
echo "  2. This v1 scene removes the workspace table and mounts the dex-hand on the arm tool frame."
echo
if [[ ! -f "${INITIAL_SNAPSHOT_PATH}" ]]; then
  echo "No RealMan snapshot found. MuJoCo-only validation will start from the config seed."
  echo "To use the real arm's current pose later, capture it with:"
  echo "  /home/xwen/anaconda3/envs/sdk/bin/python ${ROOT_DIR}/read_realman_joint_state.py"
  echo
fi

export MUJOCO_GL="${MUJOCO_GL_BACKEND}"

if [[ "${MUJOCO_GL_BACKEND}" == "osmesa" ]]; then
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
  export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
  export MESA_LOADER_DRIVER_OVERRIDE="${MESA_LOADER_DRIVER_OVERRIDE:-llvmpipe}"
  export GALLIUM_DRIVER="${GALLIUM_DRIVER:-llvmpipe}"
  if [[ -f "${SYSTEM_LIBSTDCXX}" ]]; then
    if [[ -n "${LD_PRELOAD:-}" ]]; then
      export LD_PRELOAD="${SYSTEM_LIBSTDCXX}:${LD_PRELOAD}"
    else
      export LD_PRELOAD="${SYSTEM_LIBSTDCXX}"
    fi
  fi
elif [[ "${MUJOCO_GL_BACKEND}" == "glfw" ]]; then
  export DISPLAY="${DISPLAY:-:0}"
  export GLFW_PLATFORM="${GLFW_PLATFORM:-x11}"
  export GDK_BACKEND="${GDK_BACKEND:-x11}"
  export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
  export XDG_SESSION_TYPE="${XDG_SESSION_TYPE:-x11}"
  unset WAYLAND_DISPLAY || true
  if [[ -f "${SYSTEM_LIBSTDCXX}" ]]; then
    if [[ -n "${LD_PRELOAD:-}" ]]; then
      export LD_PRELOAD="${SYSTEM_LIBSTDCXX}:${LD_PRELOAD}"
    else
      export LD_PRELOAD="${SYSTEM_LIBSTDCXX}"
    fi
  fi
fi

exec "${PYTHON_BIN}" "${ROOT_DIR}/mujoco_realman_arm_only_validate.py" "${PY_ARGS[@]}" "$@"
