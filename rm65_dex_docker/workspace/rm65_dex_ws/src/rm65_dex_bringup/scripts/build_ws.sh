#!/usr/bin/env bash
set -euo pipefail

cd /workspace/rm65_dex_ws

# ROS setup scripts read optional environment variables that may be unset.
set +u
source /opt/ros/humble/setup.bash
set -u

install_ruckig_python() {
  local default_src="/workspace/ruckig"
  local alt_src="/workspace/rm65_dex_ws/src/ruckig"
  local ruckig_src="${RUCKIG_SOURCE_DIR:-$default_src}"

  if [[ ! -f "${ruckig_src}/pyproject.toml" && -f "${alt_src}/pyproject.toml" ]]; then
    ruckig_src="${alt_src}"
  fi

  if [[ ! -f "${ruckig_src}/pyproject.toml" ]]; then
    echo "Ruckig source not mounted; wrist bridge will use its fallback smoother if Python ruckig is absent."
    return
  fi

  if python3 - <<'PY' >/dev/null 2>&1
from ruckig import InputParameter, OutputParameter, Ruckig
PY
  then
    echo "Python ruckig is already importable."
    return
  fi

  echo "Installing Python ruckig from ${ruckig_src} ..."
  python3 -m pip install --user --upgrade --no-cache-dir scikit-build-core nanobind
  python3 -m pip install --user --upgrade --no-cache-dir "${ruckig_src}"
}

install_ruckig_python

rosdep install --from-paths src --ignore-src -r -y || true
colcon build --symlink-install

set +u
source install/setup.bash
set -u

echo "Workspace built and sourced."
