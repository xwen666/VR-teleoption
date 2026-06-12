#!/usr/bin/env bash
set -euo pipefail

if docker info >/dev/null 2>&1; then
  docker exec -it rm65_dex_ros2 bash
else
  sudo docker exec -it rm65_dex_ros2 bash
fi
