#!/usr/bin/env bash
set -euo pipefail

xhost +local:
xhost +SI:localuser:"$(id -un)" || true
