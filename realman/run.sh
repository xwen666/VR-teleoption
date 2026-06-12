#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-ros:0415}"
CONTAINER_NAME="${CONTAINER_NAME:-ubuntu2204-dev}"
DOCKER_NETWORK_MODE="${DOCKER_NETWORK_MODE:-host}"
UBUNTU_SERIES="${UBUNTU_SERIES:-22.04}"
ROOTFS_NAME="${ROOTFS_NAME:-ubuntu-base.tar.gz}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOTFS_PATH="$SCRIPT_DIR/$ROOTFS_NAME"
DOCKER_CMD=(docker)
RUN_ARGS=()

add_run_args() {
  RUN_ARGS+=("$@")
}

add_devices_from_glob() {
  local pattern="$1"
  local found=1
  local device

  shopt -s nullglob
  for device in $pattern; do
    found=0
    add_run_args --device "$device:$device"
    echo "Passing device into container: $device"
  done
  shopt -u nullglob
  return "$found"
}

detect_arch() {
  case "$(uname -m)" in
    x86_64|amd64)
      echo "amd64"
      ;;
    aarch64|arm64)
      echo "arm64"
      ;;
    *)
      echo "Unsupported CPU architecture: $(uname -m)" >&2
      exit 1
      ;;
  esac
}

resolve_rootfs_url() {
  local arch release_page filename

  arch="$(detect_arch)"
  release_page="https://cdimage.ubuntu.com/ubuntu-base/releases/${UBUNTU_SERIES}/release/"

  if [[ -n "${ROOTFS_URL:-}" ]]; then
    echo "$ROOTFS_URL"
    return
  fi

  if [[ -n "${UBUNTU_BASE_VERSION:-}" ]]; then
    echo "${release_page}ubuntu-base-${UBUNTU_BASE_VERSION}-base-${arch}.tar.gz"
    return
  fi

  filename="$(
    curl -fsSL "$release_page" \
      | grep -oE "ubuntu-base-${UBUNTU_SERIES//./\\.}(\\.[0-9]+)?-base-${arch}\\.tar\\.gz" \
      | sort -Vu \
      | tail -n 1
  )"

  if [[ -z "$filename" ]]; then
    echo "Could not find an Ubuntu Base tarball for ${UBUNTU_SERIES} (${arch})." >&2
    exit 1
  fi

  echo "${release_page}${filename}"
}

download_rootfs() {
  local url tmp_path

  if [[ -f "$ROOTFS_PATH" ]]; then
    echo "Using cached rootfs tarball: $ROOTFS_PATH"
    return
  fi

  url="$(resolve_rootfs_url)"
  tmp_path="${ROOTFS_PATH}.tmp"

  echo "Downloading Ubuntu Base rootfs from:"
  echo "  $url"
  curl -fL --retry 3 --connect-timeout 15 "$url" -o "$tmp_path"
  mv "$tmp_path" "$ROOTFS_PATH"
}

ensure_image_available() {
  if "${DOCKER_CMD[@]}" image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "Using existing image: $IMAGE_NAME"
    return
  fi

  download_rootfs
  echo "Image $IMAGE_NAME was not found locally. Building it from $SCRIPT_DIR."
  "${DOCKER_CMD[@]}" build -t "$IMAGE_NAME" "$SCRIPT_DIR"
}

ensure_docker_access() {
  if docker info >/dev/null 2>&1; then
    return
  fi

  if ! command -v sudo >/dev/null 2>&1; then
    echo "Docker daemon is not accessible for the current user, and sudo is unavailable." >&2
    exit 1
  fi

  echo "Docker daemon requires sudo on this machine. You may be prompted for your password."
  sudo -v
  DOCKER_CMD=(sudo docker)
}

if [[ "${EUID:-$(id -u)}" -eq 0 ]] && [[ -z "${http_proxy:-}${HTTP_PROXY:-}${https_proxy:-}${HTTPS_PROXY:-}" ]]; then
  echo "No proxy variables were detected as root." >&2
  echo "If the rootfs download times out, run ./run.sh as your normal user instead of sudo ./run.sh." >&2
fi

ensure_docker_access

ensure_image_available

add_run_args --rm -it
add_run_args --name "$CONTAINER_NAME"
add_run_args --network "$DOCKER_NETWORK_MODE"
add_run_args -v "$SCRIPT_DIR":/workspace
add_run_args -w /workspace

if [[ -d /dev/bus/usb ]]; then
  add_run_args -v /dev/bus/usb:/dev/bus/usb
  add_run_args --device-cgroup-rule "c 189:* rmw"
  echo "Passing USB bus into container: /dev/bus/usb"
fi

if [[ -d /run/udev ]]; then
  add_run_args -v /run/udev:/run/udev:ro
  echo "Passing udev runtime into container: /run/udev"
fi

if ! add_devices_from_glob "/dev/video*"; then
  echo "No /dev/video* devices found on the host. Camera access may fail." >&2
fi

add_devices_from_glob "/dev/media*" || true
add_devices_from_glob "/dev/ttyUSB*" || true
add_devices_from_glob "/dev/ttyACM*" || true
add_devices_from_glob "/dev/ttyCH*" || true

if [[ -n "${DISPLAY:-}" ]] && [[ -d /tmp/.X11-unix ]]; then
  add_run_args -e "DISPLAY=${DISPLAY}"
  add_run_args -e QT_X11_NO_MITSHM=1
  add_run_args -v /tmp/.X11-unix:/tmp/.X11-unix:rw
  echo "Passing X11 socket into container for GUI apps."

  if [[ -n "${XAUTHORITY:-}" ]] && [[ -f "${XAUTHORITY}" ]]; then
    add_run_args -e XAUTHORITY=/tmp/.docker.xauth
    add_run_args -v "${XAUTHORITY}:/tmp/.docker.xauth:ro"
  elif [[ -f "${HOME}/.Xauthority" ]]; then
    add_run_args -e XAUTHORITY=/tmp/.docker.xauth
    add_run_args -v "${HOME}/.Xauthority:/tmp/.docker.xauth:ro"
  fi
fi

echo "Starting container ${CONTAINER_NAME} from image ${IMAGE_NAME} with network mode ${DOCKER_NETWORK_MODE}"
"${DOCKER_CMD[@]}" run "${RUN_ARGS[@]}" "$IMAGE_NAME" /bin/bash
