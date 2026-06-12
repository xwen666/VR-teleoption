#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC=/home/xwen/vr/assets/robots/arms/rm_description
DST="$ROOT/workspace/rm65_dex_ws/src/rm_description"

if [[ ! -d "$SRC" ]]; then
  echo "Source rm_description not found: $SRC" >&2
  exit 1
fi

rm -rf "$DST"
cp -a "$SRC" "$DST"
echo "Synced rm_description:"
echo "  $SRC"
echo "  -> $DST"
