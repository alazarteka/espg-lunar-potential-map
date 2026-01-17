#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 {cuda12|cuda11|modern|legacy} [uv sync args...]"
  exit 1
}

if [[ $# -lt 1 ]]; then
  usage
fi

case "$1" in
  cuda12|modern)
    lock_name="uv.lock.cuda12"
    extra_name="gpu"
    ;;
  cuda11|legacy)
    lock_name="uv.lock.cuda11"
    extra_name="gpu-legacy"
    ;;
  *)
    usage
    ;;
esac
shift

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
lock_src="$root_dir/locks/$lock_name"
lock_dst="$root_dir/uv.lock"

if [[ ! -f "$lock_src" ]]; then
  echo "Missing lockfile: $lock_src"
  exit 1
fi

cp "$lock_src" "$lock_dst"
echo "Selected $lock_name ($extra_name)"
uv sync --frozen --extra "$extra_name" --dev "$@"
