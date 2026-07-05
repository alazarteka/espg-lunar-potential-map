#!/usr/bin/env bash
#
# Bootstrap a runnable environment from a fresh clone.
#
# Zero-argument happy path:
#
#     ./scripts/bootstrap.sh
#
# It detects the GPU architecture and syncs the matching PyTorch build:
#
#   * sm_70+  (Volta / Turing / Ampere / Ada / Hopper / Blackwell — RTX 20xx
#             and newer)      -> modern CUDA 12.8 stack, latest torch  [DEFAULT]
#   * sm_61   (Pascal — GTX 10xx, TITAN Xp)
#                             -> legacy CUDA 11.8 stack (the last torch build
#                                that still ships Pascal kernels)
#
# The modern stack is the default and the path we optimize for. The legacy
# stack exists so Pascal cards keep working as long as upstream allows; expect
# it to lag the latest torch (Pascal is deprecated in CUDA 12 and dropped in
# CUDA 13, and torch.compile/Triton never supported it).
#
# Override auto-detection (e.g. building on a GPU-less node for a known target):
#
#     ./scripts/bootstrap.sh modern      # force CUDA 12.8 stack
#     ./scripts/bootstrap.sh legacy      # force CUDA 11.8 stack
#
# Extra arguments are forwarded to `uv sync`, e.g.:
#
#     ./scripts/bootstrap.sh modern --extra notebook
#
# Re-running is safe and idempotent: it just re-copies the lockfile and re-runs
# `uv sync --frozen`, which is a no-op when the environment is already current.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

log()  { printf '\n\033[1;34m[bootstrap]\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;33m[bootstrap:warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\n\033[1;31m[bootstrap:error]\033[0m %s\n' "$*" >&2; exit 1; }

command -v uv >/dev/null || die "uv not found. Install from https://docs.astral.sh/uv/"

# --- Parse the target: explicit arg wins, otherwise auto-detect --------------
choice="auto"
if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help)
      grep '^#' "$0" | grep -v '^#!' | sed 's/^#\{1,\} \{0,1\}//'
      exit 0
      ;;
    *)
      choice="$1"
      shift
      ;;
  esac
fi

# Pick the stack by the OLDEST visible GPU: the legacy cu118 wheel spans
# sm_61..sm_90, but the modern cu128 wheel has no Pascal kernels, so a mixed
# host (one Pascal + one Ampere card) must use the legacy stack to run on all.
detect_stack() {
  local caps cap major minor min_major=99 min_minor=99
  command -v nvidia-smi >/dev/null 2>&1 || { echo "none"; return; }
  caps="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' ')" || true
  [[ -n "$caps" ]] || { echo "none"; return; }
  while IFS= read -r cap; do
    [[ -n "$cap" ]] || continue
    major="${cap%%.*}"
    minor="${cap##*.}"
    if [[ "$major" -lt "$min_major" ]] ||
       { [[ "$major" -eq "$min_major" ]] && [[ "$minor" -lt "$min_minor" ]]; }; then
      min_major="$major"
      min_minor="$minor"
    fi
  done <<< "$caps"
  if   [[ "$min_major" -ge 7 ]]; then echo "modern"
  elif [[ "$min_major" -eq 6 ]]; then echo "legacy"
  else echo "unknown:${min_major}.${min_minor}"; fi
}

case "$choice" in
  auto)
    stack="$(detect_stack)"
    case "$stack" in
      modern)
        log "Detected sm_70+ GPU(s) -> modern CUDA 12.8 stack (latest torch)." ;;
      legacy)
        warn "Detected Pascal (sm_61) GPU(s) -> legacy CUDA 11.8 stack (last cu118 torch; runs on Pascal via PTX JIT)." ;;
      none)
        warn "No NVIDIA GPU detected -> defaulting to the modern stack. Force legacy with: ./scripts/bootstrap.sh legacy"
        stack="modern" ;;
      unknown:*)
        warn "Unrecognized GPU arch ${stack#unknown:} -> defaulting to the legacy stack for safety. Override with: ./scripts/bootstrap.sh modern"
        stack="legacy" ;;
    esac
    ;;
  modern|cuda12) stack="modern" ;;
  legacy|cuda11) stack="legacy" ;;
  *) die "Unknown target '$choice'. Use 'modern', 'legacy', or no argument to auto-detect." ;;
esac

case "$stack" in
  modern) lock_name="uv.lock.cuda12"; extra_name="gpu" ;;
  legacy) lock_name="uv.lock.cuda11"; extra_name="gpu-legacy" ;;
esac

lock_src="$ROOT/locks/$lock_name"
[[ -f "$lock_src" ]] || die "Missing lockfile: $lock_src"

log "Selecting $lock_name (extra: $extra_name)"
cp "$lock_src" "$ROOT/uv.lock"

log "Running: uv sync --frozen --extra $extra_name --dev $*"
uv sync --frozen --extra "$extra_name" --dev "$@"

log "Environment ready. Run tools with 'uv run ...' (e.g. uv run pytest -q)."
