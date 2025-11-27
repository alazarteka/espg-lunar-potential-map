#!/usr/bin/env bash
# Run the potential cache reader with optional spherical harmonic regularization.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

usage() {
    cat <<'EOF'
Usage: potential_cache_reader.sh <YYYYMMDD> [m] [optional flags...]

DATE (YYYYMMDD) sets the --start date. When the optional `m` positional is
present the run enables --plot-measurements and switches the output filename
to harmonics-m-<DATE>.png.

Defaults:
  --lmax 60
  --regularize-l2 1.0         (0.5 when `m` is provided)
  --plot-output artifacts/plots/harmonics[-m]-<DATE>.png

Additional flags (e.g., --end, --plot-lat-steps) are forwarded to the Python
CLI unchanged.
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

DATE_ARG="$1"
shift

if [[ ! "${DATE_ARG}" =~ ^[0-9]{8}$ ]]; then
    echo "ERROR: date must be in YYYYMMDD format" >&2
    exit 1
fi

START_DATE="${DATE_ARG:0:4}-${DATE_ARG:4:2}-${DATE_ARG:6:2}"

MEASUREMENTS=false
if [[ $# -gt 0 && "$1" == "m" ]]; then
    MEASUREMENTS=true
    shift
fi

DEFAULT_LMAX=60
DEFAULT_L2_NO_MEAS=1.0
DEFAULT_L2_WITH_MEAS=0.5

if [[ "${MEASUREMENTS}" == true ]]; then
    L2_PENALTY="${DEFAULT_L2_WITH_MEAS}"
else
    L2_PENALTY="${DEFAULT_L2_NO_MEAS}"
fi
LMAX_VALUE="${DEFAULT_LMAX}"
PLOT_OUTPUT=""
CUSTOM_PLOT_OUTPUT=false
FORWARDED_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --l2|--regularize-l2)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --l2 expects a value" >&2
                exit 1
            fi
            L2_PENALTY="$2"
            shift 2
            ;;
        --lmax)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --lmax expects a value" >&2
                exit 1
            fi
            LMAX_VALUE="$2"
            shift 2
            ;;
        --plot-output)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --plot-output expects a value" >&2
                exit 1
            fi
            PLOT_OUTPUT="$2"
            CUSTOM_PLOT_OUTPUT=true
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            FORWARDED_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ "${CUSTOM_PLOT_OUTPUT}" == false ]]; then
    if [[ "${MEASUREMENTS}" == true ]]; then
        PLOT_OUTPUT="artifacts/plots/harmonics-m-${DATE_ARG}.png"
    else
        PLOT_OUTPUT="artifacts/plots/harmonics-${DATE_ARG}.png"
    fi
fi

PLOT_DIR="$(dirname "${PLOT_OUTPUT}")"
mkdir -p "${PLOT_DIR}"

CMD=(
    uv run python "${SCRIPT_DIR}/potential_cache_reader.py"
    --start "${START_DATE}"
    --lmax "${LMAX_VALUE}"
    --regularize-l2 "${L2_PENALTY}"
    --plot-output "${PLOT_OUTPUT}"
)

if [[ "${MEASUREMENTS}" == true ]]; then
    CMD+=(--plot-measurements)
fi

CMD+=("${FORWARDED_ARGS[@]}")

cd "${REPO_ROOT}"
exec "${CMD[@]}"
