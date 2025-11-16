#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: render_monthly_temporal_animation.sh YEAR MONTH

Reads the monthly NPZ produced by monthly_temporal_fit.sh (default location
data/temporal_coeffs/YYYY/YYYY-MM.npz) and renders hemisphere/global
animations into plots/temporal/YYYY/YYYY-MM/.

Environment overrides:
    INPUT_ROOT             (default: data/temporal_coeffs)
    OUTPUT_ROOT            (default: plots/temporal)
    ANIMATE_WRITER         (default: pillow)
    FPS                    (default: 10)
    DPI                    (default: 120)
    LAT_STEPS              (default: 181)
    LON_STEPS              (default: 361)
    SYMMETRIC_PERCENTILE   (default: 99.0)
    VMIN / VMAX            (default: unset → auto)
    LIMIT_FRAMES           (default: unset)
USAGE
}

if [[ $# -ne 2 ]]; then
    usage
    exit 1
fi

year=$1
month=$2

if ! [[ $year =~ ^[0-9]{4}$ ]]; then
    echo "ERROR: YEAR must be a four-digit number" >&2
    exit 1
fi

if ! [[ $month =~ ^[0-9]{1,2}$ ]] || (( month < 1 || month > 12 )); then
    echo "ERROR: MONTH must be between 1 and 12" >&2
    exit 1
fi

printf -v month_padded "%02d" "$month"
input_root=${INPUT_ROOT:-data/temporal_coeffs}
output_root=${OUTPUT_ROOT:-plots/temporal}

input_path="${input_root}/${year}/${year}-${month_padded}.npz"
if [[ ! -f ${input_path} ]]; then
    echo "ERROR: Input file ${input_path} not found" >&2
    exit 1
fi

output_dir="${output_root}/${year}/${year}-${month_padded}"
mkdir -p "${output_dir}"

args=(
    --input "${input_path}"
    --output-dir "${output_dir}"
    --writer "${ANIMATE_WRITER:-pillow}"
    --fps "${FPS:-3}"
    --dpi "${DPI:-120}"
    --lat-steps "${LAT_STEPS:-181}"
    --lon-steps "${LON_STEPS:-361}"
    --symmetric-percentile "${SYMMETRIC_PERCENTILE:-99.0}"
)

if [[ -n ${LIMIT_FRAMES:-} ]]; then
    args+=(--limit-frames "${LIMIT_FRAMES}")
fi
if [[ -n ${VMIN:-} ]]; then
    args+=(--vmin "${VMIN}")
fi
if [[ -n ${VMAX:-} ]]; then
    args+=(--vmax "${VMAX}")
fi

echo "Rendering animations for ${year}-${month_padded}"
echo "Input : ${input_path}"
echo "Output: ${output_dir}"

uv run python scripts/analysis/temporal_harmonics_animate.py "${args[@]}"
