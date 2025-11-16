#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: monthly_temporal_fit.sh YEAR MONTH

Example:
    ./scripts/dev/monthly_temporal_fit.sh 1998 04

Environment overrides:
    CACHE_DIR                 (default: data/potential_cache)
    OUTPUT_ROOT               (default: data/temporal_coeffs)
    L_MAX                     (default: 15)
    WINDOW_HOURS              (default: 24)
    WINDOW_STRIDE             (default: unset → non-overlapping)
    REGULARIZE_L2             (default: 10.0)
    TEMPORAL_LAMBDA           (default: 10.0)
    MIN_SAMPLES               (default: 100)
    MIN_COVERAGE              (default: 0.1)
    CO_ROTATE                 (default: true)  # set to "true" to enable
    ROTATION_PERIOD_DAYS      (default: 29.530588)
    SPATIAL_WEIGHT_EXPONENT   (default: unset)
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

printf -v start_date "%04d-%02d-01" "$year" "$month"

if ! end_date=$(date -u -d "${start_date} +1 month -1 day" +%Y-%m-%d 2>/dev/null); then
    echo "ERROR: failed to compute end date; requires GNU date" >&2
    exit 1
fi

cache_dir=${CACHE_DIR:-data/potential_cache}
output_root=${OUTPUT_ROOT:-data/temporal_coeffs}

printf -v month_padded "%02d" "$month"
output_dir="${output_root}/${year}"
output_file="${output_dir}/${year}-${month_padded}.npz"

mkdir -p "${output_dir}"

args=(
    --start "${start_date}"
    --end "${end_date}"
    --cache-dir "${cache_dir}"
    --output "${output_file}"
    --lmax "${L_MAX:-15}"
    --window-hours "${WINDOW_HOURS:-24}"
    --regularize-l2 "${REGULARIZE_L2:-10.0}"
    --temporal-lambda "${TEMPORAL_LAMBDA:-10.0}"
    --min-samples "${MIN_SAMPLES:-100}"
    --min-coverage "${MIN_COVERAGE:-0.1}"
)

if [[ -n ${WINDOW_STRIDE:-} ]]; then
    args+=(--window-stride "${WINDOW_STRIDE}")
fi

if [[ ${CO_ROTATE:-true} == "true" ]]; then
    args+=(--co-rotate --rotation-period-days "${ROTATION_PERIOD_DAYS:-29.530588}")
fi

if [[ -n ${SPATIAL_WEIGHT_EXPONENT:-} ]]; then
    args+=(--spatial-weight-exponent "${SPATIAL_WEIGHT_EXPONENT}")
fi

echo "Running monthly fit for ${start_date} → ${end_date}"
echo "Cache dir : ${cache_dir}"
echo "Output    : ${output_file}"

uv run python -m src.temporal.coefficients "${args[@]}"
