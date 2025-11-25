#!/usr/bin/env bash
set -euo pipefail

# Benchmark old (pipeline_seq) vs new (pipeline) across a date range.
# New pipeline runs with parallel fitting enabled to exercise max speed.
# Usage: scripts/dev/bench_pipelines.sh [START_DATE] [DAYS] [OUTPUT_FILE]
# Example: scripts/dev/bench_pipelines.sh 1998-01-16 5 /tmp/pipeline_times.txt

START_DATE=${1:-1998-01-16}
DAYS=${2:-3}
OUT_PATH=${3:-reports/pipeline_benchmark.txt}

UV_CACHE_DIR=${UV_CACHE_DIR:-.uv-cache}

log() {
  printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

# Build date list (YYYY-MM-DD) for the requested range.
build_dates() {
  local start=$1 count=$2
  local dates=()
  for ((i = 0; i < count; i++)); do
    dates+=("$(date -d "$start + $i day" +%Y-%m-%d)")
  done
  printf "%s\n" "${dates[@]}"
}

DATES=()
while IFS= read -r d; do DATES+=("$d"); done < <(build_dates "$START_DATE" "$DAYS")
DATES_STR=$(IFS=,; echo "${DATES[*]}")

measure_time() {
  # $1 label, $2... command
  local label=$1
  shift
  local start end
  start=$(date +%s.%N)
  "$@"
  end=$(date +%s.%N)
  awk -v l="$label" -v s="$start" -v e="$end" 'BEGIN { printf "%s_seconds=%.2f\n", l, (e - s) }'
}

run_old_pipeline() {
  UV_CACHE_DIR="$UV_CACHE_DIR" DATES="$DATES_STR" \
    uv run python - <<'PY'
import logging
import os
from types import SimpleNamespace

from src.potential_mapper import pipeline_seq as pipe
from src.potential_mapper.spice import load_spice_files

dates = os.environ["DATES"].split(",")
logging.basicConfig(level=logging.INFO)
load_spice_files()

codes = []
for d in dates:
    y, m, day = map(int, d.split("-"))
    args = SimpleNamespace(
        year=y, month=m, day=day, output=None, display=False, illumination=None
    )
    codes.append(pipe.run(args))

if any(c != 0 for c in codes):
    raise SystemExit(max(codes))

print(f"old_exit_codes={','.join(map(str, codes))}")
PY
}

run_new_pipeline() {
  UV_CACHE_DIR="$UV_CACHE_DIR" DATES="$DATES_STR" \
    uv run python - <<'PY'
import logging
import os
from pathlib import Path

from src.potential_mapper import pipeline as pipe
from src.potential_mapper.spice import load_spice_files

dates = os.environ["DATES"].split(",")
logging.basicConfig(level=logging.INFO)
load_spice_files()

files: list[Path] = []
for d in dates:
    y, m, day = map(int, d.split("-"))
    files.extend(pipe.DataLoader.discover_flux_files(y, m, day))

# Deduplicate while preserving order
seen = set()
unique_files = []
for f in files:
    if f not in seen:
        seen.add(f)
        unique_files.append(f)

if not unique_files:
    print("new_exit_code=1")
    raise SystemExit(1)

er_data = pipe.load_all_data(unique_files)
mode = "parallel"
try:
    agg = pipe.process_merged_data(er_data, use_parallel=True)
except PermissionError as exc:
    logging.warning(
        "Parallel fitting unavailable (%s); falling back to sequential", exc
    )
    mode = "sequential_fallback"
    agg = pipe.process_merged_data(er_data, use_parallel=False)

_ = agg.projected_potential.size  # force evaluation

print(f"new_exit_code=0")
print(f"new_mode={mode}")
print(f"new_files={len(unique_files)}")
print(f"new_rows={len(er_data.data)}")
PY
}

main() {
  mkdir -p "$(dirname "$OUT_PATH")"

  log "Benchmarking pipelines for ${START_DATE} + ${DAYS} day(s)"
  log "Cache dir: ${UV_CACHE_DIR}"
  log "Dates: ${DATES_STR}"

  old_result=$(measure_time "old" run_old_pipeline)
  new_result=$(measure_time "new_parallel" run_new_pipeline)

  {
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "start_date=${START_DATE}"
    echo "days=${DAYS}"
    echo "dates=${DATES_STR}"
    echo "$old_result"
    echo "$new_result"
    echo "cache_dir=${UV_CACHE_DIR}"
    echo "----"
  } >>"$OUT_PATH"

  log "Results appended to ${OUT_PATH}"
  printf "%s\n%s\n" "$old_result" "$new_result"
}

main "$@"
