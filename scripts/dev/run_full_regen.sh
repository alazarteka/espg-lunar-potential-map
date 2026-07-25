#!/usr/bin/env bash
#
# Full-dataset regeneration driver for the potential-mapper batch pipeline.
#
# Runs the batch CLI once per calendar month that has data under data/1998 and
# data/1999 (the month list is derived from the PDS directory names, e.g.
# "091_120APR" -> April). Month-level runs merge every day in the month into a
# single NPZ named potential_batch_YYYY_MM.npz, so one run per month covers the
# full 560-day dataset in 19 invocations.
#
# Properties:
#   * idempotent restart - a month whose output NPZ already exists is skipped
#     (the batch CLI itself also refuses to overwrite without --overwrite)
#   * per-month wall time and exit code are appended to the logfile
#   * git sha + the exact command line are recorded in provenance.json
#
# Usage:
#   scripts/dev/run_full_regen.sh                     # full run
#   scripts/dev/run_full_regen.sh --emit-u-width-qc   # extra batch CLI flags
#   OUTPUT_DIR=artifacts/foo scripts/dev/run_full_regen.sh
#
# Any extra arguments are appended verbatim to every batch invocation.
#
# Recommended: pass --emit-u-width-qc. Re-running 1999-04-29 day-chunked vs
# month-chunked showed 2% of spectra move by >50 V (max 1435 V) at identical
# chi2 - i.e. the DE lands in a different basin of a U-flat likelihood. Those
# cases concentrate in the spectra the QC flags as non-identifiable (5.8% move
# >50 V vs 0.7% for identifiable ones), so without the flag there is no way to
# separate a fitted U from an arbitrary one downstream.
#
set -euo pipefail
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_DIR="${OUTPUT_DIR:-artifacts/potential_cache_2026-07-25}"
DATA_DIR="${DATA_DIR:-data}"
# Space-separated so it can be overridden from the environment.
YEARS="${YEARS:-1998 1999}"
EXTRA_ARGS=("$@")

BATCH_ARGS=(
    --fast
    --losscone-fit-method lillis
    --u-spacecraft 0
    --output-dir "$OUTPUT_DIR"
)

mkdir -p "$OUTPUT_DIR"
LOGFILE="${LOGFILE:-$OUTPUT_DIR/regen.log}"

log() {
    printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOGFILE"
}

# --- month discovery -------------------------------------------------------
# PDS directories are named <doy_start>_<doy_end><MON>, e.g. 091_120APR.
month_number() {
    case "${1^^}" in
        JAN) echo 1 ;;  FEB) echo 2 ;;  MAR) echo 3 ;;  APR) echo 4 ;;
        MAY) echo 5 ;;  JUN) echo 6 ;;  JUL) echo 7 ;;  AUG) echo 8 ;;
        SEP) echo 9 ;;  OCT) echo 10 ;; NOV) echo 11 ;; DEC) echo 12 ;;
        *) return 1 ;;
    esac
}

declare -a TARGETS=()
for year in $YEARS; do
    year_dir="$DATA_DIR/$year"
    [[ -d "$year_dir" ]] || continue
    for month_dir in "$year_dir"/*/; do
        [[ -d "$month_dir" ]] || continue
        base="$(basename "$month_dir")"
        mon_abbr="${base: -3}"
        if ! month="$(month_number "$mon_abbr")"; then
            log "WARN  skipping unrecognized directory: $month_dir"
            continue
        fi
        n_files="$(find "$month_dir" -maxdepth 1 -name '*.TAB' | wc -l)"
        if [[ "$n_files" -eq 0 ]]; then
            log "WARN  no .TAB files in $month_dir; skipping"
            continue
        fi
        TARGETS+=("$year $month $n_files")
    done
done

if [[ ${#TARGETS[@]} -eq 0 ]]; then
    log "ERROR no month directories with data found under $DATA_DIR"
    exit 1
fi

# --- provenance ------------------------------------------------------------
GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="$(if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then echo true; else echo false; fi)"
FULL_COMMAND="python -m src.potential_mapper.batch ${BATCH_ARGS[*]} --year <YEAR> --month <MONTH>"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    FULL_COMMAND="$FULL_COMMAND ${EXTRA_ARGS[*]}"
fi

{
    printf '{\n'
    printf '  "generated_utc": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '  "git_sha": "%s",\n' "$GIT_SHA"
    printf '  "git_branch": "%s",\n' "$GIT_BRANCH"
    printf '  "git_dirty": %s,\n' "$GIT_DIRTY"
    printf '  "driver": "scripts/dev/run_full_regen.sh",\n'
    printf '  "output_dir": "%s",\n' "$OUTPUT_DIR"
    printf '  "command_template": "%s",\n' "$FULL_COMMAND"
    printf '  "months": ['
    sep=''
    for target in "${TARGETS[@]}"; do
        read -r y m n <<<"$target"
        printf '%s\n    {"year": %d, "month": %d, "n_files": %d}' "$sep" "$y" "$m" "$n"
        sep=','
    done
    printf '\n  ]\n'
    printf '}\n'
} >"$OUTPUT_DIR/provenance.json"

log "INFO  regen start: ${#TARGETS[@]} month(s), sha=$GIT_SHA branch=$GIT_BRANCH dirty=$GIT_DIRTY"
log "INFO  output_dir=$OUTPUT_DIR logfile=$LOGFILE"

# --- run -------------------------------------------------------------------
run_start=$SECONDS
n_ok=0
n_skipped=0
n_failed=0

for target in "${TARGETS[@]}"; do
    read -r year month n_files <<<"$target"
    npz="$OUTPUT_DIR/$(printf 'potential_batch_%d_%02d.npz' "$year" "$month")"
    tag="$(printf '%d-%02d' "$year" "$month")"

    if [[ -f "$npz" ]]; then
        log "SKIP  $tag output already exists ($npz)"
        n_skipped=$((n_skipped + 1))
        continue
    fi

    log "RUN   $tag ($n_files file(s))"
    month_start=$SECONDS
    set +e
    uv run python -m src.potential_mapper.batch \
        "${BATCH_ARGS[@]}" \
        --year "$year" \
        --month "$month" \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
        >>"$LOGFILE" 2>&1
    status=$?
    set -e
    elapsed=$((SECONDS - month_start))

    if [[ $status -eq 0 ]]; then
        if [[ -f "$npz" ]]; then
            log "OK    $tag exit=0 wall=${elapsed}s out=$npz"
            n_ok=$((n_ok + 1))
        else
            log "FAIL  $tag exit=0 but no NPZ written wall=${elapsed}s"
            n_failed=$((n_failed + 1))
        fi
    else
        log "FAIL  $tag exit=$status wall=${elapsed}s (see $LOGFILE)"
        n_failed=$((n_failed + 1))
    fi
done

total=$((SECONDS - run_start))
log "INFO  regen done: ok=$n_ok skipped=$n_skipped failed=$n_failed total_wall=${total}s"

if [[ $n_failed -gt 0 ]]; then
    exit 1
fi
exit 0
