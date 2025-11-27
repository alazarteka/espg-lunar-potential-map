#!/bin/bash

# usage: ./losscone_run_chunk.sh <chunk_number>

CHUNK=$1

if [ -z "$CHUNK" ]; then
  echo "usage: $0 <chunk_number>"
  exit 1
fi

uv run python scripts/dev/losscone_plot_fit.py \
  --file data/1998/060_090MAR/3D980323.TAB \
  --chunk "$CHUNK" \
  --output "scratch/perf_runs/1998-03-23/losscone_chunk${CHUNK}.png" \
  --dump-dir "scratch/perf_runs/1998-03-23/chunk${CHUNK}_dump"
