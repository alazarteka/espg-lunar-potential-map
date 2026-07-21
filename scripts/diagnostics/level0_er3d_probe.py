"""Inspect complete compressed ER 3-D sweeps in a Level-0 telemetry file.

This is a Phase-1 feasibility probe. It assembles and decompresses real-time
format-1/2 15x88 ER arrays, but does not calibrate them or align them to the
calibrated PDS product.

Usage:
    uv run python scripts/diagnostics/level0_er3d_probe.py M9801523.B
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.losscone.level0 import decompress_er_counter, iter_level0_er3d_sweeps


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("level0_file", type=Path)
    parser.add_argument("--max-sweeps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    all_sweeps = list(iter_level0_er3d_sweeps(args.level0_file))
    print(f"complete format-1/2 sweeps found: {len(all_sweeps)}")
    sweeps = all_sweeps
    if args.max_sweeps is not None:
        sweeps = sweeps[: args.max_sweeps]
    if not sweeps:
        return

    first = sweeps[0]
    decoded = decompress_er_counter(first.compressed_counts)
    print(f"first sweep starts at Level-0 record: {first.first_record_index}")
    print(
        f"first sweep Digital Subcom software version: {first.telemetry_format_version}"
    )
    print(
        f"compressed code range: {first.compressed_counts.min()}-{first.compressed_counts.max()}"
    )
    print(f"decoded lower-endpoint range: {decoded.min()}-{decoded.max()} counts")
    print(f"zero-code fraction: {np.mean(first.compressed_counts == 0):.3f}")


if __name__ == "__main__":
    main()
