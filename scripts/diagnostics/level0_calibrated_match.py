"""Rank coarse Level-0 ER-code to calibrated-flux association candidates.

This forensic diagnostic compares time-near 15x88 arrays using a scale-free
angular-shape score. It does not recover counts or reproduce calibrated flux.

Usage:
    uv run python scripts/diagnostics/level0_calibrated_match.py \
        /tmp/lp_level0_reference/M9801523.B data/1998/001_031JAN/3D980116.TAB
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

from src.losscone.level0 import iter_level0_er3d_sweeps
from src.losscone.raw_calibration import (
    RawCalibratedCandidate,
    rank_time_near_raw_calibrated_candidates,
    read_calibrated_er3d_sweeps,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_file", type=Path)
    parser.add_argument("calibrated_file", type=Path)
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Year for the raw Earth-receive field; inferred from standard raw filename.",
    )
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--max-time-delta-seconds", type=float, default=600.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report path; raw matrices are never written.",
    )
    return parser.parse_args()


def _candidate_to_json(candidate: RawCalibratedCandidate) -> dict[str, object]:
    values = asdict(candidate)
    for key in (
        "raw_earth_receive_time",
        "calibrated_start_time",
        "calibrated_end_time",
    ):
        values[key] = values[key].isoformat()
    if not math.isfinite(candidate.angular_log_correlation):
        values["angular_log_correlation"] = None
    return values


def main() -> None:
    args = _parse_args()
    raw_sweeps = list(iter_level0_er3d_sweeps(args.raw_file, year=args.year))
    calibrated_sweeps = read_calibrated_er3d_sweeps(args.calibrated_file)
    candidates = rank_time_near_raw_calibrated_candidates(
        raw_sweeps,
        calibrated_sweeps,
        max_receive_minus_end_seconds=args.max_time_delta_seconds,
    )
    top_candidates = candidates[: args.top]

    print(f"complete raw format-1/2 sweeps: {len(raw_sweeps)}")
    print(f"complete calibrated 15-row sweeps: {len(calibrated_sweeps)}")
    print(f"time-near candidates: {len(candidates)}")
    for candidate in top_candidates:
        print(
            f"raw_record={candidate.raw_first_record_index} "
            f"raw_ert={candidate.raw_earth_receive_time.isoformat()} "
            f"spec={candidate.calibrated_spec_no} "
            f"cal_end={candidate.calibrated_end_time.isoformat()} "
            f"ert_minus_end_s={candidate.receive_minus_end_seconds:.3f} "
            f"angular_log_corr={candidate.angular_log_correlation:.5f} "
            f"positive_fraction={candidate.positive_cell_fraction:.3f}"
        )

    if args.output is not None:
        report = {
            "status": "time_association_coarse",
            "prohibited_inference": (
                "Candidate ranking is not a raw-count calibration or a likelihood input."
            ),
            "raw_file": str(args.raw_file),
            "raw_sha256": hashlib.sha256(args.raw_file.read_bytes()).hexdigest(),
            "calibrated_file": str(args.calibrated_file),
            "calibrated_sha256": hashlib.sha256(
                args.calibrated_file.read_bytes()
            ).hexdigest(),
            "max_receive_minus_end_seconds": args.max_time_delta_seconds,
            "candidates": [_candidate_to_json(candidate) for candidate in candidates],
        }
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
