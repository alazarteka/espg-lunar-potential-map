#!/usr/bin/env python3
"""
Scan loss-cone normalized flux for mid-energy peaks at high pitch angles.

This script loads a single ER file, applies loss-cone normalization, and
checks each sweep for a peak in the high-pitch band (default 150–180°).
It reports the fraction of total sweeps and polarity-valid sweeps that
show a peak.

Usage:
  uv run python scripts/diagnostics/losscone_peak_scan.py data/.../3D*.TAB
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from src import config
from src.diagnostics import LossConeSession
from src.flux import ERData


def _build_energy_profile(
    energies: np.ndarray,
    pitches: np.ndarray,
    norm2d: np.ndarray,
    pitch_min: float,
    pitch_max: float,
    min_band_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    band_mask = (pitches >= pitch_min) & (pitches <= pitch_max)
    values = np.full(energies.shape, np.nan, dtype=float)

    for idx in range(len(energies)):
        row_mask = band_mask[idx] & np.isfinite(norm2d[idx])
        if min_band_points > 0 and np.count_nonzero(row_mask) < min_band_points:
            continue
        if np.any(row_mask):
            values[idx] = float(np.nanmean(norm2d[idx][row_mask]))

    order = np.argsort(energies)
    return energies[order], values[order]


def _has_peak(
    profile: np.ndarray,
    *,
    contrast: float,
    min_peak: float,
    neighbor_window: int,
    edge_skip: int,
) -> bool:
    if profile.size == 0:
        return False

    window = max(1, neighbor_window)
    start = max(edge_skip, window)
    end = profile.size - max(edge_skip, window)
    if end <= start:
        return False

    for idx in range(start, end):
        value = profile[idx]
        if not np.isfinite(value) or value < min_peak:
            continue
        left = profile[idx - window : idx]
        right = profile[idx + 1 : idx + 1 + window]
        left_max = np.nanmax(left) if left.size else np.nan
        right_max = np.nanmax(right) if right.size else np.nan
        if not np.isfinite(left_max) or not np.isfinite(right_max):
            continue
        if value >= left_max * contrast and value >= right_max * contrast:
            return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan loss-cone normalized flux for high-pitch peaks."
    )
    parser.add_argument("er_file", type=Path, help="Path to ER .TAB file")
    parser.add_argument(
        "--theta-file",
        type=Path,
        default=config.DATA_DIR / config.THETA_FILE,
        help="Theta file for pitch-angle calculations",
    )
    parser.add_argument(
        "--normalization",
        choices=["global", "ratio", "ratio2", "ratio_rescaled"],
        default="ratio2",
        help="Loss-cone normalization mode",
    )
    parser.add_argument(
        "--incident-stat",
        choices=["mean", "max"],
        default="mean",
        help="Incident flux statistic (used in ratio/global modes)",
    )
    parser.add_argument("--pitch-min", type=float, default=150.0)
    parser.add_argument("--pitch-max", type=float, default=180.0)
    parser.add_argument(
        "--min-band-points",
        type=int,
        default=3,
        help="Minimum points in pitch band per energy row",
    )
    parser.add_argument(
        "--peak-contrast",
        type=float,
        default=1.2,
        help="Peak must exceed neighbors by this multiplicative factor",
    )
    parser.add_argument(
        "--min-peak",
        type=float,
        default=1.0,
        help="Minimum normalized value to qualify as a peak",
    )
    parser.add_argument(
        "--neighbor-window",
        type=int,
        default=1,
        help="Number of energy bins on each side for peak comparison",
    )
    parser.add_argument(
        "--edge-skip",
        type=int,
        default=1,
        help="Skip this many energy bins at both ends",
    )
    parser.add_argument(
        "--no-polarity",
        action="store_true",
        help="Disable polarity filtering (legacy pitch-angle orientation)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use torch model if available",
    )
    parser.add_argument(
        "--list-peaks",
        action="store_true",
        help="Print spec_no values with detected peaks",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    er_data = ERData(str(args.er_file))
    if er_data.data.empty:
        raise RuntimeError(f"No data loaded from {args.er_file}")
    total_spec_nos = np.unique(er_data.data[config.SPEC_NO_COLUMN].to_numpy())

    session = LossConeSession(
        er_file=args.er_file,
        theta_file=args.theta_file,
        normalization_mode=args.normalization,
        incident_flux_stat=args.incident_stat,
        use_torch=args.fast,
        use_polarity=not args.no_polarity,
    )
    valid_spec_nos = np.unique(session.er_data.data[config.SPEC_NO_COLUMN].to_numpy())

    peak_specs: list[int] = []
    for chunk_idx in range(session.chunk_count()):
        chunk = session.get_chunk_data(chunk_idx)
        norm2d = session.get_norm2d(chunk_idx)
        _, profile = _build_energy_profile(
            energies=chunk.energies,
            pitches=chunk.pitches,
            norm2d=norm2d,
            pitch_min=args.pitch_min,
            pitch_max=args.pitch_max,
            min_band_points=args.min_band_points,
        )
        if _has_peak(
            profile,
            contrast=args.peak_contrast,
            min_peak=args.min_peak,
            neighbor_window=args.neighbor_window,
            edge_skip=args.edge_skip,
        ):
            peak_specs.append(int(chunk.spec_no))

    peak_specs = sorted(set(peak_specs))
    total_count = len(total_spec_nos)
    valid_count = len(valid_spec_nos)
    peak_count = len(peak_specs)

    total_pct = (peak_count / total_count * 100.0) if total_count else 0.0
    valid_pct = (peak_count / valid_count * 100.0) if valid_count else 0.0

    print("Loss Cone Peak Scan Summary")
    print("-" * 60)
    print(f"File: {args.er_file}")
    print(f"Pitch band: {args.pitch_min:.1f}–{args.pitch_max:.1f} deg")
    print(f"Normalization: {args.normalization} (incident={args.incident_stat})")
    print(f"Total spec_nos (cleaned): {total_count}")
    print(f"Valid spec_nos (polarity!=0): {valid_count}")
    print(f"Peaks detected: {peak_count}")
    print(f"Peak % of total: {total_pct:.1f}%")
    print(f"Peak % of valid: {valid_pct:.1f}%")

    if args.list_peaks:
        print("\nSpec_nos with peaks:")
        print(", ".join(str(s) for s in peak_specs) if peak_specs else "(none)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
