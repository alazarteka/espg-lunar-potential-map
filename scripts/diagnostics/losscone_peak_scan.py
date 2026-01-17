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
from src.diagnostics import LossConeSession, _build_energy_profile, detect_peak
from src.flux import ERData


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
        default=5,
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
        default=2.0,
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
        "--min-neighbor",
        type=float,
        default=1.5,
        help="Minimum normalized value for adjacent bin to confirm peak",
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
        nargs="?",
        const=10,
        type=int,
        metavar="N",
        help="Print spec_no values with detected peaks (default: first 10, use 0 for all)",
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

    # Track detected peaks: list of (spec_no, beam_energy, peak_amplitude)
    peak_data: list[tuple[int, float, float]] = []

    for chunk_idx in range(session.chunk_count()):
        chunk = session.get_chunk_data(chunk_idx)
        norm2d = session.get_norm2d(chunk_idx)
        energies_sorted, profile = _build_energy_profile(
            energies=chunk.energies,
            pitches=chunk.pitches,
            norm2d=norm2d,
            pitch_min=args.pitch_min,
            pitch_max=args.pitch_max,
            min_band_points=args.min_band_points,
        )
        result = detect_peak(
            profile,
            contrast=args.peak_contrast,
            min_peak=args.min_peak,
            neighbor_window=args.neighbor_window,
            edge_skip=args.edge_skip,
            min_neighbor=args.min_neighbor,
        )
        if result.has_peak and result.peak_idx is not None and result.peak_value is not None:
            beam_energy = float(energies_sorted[result.peak_idx])
            peak_data.append((int(chunk.spec_no), beam_energy, result.peak_value))

    # Sort by spec_no
    peak_data.sort(key=lambda x: x[0])

    total_count = len(total_spec_nos)
    valid_count = len(valid_spec_nos)
    peak_count = len(peak_data)

    total_pct = (peak_count / total_count * 100.0) if total_count else 0.0
    valid_pct = (peak_count / valid_count * 100.0) if valid_count else 0.0

    print("Loss Cone Peak Scan Summary")
    print("-" * 60)
    print(f"File: {args.er_file}")
    print(f"Pitch band: {args.pitch_min:.1f}–{args.pitch_max:.1f} deg")
    print(f"Normalization: {args.normalization} (incident={args.incident_stat})")
    print(f"Thresholds: min_peak={args.min_peak}, min_neighbor={args.min_neighbor}")
    print(f"Total spec_nos (cleaned): {total_count}")
    print(f"Valid spec_nos (polarity!=0): {valid_count}")
    print(f"Peaks detected: {peak_count}")
    print(f"Peak % of total: {total_pct:.1f}%")
    print(f"Peak % of valid: {valid_pct:.1f}%")

    # Beam energy summary
    if peak_count > 0:
        beam_energies = [p[1] for p in peak_data]
        peak_amplitudes = [p[2] for p in peak_data]
        print()
        print("Beam Energy Summary (ΔU estimates):")
        print(f"  Mean:   {np.mean(beam_energies):.1f} eV")
        print(f"  Median: {np.median(beam_energies):.1f} eV")
        print(f"  Range:  {np.min(beam_energies):.1f} – {np.max(beam_energies):.1f} eV")
        print(f"  Peak amplitude range: {np.min(peak_amplitudes):.2f} – {np.max(peak_amplitudes):.2f}")

    if args.list_peaks is not None:
        if args.list_peaks == 0 or len(peak_data) <= args.list_peaks:
            shown = peak_data
            suffix = ""
        else:
            shown = peak_data[: args.list_peaks]
            suffix = f"\n  ... ({len(peak_data) - args.list_peaks} more, use --list-peaks 0 for all)"
        print("\nDetected peaks (spec_no: energy, amplitude):")
        if shown:
            for spec_no, energy, amplitude in shown:
                print(f"  {spec_no}: {energy:.1f} eV, {amplitude:.2f}")
            print(suffix, end="")
        else:
            print("  (none)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
