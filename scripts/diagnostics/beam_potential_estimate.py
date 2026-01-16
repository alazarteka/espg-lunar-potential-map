#!/usr/bin/env python3
"""
Estimate lunar surface potential directly from secondary electron beam detection.

This script bypasses the complex loss cone fitter and estimates U_surface directly
from the beam peak energy using the simple relationship:

    U_surface = U_spacecraft - beam_energy

The beam energy is detected as a peak in the high-pitch-angle (150-180°) normalized
flux, where secondary electrons emitted from the surface appear after being
accelerated through the potential difference.

Usage:
  uv run python scripts/diagnostics/beam_potential_estimate.py data/.../3D*.TAB

  # With custom U_spacecraft
  uv run python scripts/diagnostics/beam_potential_estimate.py data/.../3D*.TAB --u-spacecraft 15

  # Output per-sweep estimates
  uv run python scripts/diagnostics/beam_potential_estimate.py data/.../3D*.TAB --verbose
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np

from src import config
from src.diagnostics import LossConeSession


def _build_energy_profile(
    energies: np.ndarray,
    pitches: np.ndarray,
    norm2d: np.ndarray,
    pitch_min: float,
    pitch_max: float,
    min_band_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build 1D energy profile by averaging over high-pitch band."""
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


def _detect_beam(
    profile: np.ndarray,
    energies: np.ndarray,
    *,
    min_peak: float = 2.0,
    min_neighbor: float = 1.5,
    contrast: float = 1.2,
    neighbor_window: int = 1,
    edge_skip: int = 1,
    energy_min: float = 20.0,
    energy_max: float = 500.0,
) -> tuple[bool, float | None, float | None]:
    """
    Detect beam peak in energy profile.

    Returns:
        Tuple of (has_beam, beam_energy, peak_amplitude)
    """
    if profile.size == 0:
        return False, None, None

    window = max(1, neighbor_window)
    start = max(edge_skip, window)
    end = profile.size - max(edge_skip, window)
    if end <= start:
        return False, None, None

    best_idx: int | None = None
    best_value: float = -np.inf

    for idx in range(start, end):
        value = profile[idx]
        energy = energies[idx]

        # Skip if outside energy window
        if energy < energy_min or energy > energy_max:
            continue

        if not np.isfinite(value) or value < min_peak:
            continue

        left = profile[idx - window : idx]
        right = profile[idx + 1 : idx + 1 + window]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            left_max = np.nanmax(left) if left.size else np.nan
            right_max = np.nanmax(right) if right.size else np.nan

        if not np.isfinite(left_max) or not np.isfinite(right_max):
            continue

        # Check contrast requirement
        if not (value >= left_max * contrast and value >= right_max * contrast):
            continue

        # Check neighbor threshold requirement
        if left_max < min_neighbor and right_max < min_neighbor:
            continue

        # This is a valid peak; track the best one
        if value > best_value:
            best_value = value
            best_idx = idx

    if best_idx is not None:
        return True, float(energies[best_idx]), float(best_value)
    return False, None, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate surface potential from beam detection."
    )
    parser.add_argument("er_file", type=Path, help="Path to ER .TAB file")
    parser.add_argument(
        "--theta-file",
        type=Path,
        default=config.DATA_DIR / config.THETA_FILE,
        help="Theta file for pitch-angle calculations",
    )
    parser.add_argument(
        "--u-spacecraft",
        type=float,
        default=10.0,
        help="Spacecraft potential in volts (default: 10)",
    )
    parser.add_argument(
        "--normalization",
        choices=["global", "ratio", "ratio2", "ratio_rescaled"],
        default="ratio2",
        help="Loss-cone normalization mode",
    )
    parser.add_argument(
        "--min-peak",
        type=float,
        default=2.0,
        help="Minimum normalized flux for beam detection",
    )
    parser.add_argument(
        "--min-neighbor",
        type=float,
        default=1.5,
        help="Minimum neighbor threshold for peak validation",
    )
    parser.add_argument(
        "--energy-min",
        type=float,
        default=20.0,
        help="Minimum beam energy to consider (eV)",
    )
    parser.add_argument(
        "--energy-max",
        type=float,
        default=500.0,
        help="Maximum beam energy to consider (eV)",
    )
    parser.add_argument(
        "--max-sweeps",
        type=int,
        default=None,
        help="Limit number of sweeps to process",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-sweep estimates",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use torch model if available",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Write results to CSV file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    session = LossConeSession(
        er_file=args.er_file,
        theta_file=args.theta_file,
        normalization_mode=args.normalization,
        incident_flux_stat="mean",
        use_torch=args.fast,
        use_polarity=True,
    )

    n_chunks = session.chunk_count()
    if args.max_sweeps:
        n_chunks = min(n_chunks, args.max_sweeps)

    # Collect estimates
    results: list[dict] = []

    print(f"Processing {n_chunks} sweeps from {args.er_file.name}...")
    print(f"U_spacecraft = {args.u_spacecraft:.1f} V")
    print(f"Detection: min_peak={args.min_peak}, energy=[{args.energy_min}, {args.energy_max}] eV")
    print()

    for chunk_idx in range(n_chunks):
        chunk = session.get_chunk_data(chunk_idx)
        norm2d = session.get_norm2d(chunk_idx)

        # Beam detection
        energies_sorted, profile = _build_energy_profile(
            energies=chunk.energies,
            pitches=chunk.pitches,
            norm2d=norm2d,
            pitch_min=150.0,
            pitch_max=180.0,
            min_band_points=5,
        )

        has_beam, beam_energy, beam_amplitude = _detect_beam(
            profile,
            energies_sorted,
            min_peak=args.min_peak,
            min_neighbor=args.min_neighbor,
            energy_min=args.energy_min,
            energy_max=args.energy_max,
        )

        if not has_beam or beam_energy is None:
            continue

        # Direct U_surface estimate: U_surface = U_spacecraft - beam_energy
        u_surface_estimate = args.u_spacecraft - beam_energy

        results.append({
            "spec_no": chunk.spec_no,
            "timestamp": chunk.timestamp,
            "beam_energy": beam_energy,
            "beam_amplitude": beam_amplitude,
            "u_surface": u_surface_estimate,
        })

        if args.verbose:
            print(
                f"spec_no {chunk.spec_no:5d}: "
                f"beam={beam_energy:6.1f} eV, "
                f"amp={beam_amplitude:4.2f}, "
                f"U_surface={u_surface_estimate:7.1f} V"
            )

    if not results:
        print("No beams detected.")
        return 0

    # Summary statistics
    beam_energies = np.array([r["beam_energy"] for r in results])
    u_surface_values = np.array([r["u_surface"] for r in results])
    amplitudes = np.array([r["beam_amplitude"] for r in results])

    print()
    print("=" * 70)
    print("Beam-Based Surface Potential Estimates")
    print("=" * 70)
    print(f"File: {args.er_file}")
    print(f"Total sweeps: {n_chunks}")
    print(f"Beams detected: {len(results)} ({100*len(results)/n_chunks:.1f}%)")
    print()
    print("Beam Energy:")
    print(f"  Mean:   {np.mean(beam_energies):.1f} eV")
    print(f"  Median: {np.median(beam_energies):.1f} eV")
    print(f"  Std:    {np.std(beam_energies):.1f} eV")
    print(f"  Range:  {np.min(beam_energies):.1f} – {np.max(beam_energies):.1f} eV")
    print()
    print("Peak Amplitude:")
    print(f"  Mean:   {np.mean(amplitudes):.2f}")
    print(f"  Median: {np.median(amplitudes):.2f}")
    print(f"  Range:  {np.min(amplitudes):.2f} – {np.max(amplitudes):.2f}")
    print()
    print(f"Surface Potential (U_surface = {args.u_spacecraft:.0f} - beam_energy):")
    print(f"  Mean:   {np.mean(u_surface_values):.1f} V")
    print(f"  Median: {np.median(u_surface_values):.1f} V")
    print(f"  Std:    {np.std(u_surface_values):.1f} V")
    print(f"  Range:  {np.min(u_surface_values):.1f} – {np.max(u_surface_values):.1f} V")

    # Histogram of U_surface by bin
    print()
    print("U_surface distribution:")
    bins = [(-500, -200), (-200, -100), (-100, -50), (-50, -20), (-20, 0)]
    for low, high in bins:
        count = np.sum((u_surface_values >= low) & (u_surface_values < high))
        pct = 100 * count / len(u_surface_values)
        bar = "#" * int(pct / 2)
        print(f"  [{low:4d}, {high:4d}) V: {count:4d} ({pct:5.1f}%) {bar}")

    # Output CSV if requested
    if args.output_csv:
        import csv
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["spec_no", "timestamp", "beam_energy", "beam_amplitude", "u_surface"])
            writer.writeheader()
            writer.writerows(results)
        print()
        print(f"Results written to: {args.output_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
