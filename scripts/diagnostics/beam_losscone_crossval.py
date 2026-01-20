#!/usr/bin/env python3
"""
Cross-validate beam detection ΔU estimates against loss cone fitted U_surface.

This script runs beam detection to get ΔU estimates (beam energy) and compares
them to U_surface values from loss cone fitting. The relationship is:

    beam_energy ≈ |U_spacecraft - U_surface| ≈ |U_surface| + U_spacecraft

Since U_surface is typically negative and U_spacecraft small positive (~10V),
the beam energy should approximately equal |U_surface|.

Usage:
  uv run python scripts/diagnostics/beam_losscone_crossval.py data/.../3D*.TAB

  # With custom U_spacecraft
  uv run python scripts/diagnostics/beam_losscone_crossval.py data/.../3D*.TAB --u-spacecraft 15

  # Output detailed comparison
  uv run python scripts/diagnostics/beam_losscone_crossval.py data/.../3D*.TAB --verbose
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np

from src import config
from src.diagnostics import LossConeSession
from src.model import synth_losscone_batch

from scipy.stats.qmc import LatinHypercube, scale


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
        description="Cross-validate beam detection against loss cone fitting."
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
        "--lhs-samples",
        type=int,
        default=400,
        help="Latin hypercube samples for fitting",
    )
    parser.add_argument(
        "--max-chi2",
        type=float,
        default=None,
        help="Maximum chi-squared for valid fits (default: no filter)",
    )
    parser.add_argument(
        "--max-sweeps",
        type=int,
        default=None,
        help="Limit number of sweeps to process (for quick tests)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-sweep comparison details",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use torch model if available",
    )
    parser.add_argument(
        "--fit-method",
        choices=["halekas", "lillis"],
        default=None,
        help="Loss-cone fitting method (defaults to config)",
    )
    parser.add_argument(
        "--constrain-from-beam",
        action="store_true",
        help="Constrain U_surface search to ±50%% of beam-derived estimate",
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
        fit_method=args.fit_method,
    )

    n_chunks = session.chunk_count()
    if args.max_sweeps:
        n_chunks = min(n_chunks, args.max_sweeps)

    # Collect cross-validation data
    results: list[dict] = []
    beam_width_ev = config.LOSS_CONE_BEAM_WIDTH_EV

    print(f"Processing {n_chunks} sweeps from {args.er_file.name}...")
    print(f"U_spacecraft = {args.u_spacecraft:.1f} V")
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
            energy_min=args.energy_min,
            energy_max=args.energy_max,
        )

        if not has_beam:
            continue

        # Loss cone fitting
        u_surface, bs_over_bm, beam_amp, chi2 = session.fit_chunk_lhs(
            chunk_idx,
            beam_width_ev=beam_width_ev,
            u_spacecraft=args.u_spacecraft,
            n_samples=args.lhs_samples,
        )

        if not np.isfinite(u_surface):
            continue

        # Skip poor fits
        if args.max_chi2 is not None and chi2 > args.max_chi2:
            continue

        # Compute expected beam energy from fit
        # ΔU = U_spacecraft - U_surface (U_surface is negative)
        expected_beam_energy = args.u_spacecraft - u_surface  # This is positive

        results.append({
            "spec_no": chunk.spec_no,
            "beam_energy": beam_energy,
            "beam_amplitude": beam_amplitude,
            "u_surface_fit": u_surface,
            "bs_over_bm": bs_over_bm,
            "expected_beam_energy": expected_beam_energy,
            "chi2": chi2,
        })

        if args.verbose:
            diff = beam_energy - expected_beam_energy
            pct_diff = 100.0 * diff / expected_beam_energy if expected_beam_energy > 0 else 0
            print(
                f"spec_no {chunk.spec_no:5d}: "
                f"beam={beam_energy:6.1f} eV, "
                f"fit_U={u_surface:7.1f} V, "
                f"expected={expected_beam_energy:6.1f} eV, "
                f"diff={diff:+6.1f} eV ({pct_diff:+5.1f}%), "
                f"chi2={chi2:.0f}"
            )

    if not results:
        print("No sweeps with both beam detection and valid fits.")
        return 0

    # Summary statistics
    beam_energies = np.array([r["beam_energy"] for r in results])
    expected_energies = np.array([r["expected_beam_energy"] for r in results])
    u_surface_fits = np.array([r["u_surface_fit"] for r in results])
    chi2_values = np.array([r["chi2"] for r in results])

    diff = beam_energies - expected_energies
    abs_diff = np.abs(diff)
    pct_diff = 100.0 * diff / np.maximum(expected_energies, 1.0)

    # Correlation
    valid_mask = np.isfinite(beam_energies) & np.isfinite(expected_energies)
    if np.sum(valid_mask) > 2:
        corr = np.corrcoef(beam_energies[valid_mask], expected_energies[valid_mask])[0, 1]
    else:
        corr = np.nan

    print()
    print("=" * 70)
    print("Cross-Validation Summary")
    print("=" * 70)
    print(f"File: {args.er_file}")
    print(f"Sweeps with both beam and fit: {len(results)}")
    print()
    print("Beam Energy (ΔU from peak detection):")
    print(f"  Mean:   {np.mean(beam_energies):.1f} eV")
    print(f"  Median: {np.median(beam_energies):.1f} eV")
    print(f"  Range:  {np.min(beam_energies):.1f} – {np.max(beam_energies):.1f} eV")
    print()
    print("Fitted U_surface:")
    print(f"  Mean:   {np.mean(u_surface_fits):.1f} V")
    print(f"  Median: {np.median(u_surface_fits):.1f} V")
    print(f"  Range:  {np.min(u_surface_fits):.1f} – {np.max(u_surface_fits):.1f} V")
    print()
    print("Fit Chi-squared:")
    print(f"  Mean:   {np.mean(chi2_values):.0f}")
    print(f"  Median: {np.median(chi2_values):.0f}")
    print(f"  Range:  {np.min(chi2_values):.0f} – {np.max(chi2_values):.0f}")
    print()
    print("Expected Beam Energy (U_spacecraft - U_surface):")
    print(f"  Mean:   {np.mean(expected_energies):.1f} eV")
    print(f"  Median: {np.median(expected_energies):.1f} eV")
    print(f"  Range:  {np.min(expected_energies):.1f} – {np.max(expected_energies):.1f} eV")
    print()
    print("Difference (Detected - Expected):")
    print(f"  Mean:   {np.mean(diff):+.1f} eV ({np.mean(pct_diff):+.1f}%)")
    print(f"  Median: {np.median(diff):+.1f} eV ({np.median(pct_diff):+.1f}%)")
    print(f"  MAE:    {np.mean(abs_diff):.1f} eV")
    print(f"  Std:    {np.std(diff):.1f} eV")
    print()
    print(f"Correlation (beam vs expected): {corr:.3f}")

    # Agreement thresholds
    within_50_eV = np.sum(abs_diff <= 50)
    within_100_eV = np.sum(abs_diff <= 100)
    within_30_pct = np.sum(np.abs(pct_diff) <= 30)

    print()
    print("Agreement:")
    print(f"  Within 50 eV:  {within_50_eV}/{len(results)} ({100*within_50_eV/len(results):.1f}%)")
    print(f"  Within 100 eV: {within_100_eV}/{len(results)} ({100*within_100_eV/len(results):.1f}%)")
    print(f"  Within 30%:    {within_30_pct}/{len(results)} ({100*within_30_pct/len(results):.1f}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
