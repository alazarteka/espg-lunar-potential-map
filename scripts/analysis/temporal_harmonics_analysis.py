"""
Analyze and visualize time-dependent spherical harmonic coefficients a_lm(t).

Plots coefficient evolution, spatial reconstructions at specific times, and
diagnostic metrics for the temporal expansion U(φ, θ, t) = Σ a_lm(t) Y_lm(φ, θ).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.temporal import load_temporal_coefficients, reconstruct_global_map


def _parse_iso_datetime(value: str) -> np.datetime64:
    """Parse ISO datetime string."""
    return np.datetime64(value)


def _lm_to_index(l: int, m: int) -> int:
    """Convert (l, m) indices to linear index in coefficient array."""
    return l * l + l + m


def _index_to_lm(idx: int) -> tuple[int, int]:
    """Convert linear index to (l, m) spherical harmonic indices."""
    l = int(np.floor(np.sqrt(idx)))
    m = idx - l * l - l
    return l, m


def plot_coefficient_timeseries(
    times: np.ndarray,
    coeffs: np.ndarray,
    lmax: int,
    output_dir: Path | None = None,
    selected_lm: list[tuple[int, int]] | None = None,
) -> None:
    """
    Plot time evolution of selected spherical harmonic coefficients.
    
    Args:
        times: Array of datetime64 timestamps
        coeffs: Complex array of shape (n_times, n_coeffs)
        lmax: Maximum degree
        output_dir: Directory to save plots (if None, display only)
        selected_lm: List of (l, m) pairs to plot (if None, plot all l<=2)
    """
    if selected_lm is None:
        # Default: plot monopole, dipole, quadrupole
        selected_lm = []
        for l in range(min(3, lmax + 1)):
            for m in range(-l, l + 1):
                selected_lm.append((l, m))

    n_plots = len(selected_lm)
    if n_plots == 0:
        return

    # Convert times to hours since start for x-axis
    t0 = times[0]
    hours = (times - t0) / np.timedelta64(1, "h")

    fig, axes = plt.subplots(n_plots, 2, figsize=(12, 3 * n_plots))
    if n_plots == 1:
        axes = axes.reshape(1, -1)

    for i, (l, m) in enumerate(selected_lm):
        idx = _lm_to_index(l, m)
        if idx >= coeffs.shape[1]:
            continue

        coeff_series = coeffs[:, idx]
        real_part = coeff_series.real
        imag_part = coeff_series.imag

        # Real part
        axes[i, 0].plot(hours, real_part, ".-", markersize=3, linewidth=1)
        axes[i, 0].set_ylabel(f"Re[a_{l},{m}(t)] (V)")
        axes[i, 0].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 0].set_title("Real Part")
        if i == n_plots - 1:
            axes[i, 0].set_xlabel("Time (hours since start)")

        # Imaginary part
        axes[i, 1].plot(hours, imag_part, ".-", markersize=3, linewidth=1, color="C1")
        axes[i, 1].set_ylabel(f"Im[a_{l},{m}(t)] (V)")
        axes[i, 1].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 1].set_title("Imaginary Part")
        if i == n_plots - 1:
            axes[i, 1].set_xlabel("Time (hours since start)")

    fig.suptitle(f"Time Evolution of Spherical Harmonic Coefficients (lmax={lmax})")
    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "coefficient_timeseries.png", dpi=150)
        print(f"Saved coefficient time series to {output_dir / 'coefficient_timeseries.png'}")
    else:
        plt.show()

    plt.close(fig)


def plot_rms_evolution(
    times: np.ndarray,
    rms_residuals: np.ndarray,
    n_samples: np.ndarray,
    spatial_coverage: np.ndarray,
    output_dir: Path | None = None,
) -> None:
    """Plot temporal evolution of fit quality metrics."""
    t0 = times[0]
    hours = (times - t0) / np.timedelta64(1, "h")

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    # RMS residual
    axes[0].plot(hours, rms_residuals, ".-", markersize=3)
    axes[0].set_ylabel("RMS Residual (V)")
    axes[0].set_title("Fit Quality Evolution")
    axes[0].grid(True, alpha=0.3)

    # Number of samples
    axes[1].plot(hours, n_samples, ".-", markersize=3, color="C1")
    axes[1].set_ylabel("Number of Samples")
    axes[1].grid(True, alpha=0.3)

    # Spatial coverage
    axes[2].plot(hours, spatial_coverage * 100, ".-", markersize=3, color="C2")
    axes[2].set_ylabel("Spatial Coverage (%)")
    axes[2].set_xlabel("Time (hours since start)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "fit_quality_evolution.png", dpi=150)
        print(f"Saved fit quality to {output_dir / 'fit_quality_evolution.png'}")
    else:
        plt.show()

    plt.close(fig)


def plot_snapshot_maps(
    times: np.ndarray,
    coeffs: np.ndarray,
    lmax: int,
    snapshot_indices: list[int],
    output_dir: Path | None = None,
) -> None:
    """Plot reconstructed global maps at selected time snapshots."""
    n_snapshots = len(snapshot_indices)
    if n_snapshots == 0:
        return

    fig, axes = plt.subplots(1, n_snapshots, figsize=(6 * n_snapshots, 5))
    if n_snapshots == 1:
        axes = [axes]

    for ax_idx, time_idx in enumerate(snapshot_indices):
        if time_idx >= len(times):
            continue

        lats, lons, potential = reconstruct_global_map(coeffs[time_idx], lmax)

        mesh = axes[ax_idx].pcolormesh(
            lons,
            lats,
            potential,
            shading="auto",
            cmap="viridis",
        )
        axes[ax_idx].set_xlabel("Longitude (°)")
        axes[ax_idx].set_ylabel("Latitude (°)")
        axes[ax_idx].set_title(f"t = {times[time_idx]}")
        axes[ax_idx].set_aspect("equal")
        fig.colorbar(mesh, ax=axes[ax_idx], label="Φ_surface (V)")

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "snapshot_maps.png", dpi=150)
        print(f"Saved snapshot maps to {output_dir / 'snapshot_maps.png'}")
    else:
        plt.show()

    plt.close(fig)


def plot_coefficient_spectrum(
    coeffs: np.ndarray,
    lmax: int,
    output_dir: Path | None = None,
) -> None:
    """Plot average power spectrum by degree l."""
    # Average over time
    coeffs_mean = np.mean(np.abs(coeffs), axis=0)

    # Bin by degree
    power_by_degree = []
    for l in range(lmax + 1):
        indices = [_lm_to_index(l, m) for m in range(-l, l + 1)]
        power_l = np.mean(coeffs_mean[indices])
        power_by_degree.append(power_l)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(lmax + 1), power_by_degree, alpha=0.7)
    ax.set_xlabel("Degree l")
    ax.set_ylabel("Average |a_lm| (V)")
    ax.set_title("Spherical Harmonic Power Spectrum (time-averaged)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(range(lmax + 1))

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "power_spectrum.png", dpi=150)
        print(f"Saved power spectrum to {output_dir / 'power_spectrum.png'}")
    else:
        plt.show()

    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze time-dependent spherical harmonic coefficients",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input NPZ file with temporal coefficients",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/temporal_harmonics"),
        help="Directory for output plots",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display plots interactively (instead of just saving)",
    )
    parser.add_argument(
        "--snapshot-times",
        nargs="+",
        type=int,
        default=None,
        help="Time indices for snapshot maps (e.g., 0 10 20)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        print(f"Error: Input file {args.input} not found")
        return 1

    print(f"Loading temporal coefficients from {args.input}")
    dataset = load_temporal_coefficients(args.input)

    times = dataset.times
    coeffs = dataset.coeffs
    lmax = dataset.lmax
    n_samples = dataset.n_samples
    coverage = dataset.spatial_coverage
    rms = dataset.rms_residuals
    if n_samples is None or coverage is None or rms is None:
        raise ValueError("Dataset is missing quality metrics required for analysis plots")

    print(f"\nDataset Summary:")
    print(f"  Time range     : {times[0]} → {times[-1]}")
    print(f"  Number of windows : {len(times)}")
    print(f"  Max degree     : lmax = {lmax}")
    print(f"  Coefficients   : {coeffs.shape[1]}")
    print(f"  Mean RMS       : {np.mean(rms):.2f} V")
    print(f"  Mean coverage  : {np.mean(coverage)*100:.1f}%")

    output_dir = args.output_dir if not args.display else None

    # Plot coefficient time series
    print("\nGenerating coefficient time series plots...")
    plot_coefficient_timeseries(times, coeffs, lmax, output_dir)

    # Plot fit quality evolution
    print("Generating fit quality plots...")
    plot_rms_evolution(times, rms, n_samples, coverage, output_dir)

    # Plot power spectrum
    print("Generating power spectrum...")
    plot_coefficient_spectrum(coeffs, lmax, output_dir)

    # Plot snapshot maps
    if args.snapshot_times:
        print(f"Generating snapshot maps for times {args.snapshot_times}...")
        plot_snapshot_maps(times, coeffs, lmax, args.snapshot_times, output_dir)
    else:
        # Default: first, middle, last
        n_times = len(times)
        snapshots = [0, n_times // 2, n_times - 1]
        print(f"Generating default snapshot maps (indices {snapshots})...")
        plot_snapshot_maps(times, coeffs, lmax, snapshots, output_dir)

    if output_dir:
        print(f"\nAll plots saved to {output_dir}")
    
    print("\nAnalysis complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
