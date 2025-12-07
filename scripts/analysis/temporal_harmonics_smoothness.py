"""
Compare temporal smoothness between independent and coupled spherical harmonic fits.

Visualizes the impact of temporal regularization on coefficient evolution.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.temporal import TemporalDataset, load_temporal_coefficients


def plot_coefficient_comparison(
    times_ind: np.ndarray,
    coeffs_ind: np.ndarray,
    times_coup: np.ndarray,
    coeffs_coup: np.ndarray,
    lmax: int,
    output_path: Path | None = None,
) -> None:
    """Compare coefficient evolution for key modes."""

    # Convert times to hours since start
    t0 = min(times_ind[0], times_coup[0])
    hours_ind = (times_ind - t0) / np.timedelta64(1, "h")
    hours_coup = (times_coup - t0) / np.timedelta64(1, "h")

    # Select key coefficients to plot
    # Index 0: (l=0, m=0) monopole
    # Index 1: (l=1, m=-1) dipole
    # Index 2: (l=1, m=0) dipole
    # Index 3: (l=1, m=1) dipole
    indices_to_plot = [0, 1, 2, 3] if lmax >= 1 else [0]
    mode_labels = ["(0,0) Monopole", "(1,-1) Dipole", "(1,0) Dipole", "(1,1) Dipole"]

    n_modes = len(indices_to_plot)
    fig, axes = plt.subplots(n_modes, 2, figsize=(14, 3 * n_modes))
    if n_modes == 1:
        axes = axes.reshape(1, -1)

    for row, (idx, label) in enumerate(zip(indices_to_plot, mode_labels, strict=False)):
        if idx >= coeffs_ind.shape[1]:
            continue

        # Real part
        axes[row, 0].plot(
            hours_ind,
            coeffs_ind[:, idx].real,
            "o-",
            label="Independent",
            alpha=0.7,
            markersize=4,
        )
        axes[row, 0].plot(
            hours_coup,
            coeffs_coup[:, idx].real,
            "s-",
            label="Coupled",
            alpha=0.7,
            markersize=4,
        )
        axes[row, 0].set_ylabel(f"Re[a_{label}] (V)")
        axes[row, 0].grid(True, alpha=0.3)
        axes[row, 0].legend(loc="best")
        if row == 0:
            axes[row, 0].set_title("Real Part")
        if row == n_modes - 1:
            axes[row, 0].set_xlabel("Time (hours since start)")

        # Imaginary part
        axes[row, 1].plot(
            hours_ind,
            coeffs_ind[:, idx].imag,
            "o-",
            label="Independent",
            alpha=0.7,
            markersize=4,
        )
        axes[row, 1].plot(
            hours_coup,
            coeffs_coup[:, idx].imag,
            "s-",
            label="Coupled",
            alpha=0.7,
            markersize=4,
        )
        axes[row, 1].set_ylabel(f"Im[a_{label}] (V)")
        axes[row, 1].grid(True, alpha=0.3)
        axes[row, 1].legend(loc="best")
        if row == 0:
            axes[row, 1].set_title("Imaginary Part")
        if row == n_modes - 1:
            axes[row, 1].set_xlabel("Time (hours since start)")

    fig.suptitle("Coefficient Evolution: Independent vs. Coupled Fitting", y=1.001)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved coefficient comparison to {output_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_temporal_roughness(
    times_ind: np.ndarray,
    coeffs_ind: np.ndarray,
    times_coup: np.ndarray,
    coeffs_coup: np.ndarray,
    output_path: Path | None = None,
) -> None:
    """Plot temporal roughness (derivative magnitude) comparison."""

    # Compute temporal differences
    diffs_ind = np.diff(coeffs_ind, axis=0)
    diffs_coup = np.diff(coeffs_coup, axis=0)

    # L2 norm of differences
    roughness_ind = np.linalg.norm(diffs_ind, axis=1)
    roughness_coup = np.linalg.norm(diffs_coup, axis=1)

    # Time axis (midpoints between windows)
    t0 = min(times_ind[0], times_coup[0])
    times_mid_ind = times_ind[:-1] + np.diff(times_ind) / 2
    times_mid_coup = times_coup[:-1] + np.diff(times_coup) / 2
    hours_mid_ind = (times_mid_ind - t0) / np.timedelta64(1, "h")
    hours_mid_coup = (times_mid_coup - t0) / np.timedelta64(1, "h")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot roughness over time
    axes[0].plot(
        hours_mid_ind, roughness_ind, "o-", label="Independent", alpha=0.7, markersize=4
    )
    axes[0].plot(
        hours_mid_coup, roughness_coup, "s-", label="Coupled", alpha=0.7, markersize=4
    )
    axes[0].set_ylabel("||a(t+1) - a(t)|| (V)")
    axes[0].set_xlabel("Time (hours since start)")
    axes[0].legend()
    axes[0].set_title("Temporal Roughness Over Time")
    axes[0].grid(True, alpha=0.3)

    # Histogram comparison
    bins = np.linspace(0, max(roughness_ind.max(), roughness_coup.max()), 30)
    axes[1].hist(
        roughness_ind,
        bins=bins,
        alpha=0.6,
        label=f"Independent (mean={roughness_ind.mean():.1f} V)",
        color="C0",
    )
    axes[1].hist(
        roughness_coup,
        bins=bins,
        alpha=0.6,
        label=f"Coupled (mean={roughness_coup.mean():.1f} V)",
        color="C1",
    )
    axes[1].set_xlabel("||a(t+1) - a(t)|| (V)")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    axes[1].set_title("Distribution of Temporal Jumps")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved roughness comparison to {output_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_residual_comparison(
    times_ind: np.ndarray,
    rms_ind: np.ndarray,
    times_coup: np.ndarray,
    rms_coup: np.ndarray,
    output_path: Path | None = None,
) -> None:
    """Compare RMS residuals between methods."""

    t0 = min(times_ind[0], times_coup[0])
    hours_ind = (times_ind - t0) / np.timedelta64(1, "h")
    hours_coup = (times_coup - t0) / np.timedelta64(1, "h")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    # Plot RMS over time
    axes[0].plot(hours_ind, rms_ind, "o-", label="Independent", alpha=0.7, markersize=4)
    axes[0].plot(hours_coup, rms_coup, "s-", label="Coupled", alpha=0.7, markersize=4)
    axes[0].set_ylabel("RMS Residual (V)")
    axes[0].set_xlabel("Time (hours since start)")
    axes[0].legend()
    axes[0].set_title("Fit Quality Over Time")
    axes[0].grid(True, alpha=0.3)

    # Histogram comparison
    bins = np.linspace(
        min(rms_ind.min(), rms_coup.min()), max(rms_ind.max(), rms_coup.max()), 25
    )
    axes[1].hist(
        rms_ind,
        bins=bins,
        alpha=0.6,
        label=f"Independent (median={np.median(rms_ind):.1f} V)",
        color="C0",
    )
    axes[1].hist(
        rms_coup,
        bins=bins,
        alpha=0.6,
        label=f"Coupled (median={np.median(rms_coup):.1f} V)",
        color="C1",
    )
    axes[1].set_xlabel("RMS Residual (V)")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    axes[1].set_title("Distribution of RMS Residuals")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved residual comparison to {output_path}")
    else:
        plt.show()

    plt.close(fig)


def print_summary_statistics(
    data_ind: TemporalDataset,
    data_coup: TemporalDataset,
) -> None:
    """Print quantitative comparison statistics."""

    # Temporal roughness
    diffs_ind = np.diff(data_ind.coeffs, axis=0)
    diffs_coup = np.diff(data_coup.coeffs, axis=0)
    roughness_ind = np.linalg.norm(diffs_ind, axis=1)
    roughness_coup = np.linalg.norm(diffs_coup, axis=1)

    print("\n" + "=" * 60)
    print("Temporal Regularization Impact Summary")
    print("=" * 60)

    print("\nTemporal Roughness:")
    print(
        f"  Independent: mean={roughness_ind.mean():.2f} V, std={roughness_ind.std():.2f} V"
    )
    print(
        f"  Coupled:     mean={roughness_coup.mean():.2f} V, std={roughness_coup.std():.2f} V"
    )
    print(
        f"  Reduction:   {(1 - roughness_coup.mean() / roughness_ind.mean()) * 100:.1f}%"
    )

    if data_ind.rms_residuals is None or data_coup.rms_residuals is None:
        print("\nRMS residual arrays missing; skipping misfit summary.")
    else:
        median_ind = float(np.median(data_ind.rms_residuals))
        median_coup = float(np.median(data_coup.rms_residuals))
        change = (median_coup / median_ind - 1) * 100 if median_ind else 0.0
        print("\nRMS Residuals:")
        print(f"  Independent: median={median_ind:.2f} V")
        print(f"  Coupled:     median={median_coup:.2f} V")
        print(f"  Change:      {change:+.1f}%")

    print("\nCoefficient Statistics (all modes):")
    print(f"  Independent: mean(|a|)={np.mean(np.abs(data_ind.coeffs)):.2f} V")
    print(f"  Coupled:     mean(|a|)={np.mean(np.abs(data_coup.coeffs)):.2f} V")

    print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare independent vs. temporally-coupled spherical harmonic fits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--independent",
        type=Path,
        required=True,
        help="NPZ file with independent fit coefficients",
    )
    parser.add_argument(
        "--coupled",
        type=Path,
        required=True,
        help="NPZ file with temporally-coupled fit coefficients",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/plots/temporal_comparison"),
        help="Directory for output plots",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display plots interactively (instead of just saving)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.independent.exists():
        print(f"Error: {args.independent} not found")
        return 1
    if not args.coupled.exists():
        print(f"Error: {args.coupled} not found")
        return 1

    print(f"Loading independent fit from {args.independent}")
    data_ind = load_temporal_coefficients(args.independent)

    print(f"Loading coupled fit from {args.coupled}")
    data_coup = load_temporal_coefficients(args.coupled)

    if data_ind.lmax != data_coup.lmax:
        print(f"Warning: lmax mismatch ({data_ind.lmax} vs {data_coup.lmax})")

    output_dir = args.output_dir if not args.display else None

    print("\nGenerating coefficient comparison plots...")
    plot_coefficient_comparison(
        data_ind.times,
        data_ind.coeffs,
        data_coup.times,
        data_coup.coeffs,
        data_ind.lmax,
        output_dir / "coefficient_comparison.png" if output_dir else None,
    )

    print("Generating temporal roughness plots...")
    plot_temporal_roughness(
        data_ind.times,
        data_ind.coeffs,
        data_coup.times,
        data_coup.coeffs,
        output_dir / "roughness_comparison.png" if output_dir else None,
    )

    print("Generating residual comparison plots...")
    plot_residual_comparison(
        data_ind.times,
        data_ind.rms_residuals,
        data_coup.times,
        data_coup.rms_residuals,
        output_dir / "residual_comparison.png" if output_dir else None,
    )

    print_summary_statistics(data_ind, data_coup)

    if output_dir:
        print(f"\nAll plots saved to {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
