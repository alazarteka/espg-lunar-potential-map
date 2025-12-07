"""
Analyze azimuthal (m) mode structure in spherical harmonic coefficients.

Investigates the distribution of power across different m values to understand
longitudinal structure and potential regularization artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.temporal import load_temporal_coefficients


def analyze_azimuthal_power(
    coeffs: np.ndarray,
    lmax: int,
    times: np.ndarray,
) -> dict:
    """
    Compute power in each azimuthal order m across all degrees l.

    Returns dict with:
        - m_values: array of m values analyzed
        - power_by_m: time-averaged RMS power for each |m|
        - power_by_m_temporal: shape (n_times, n_m) power evolution
        - mode_counts: number of (l,m) pairs for each |m|
    """
    n_times = coeffs.shape[0]
    max_m = lmax

    # Storage for results
    m_values = np.arange(0, max_m + 1)
    power_by_m_temporal = np.zeros((n_times, len(m_values)))
    mode_counts = np.zeros(len(m_values), dtype=int)

    # Compute power for each |m|
    for m_idx, m_target in enumerate(m_values):
        idx = 0
        powers_this_m = []

        for l in range(lmax + 1):
            for m in range(-l, l + 1):
                if abs(m) == m_target:
                    # RMS power across time for this specific (l,m)
                    powers_this_m.append(np.abs(coeffs[:, idx]))
                    mode_counts[m_idx] += 1
                idx += 1

        if powers_this_m:
            # Average power across all (l, ±m) pairs for this |m|
            powers_array = np.array(powers_this_m)  # shape: (n_modes, n_times)
            power_by_m_temporal[:, m_idx] = np.mean(powers_array, axis=0)

    power_by_m = np.mean(power_by_m_temporal, axis=0)

    return {
        "m_values": m_values,
        "power_by_m": power_by_m,
        "power_by_m_temporal": power_by_m_temporal,
        "mode_counts": mode_counts,
        "times": times,
    }


def analyze_degree_by_order(
    coeffs: np.ndarray,
    lmax: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create 2D power spectrum: power as function of (l, |m|).

    Returns:
        (l_grid, m_grid, power_grid) where power_grid[i,j] is power at (l=i, |m|=j)
    """
    power_grid = np.zeros((lmax + 1, lmax + 1))
    count_grid = np.zeros((lmax + 1, lmax + 1), dtype=int)

    idx = 0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            abs_m = abs(m)
            # Time-averaged RMS power
            power = np.mean(np.abs(coeffs[:, idx]))
            power_grid[l, abs_m] += power
            count_grid[l, abs_m] += 1
            idx += 1

    # Average over ±m
    mask = count_grid > 0
    power_grid[mask] /= count_grid[mask]

    return power_grid


def plot_azimuthal_power_distribution(
    analysis: dict,
    output_path: Path | None = None,
) -> None:
    """Plot power distribution across azimuthal orders."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    m_values = analysis["m_values"]
    power_by_m = analysis["power_by_m"]
    mode_counts = analysis["mode_counts"]

    # 1. Bar chart of power by |m|
    axes[0, 0].bar(m_values, power_by_m, alpha=0.7, edgecolor="black")
    axes[0, 0].set_xlabel("Azimuthal order |m|", fontsize=12)
    axes[0, 0].set_ylabel("Mean power (V)", fontsize=12)
    axes[0, 0].set_title("Power Distribution by Azimuthal Order", fontsize=13)
    axes[0, 0].grid(True, alpha=0.3, axis="y")
    axes[0, 0].set_xticks(m_values)

    # Annotate with mode counts
    for i, (m, count) in enumerate(zip(m_values, mode_counts, strict=False)):
        if i % 2 == 0 or count > 0:  # Label every other or if has modes
            axes[0, 0].text(
                m,
                power_by_m[i],
                f"n={count}",
                ha="center",
                va="bottom",
                fontsize=8,
                alpha=0.7,
            )

    # 2. Normalized power (accounting for number of modes)
    power_per_mode = power_by_m / np.maximum(mode_counts, 1)
    axes[0, 1].bar(m_values, power_per_mode, alpha=0.7, edgecolor="black", color="C1")
    axes[0, 1].set_xlabel("Azimuthal order |m|", fontsize=12)
    axes[0, 1].set_ylabel("Power per mode (V)", fontsize=12)
    axes[0, 1].set_title("Normalized Power (per mode)", fontsize=13)
    axes[0, 1].grid(True, alpha=0.3, axis="y")
    axes[0, 1].set_xticks(m_values)

    # 3. Temporal evolution of select m values
    times = analysis["times"]
    t0 = times[0]
    hours = (times - t0) / np.timedelta64(1, "h")

    power_temporal = analysis["power_by_m_temporal"]

    # Plot m=0, and a few others
    m_to_plot = [0, 1, 2, 3, 5]
    for m in m_to_plot:
        if m < len(m_values):
            label = "m=0 (zonal)" if m == 0 else f"m=±{m}"
            axes[1, 0].plot(
                hours, power_temporal[:, m], "o-", label=label, alpha=0.7, markersize=3
            )

    axes[1, 0].set_xlabel("Time (hours since start)", fontsize=12)
    axes[1, 0].set_ylabel("Mean power (V)", fontsize=12)
    axes[1, 0].set_title("Temporal Evolution of Power by m", fontsize=13)
    axes[1, 0].legend(loc="best", fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Cumulative power fraction
    cumsum = np.cumsum(power_by_m)
    cumsum_normalized = cumsum / cumsum[-1] * 100

    axes[1, 1].plot(m_values, cumsum_normalized, "o-", linewidth=2, markersize=6)
    axes[1, 1].axhline(50, color="red", linestyle="--", alpha=0.5, label="50% power")
    axes[1, 1].axhline(90, color="orange", linestyle="--", alpha=0.5, label="90% power")
    axes[1, 1].set_xlabel("Maximum |m|", fontsize=12)
    axes[1, 1].set_ylabel("Cumulative power (%)", fontsize=12)
    axes[1, 1].set_title("Cumulative Power Distribution", fontsize=13)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(loc="best")
    axes[1, 1].set_xticks(m_values)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved azimuthal power analysis to {output_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_degree_order_spectrum(
    coeffs: np.ndarray,
    lmax: int,
    output_path: Path | None = None,
) -> None:
    """Plot 2D spectrum showing power as function of (l, |m|)."""

    power_grid = analyze_degree_by_order(coeffs, lmax)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Mask zero values for better visualization
    power_masked = np.ma.masked_where(power_grid == 0, power_grid)

    im = ax.imshow(
        power_masked,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean power (V)", fontsize=12)

    # Labels
    ax.set_xlabel("Azimuthal order |m|", fontsize=12)
    ax.set_ylabel("Degree l", fontsize=12)
    ax.set_title("Power Spectrum: Degree l vs Order |m|", fontsize=13)

    # Add diagonal line (sectoral harmonics l = |m|)
    ax.plot(
        [0, lmax], [0, lmax], "r--", alpha=0.5, linewidth=2, label="Sectoral (l=|m|)"
    )

    # Add vertical line at m=0 (zonal harmonics)
    ax.axvline(
        0, color="cyan", linestyle="--", alpha=0.5, linewidth=2, label="Zonal (m=0)"
    )

    ax.legend(loc="upper right")
    ax.set_xticks(range(0, lmax + 1, max(1, lmax // 10)))
    ax.set_yticks(range(0, lmax + 1, max(1, lmax // 10)))

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved degree-order spectrum to {output_path}")
    else:
        plt.show()

    plt.close(fig)


def print_mode_statistics(analysis: dict, lmax: int) -> None:
    """Print detailed statistics about mode distribution."""

    m_values = analysis["m_values"]
    power_by_m = analysis["power_by_m"]
    mode_counts = analysis["mode_counts"]

    print("\n" + "=" * 70)
    print("AZIMUTHAL MODE ANALYSIS")
    print("=" * 70)

    total_power = np.sum(power_by_m)

    print(f"\nDegree range: l = 0 to {lmax}")
    print(f"Total modes: {np.sum(mode_counts)}")
    print(f"Total power: {total_power:.1f} V")
    print()

    print("Power by azimuthal order:")
    print("-" * 70)
    print(
        f"{'|m|':<5} {'Modes':<8} {'Power (V)':<12} {'% Total':<10} {'Power/Mode':<12}"
    )
    print("-" * 70)

    for m, count, power in zip(m_values, mode_counts, power_by_m, strict=False):
        pct = (power / total_power * 100) if total_power > 0 else 0
        per_mode = power / count if count > 0 else 0
        label = "ZONAL" if m == 0 else ""
        print(
            f"{m:<5} {count:<8} {power:<12.1f} {pct:<10.1f} {per_mode:<12.1f} {label}"
        )

    print("-" * 70)

    # Find which m has most power
    max_m = m_values[np.argmax(power_by_m)]
    print(f"\nDominant azimuthal order: |m| = {max_m}")

    # Zonal vs non-zonal
    zonal_power = power_by_m[0]
    nonzonal_power = total_power - zonal_power
    print(
        f"\nZonal (m=0) power:     {zonal_power:8.1f} V ({zonal_power / total_power * 100:5.1f}%)"
    )
    print(
        f"Non-zonal (m≠0) power: {nonzonal_power:8.1f} V ({nonzonal_power / total_power * 100:5.1f}%)"
    )

    # Sectoral vs tesseral
    sectoral_power = 0.0
    tesseral_power = 0.0

    idx = 0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            power = np.mean(np.abs(analysis["power_by_m_temporal"][:, abs(m)]))
            if abs(m) == l and m != 0:
                sectoral_power += power
            elif abs(m) != 0 and abs(m) != l:
                tesseral_power += power
            idx += 1

    print(f"\nSectoral (l=|m|, m≠0):  {sectoral_power:8.1f} V")
    print(f"Tesseral (l≠|m|, m≠0):  {tesseral_power:8.1f} V")

    print("=" * 70 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze azimuthal mode structure in spherical harmonic coefficients",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="NPZ file with temporal coefficients",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/plots/azimuthal_analysis"),
        help="Directory for output plots",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found")
        return 1

    print(f"Loading temporal coefficients from {args.input}")
    dataset = load_temporal_coefficients(args.input)
    coeffs = dataset.coeffs
    lmax = dataset.lmax
    times = dataset.times

    print(f"Dataset: {len(times)} windows, lmax={lmax}, {coeffs.shape[1]} coefficients")

    # Perform analysis
    print("\nAnalyzing azimuthal mode structure...")
    analysis = analyze_azimuthal_power(coeffs, lmax, times)

    # Print statistics
    print_mode_statistics(analysis, lmax)

    # Generate plots
    print("Generating azimuthal power plots...")
    plot_azimuthal_power_distribution(
        analysis,
        args.output_dir / "azimuthal_power.png",
    )

    print("Generating degree-order spectrum...")
    plot_degree_order_spectrum(
        coeffs,
        lmax,
        args.output_dir / "degree_order_spectrum.png",
    )

    print(f"\nAll plots saved to {args.output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
