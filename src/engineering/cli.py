"""Command-line interface for engineering analysis."""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..temporal.dataset import load_temporal_coefficients
from ..visualization import style
from .analysis import DEFAULT_CURRENT_DENSITY, compute_global_stats, extract_site_stats
from .sites import SITES_OF_INTEREST


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m src.engineering.cli",
        description="Generate engineering products from lunar potential model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "coeffs_file",
        type=Path,
        help="Path to temporal coefficients NPZ file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/engineering"),
        help="Directory to save maps and tables.",
    )
    parser.add_argument(
        "--current-density",
        type=float,
        default=DEFAULT_CURRENT_DENSITY,
        help="Representative current density (A/m^2) for power calc.",
    )
    parser.add_argument(
        "--lat-res",
        type=float,
        default=1.0,
        help="Latitude resolution in degrees.",
    )
    parser.add_argument(
        "--lon-res",
        type=float,
        default=1.0,
        help="Longitude resolution in degrees.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args()


def plot_global_map(
    lat: np.ndarray,
    lon: np.ndarray,
    data: np.ndarray,
    title: str,
    label: str,
    output_path: Path,
    cmap: str = style.CMAP_MAGNITUDE,
) -> None:
    """Helper to plot and save a global map."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Construct meshgrid for pcolormesh
    # data is (lat, lon), lat is -90 to 90

    im = ax.pcolormesh(lon, lat, data, shading="auto", cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, label=label)
    cbar.ax.tick_params(labelsize=style.FONT_SIZE_TEXT)

    ax.set_xlabel("Longitude (deg)", fontsize=style.FONT_SIZE_LABEL)
    ax.set_ylabel("Latitude (deg)", fontsize=style.FONT_SIZE_LABEL)
    ax.set_title(title, fontsize=style.FONT_SIZE_TITLE)
    style.apply_paper_style(ax)

    # Aspect ratio
    ax.set_aspect("equal")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved map to %s", output_path)


def main() -> int:
    """Entry point."""
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not args.coeffs_file.exists():
        logging.error("Coefficient file not found: %s", args.coeffs_file)
        return 1

    logging.info("Loading coefficients from %s", args.coeffs_file)
    dataset = load_temporal_coefficients(args.coeffs_file)

    # 1. Global Analysis
    lat_steps = int(180 / args.lat_res) + 1
    lon_steps = int(360 / args.lon_res) + 1

    stats = compute_global_stats(
        dataset,
        current_density=args.current_density,
        lat_steps=lat_steps,
        lon_steps=lon_steps,
    )

    # 2. Site Analysis
    logging.info("Analyzing %d sites of interest...", len(SITES_OF_INTEREST))
    site_results = []
    for site in SITES_OF_INTEREST:
        s_stats = extract_site_stats(dataset, site, args.current_density)
        site_results.append(
            {
                "Site": site.name,
                "Lat": site.lat,
                "Lon": site.lon,
                "Mean |U| (V)": f"{s_stats.mean_potential:.1f}",
                "P95 |U| (V)": f"{s_stats.p95_potential:.1f}",
                "Frac >1kV": f"{s_stats.frac_1kV:.2f}",
                "Mean Power (mW/m^2)": f"{s_stats.mean_power * 1e3:.2f}",
                "Assessment": s_stats.risk_assessment,
            }
        )

    df_sites = pd.DataFrame(site_results)

    # 3. Outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save Site Table
    table_path = args.output_dir / "site_analysis.csv"
    df_sites.to_csv(table_path, index=False)
    logging.info("Saved site analysis table to %s", table_path)
    print("\nSite Analysis Summary:")
    print(df_sites.to_string(index=False))
    print(f"\nSaved full table to {table_path}")

    # Save Maps
    # Mean Power
    plot_global_map(
        stats.latitudes,
        stats.longitudes,
        stats.mean_power * 1e3,
        "Expected ESPG Power Density",
        "Power Density (mW/m²)",
        args.output_dir / "map_mean_power.png",
        cmap="plasma",
    )

    # P95 Potential
    plot_global_map(
        stats.latitudes,
        stats.longitudes,
        stats.p95_potential,
        "95th Percentile Surface Potential",
        "Potential (V)",
        args.output_dir / "map_p95_potential.png",
        cmap="inferno",
    )

    # Risk Map (>1kV)
    plot_global_map(
        stats.latitudes,
        stats.longitudes,
        stats.frac_1kV,
        "Fraction of Time |U| > 1 kV",
        "Fraction",
        args.output_dir / "map_risk_1kV.png",
        cmap="Reds",
    )

    # Save raw stats as NPZ for further processing
    np.savez_compressed(
        args.output_dir / "global_stats.npz",
        latitudes=stats.latitudes,
        longitudes=stats.longitudes,
        mean_power=stats.mean_power,
        p95_potential=stats.p95_potential,
        frac_1kV=stats.frac_1kV,
    )

    # 4. Interpretive Summary
    summary_path = args.output_dir / "summary_report.md"

    # Calc global metrics for summary
    global_high_risk = np.mean(stats.frac_1kV > 0.1) * 100
    global_high_power = np.mean(stats.mean_power > 1e-3) * 100

    summary_text = f"""# ESPG Analysis Report

## Overview
Analysis based on temporal spherical harmonic model (lmax={dataset.lmax}).
Assumed current density: {args.current_density * 1e6:.1f} µA/m².

## Global Key Findings
- **High Risk Areas**: {global_high_risk:.1f}% of the lunar surface experiences potentials exceeding 1 kV for >10% of the time.
- **Resource Potential**: {global_high_power:.1f}% of the surface offers expected power densities > 1 mW/m².

## Site Highlights
{df_sites.to_markdown(index=False)}

## Interpretation
Maps indicate that while extreme charging events (>1 kV) are widespread, specific regions (e.g., near terminator or poles depending on model) offer consistent ESPG resource potential with manageable risk profiles.
"""
    summary_path.write_text(summary_text)
    logging.info("Saved summary report to %s", summary_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
