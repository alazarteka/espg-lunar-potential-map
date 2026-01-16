#!/usr/bin/env python3
"""
Visualize the normalized 2D flux grid for a specific spec_no.

Displays a heatmap of the loss-cone normalized flux with pitch angle (Y) vs
energy (X). Uses a diverging colormap centered at 1.0 to highlight excess
(>1, red) and deficit (<1, blue) flux regions.

Usage:
  uv run python scripts/diagnostics/view_norm2d.py data/.../3D*.TAB --spec-no 42
  uv run python scripts/diagnostics/view_norm2d.py data/.../3D*.TAB --spec-no 42 --save plot.png
  uv run python scripts/diagnostics/view_norm2d.py data/.../3D*.TAB --spec-no 42 --show-band
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.diagnostics import LossConeSession


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize normalized 2D flux grid for a specific spec_no."
    )
    parser.add_argument("er_file", type=Path, help="Path to ER .TAB file")
    parser.add_argument(
        "--spec-no",
        type=int,
        required=True,
        help="Spectrum number to visualize",
    )
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
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        metavar="FILE",
        help="Save to PNG file instead of displaying interactively",
    )
    parser.add_argument(
        "--show-band",
        action="store_true",
        help="Show horizontal lines at 150° and 180° (peak detection band)",
    )
    parser.add_argument(
        "--pitch-min",
        type=float,
        default=150.0,
        help="Lower bound of pitch band to highlight (with --show-band)",
    )
    parser.add_argument(
        "--pitch-max",
        type=float,
        default=180.0,
        help="Upper bound of pitch band to highlight (with --show-band)",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Minimum value for color scale",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Maximum value for color scale",
    )
    parser.add_argument(
        "--no-polarity",
        action="store_true",
        help="Disable polarity filtering",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use torch model if available",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output resolution for saved figures",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # Load session
    session = LossConeSession(
        er_file=args.er_file,
        theta_file=args.theta_file,
        normalization_mode=args.normalization,
        incident_flux_stat=args.incident_stat,
        use_torch=args.fast,
        use_polarity=not args.no_polarity,
    )

    # Look up chunk index from spec_no
    chunk_idx = session.spec_to_chunk(args.spec_no)
    if chunk_idx is None:
        print(f"Error: spec_no {args.spec_no} not found in data")
        print("  (It may have been filtered out due to zero polarity)")
        return 1

    # Get data
    chunk = session.get_chunk_data(chunk_idx)
    norm2d = session.get_norm2d(chunk_idx)

    # norm2d is (15 energies, 88 pitch bins)
    # chunk.pitches is (15, 88) - pitch angle for each (energy, bin)
    # chunk.energies is (15,)

    # For plotting, we need a representative pitch axis.
    # Use the mean pitch angle across energy rows for each bin, then sort.
    pitch_axis_raw = np.nanmean(chunk.pitches, axis=0)  # (88,)
    sort_order = np.argsort(pitch_axis_raw)
    pitch_axis = pitch_axis_raw[sort_order]
    # Reorder norm2d columns to match sorted pitch axis
    norm2d = norm2d[:, sort_order]
    energy_axis = chunk.energies  # (15,)

    # Determine color scale
    if args.vmin is not None:
        vmin = args.vmin
    else:
        vmin = 0.0

    if args.vmax is not None:
        vmax = args.vmax
    else:
        # Auto-scale: use 95th percentile or 2.0, whichever is larger
        finite_vals = norm2d[np.isfinite(norm2d)]
        if len(finite_vals) > 0:
            vmax = max(2.0, np.percentile(finite_vals, 95))
        else:
            vmax = 2.0

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=args.dpi)

    # Plot heatmap: norm2d is (energy, pitch), we want pitch on Y, energy on X
    # pcolormesh expects (X, Y, C) where C has shape (len(Y)-1, len(X)-1) or (len(Y), len(X))
    # We'll transpose norm2d to get (pitch, energy) for display
    im = ax.pcolormesh(
        energy_axis,
        pitch_axis,
        norm2d.T,  # (88 pitch, 15 energy)
        cmap="RdBu_r",  # Red = high (>1), Blue = low (<1)
        vmin=vmin,
        vmax=vmax,
        shading="auto",
    )

    # Draw band lines if requested
    if args.show_band:
        ax.axhline(args.pitch_min, color="green", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.axhline(args.pitch_max, color="green", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.fill_between(
            energy_axis,
            args.pitch_min,
            args.pitch_max,
            alpha=0.1,
            color="green",
        )

    # Formatting
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Pitch Angle (°)")
    ax.set_xscale("log")
    ax.set_ylim(0, 180)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, label="Normalized Flux")
    # Mark 1.0 on colorbar
    cbar.ax.axhline(1.0, color="black", linestyle="-", linewidth=1)

    # Title
    title = (
        f"spec_no={chunk.spec_no}  |  {chunk.timestamp}\n"
        f"norm={args.normalization}, incident={args.incident_stat}"
    )
    ax.set_title(title, fontsize=10)

    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.4)

    plt.tight_layout()

    # Save or show
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
