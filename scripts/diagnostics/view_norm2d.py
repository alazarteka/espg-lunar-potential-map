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
  uv run python scripts/diagnostics/view_norm2d.py data/.../3D*.TAB --spec-no 42 --row-mad 3 --smooth-pitch 5 --compare
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from src import config
from src.diagnostics import LossConeSession


def _odd_window(window: int) -> int:
    if window <= 1:
        return 1
    return window + 1 if window % 2 == 0 else window


def _rolling_mean_1d(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    half = window // 2
    out = np.full_like(values, np.nan, dtype=float)
    for idx in range(len(values)):
        start = max(0, idx - half)
        end = min(len(values), idx + half + 1)
        segment = values[start:end]
        finite = np.isfinite(segment)
        if np.any(finite):
            out[idx] = float(np.mean(segment[finite]))
    return out


def _apply_smooth(norm2d: np.ndarray, axis: int, window: int) -> np.ndarray:
    window = _odd_window(window)
    if window <= 1:
        return norm2d
    return np.apply_along_axis(_rolling_mean_1d, axis, norm2d, window)


def _mask_range_1d(
    values: np.ndarray, vmin: float | None, vmax: float | None
) -> np.ndarray:
    if vmin is None and vmax is None:
        return np.ones_like(values, dtype=bool)
    mask = np.ones_like(values, dtype=bool)
    if vmin is not None:
        mask &= values >= vmin
    if vmax is not None:
        mask &= values <= vmax
    return mask


def _mask_values(
    norm2d: np.ndarray,
    *,
    mask_below: float | None,
    mask_above: float | None,
) -> np.ndarray:
    if mask_below is None and mask_above is None:
        return norm2d
    out = norm2d.copy()
    mask = np.isfinite(out)
    if mask_below is not None:
        mask &= out >= mask_below
    if mask_above is not None:
        mask &= out <= mask_above
    out[~mask] = np.nan
    return out


def _row_min_finite(norm2d: np.ndarray, min_count: int) -> np.ndarray:
    if min_count <= 0:
        return norm2d
    out = norm2d.copy()
    counts = np.sum(np.isfinite(out), axis=1)
    out[counts < min_count, :] = np.nan
    return out


def _row_mad_clip(norm2d: np.ndarray, n_sigma: float) -> np.ndarray:
    if n_sigma <= 0:
        return norm2d
    out = norm2d.copy()
    for idx in range(out.shape[0]):
        row = out[idx]
        finite = np.isfinite(row)
        if not np.any(finite):
            continue
        median = float(np.nanmedian(row))
        mad = float(np.nanmedian(np.abs(row - median)))
        if not np.isfinite(mad) or mad <= 0:
            continue
        outlier_mask = np.abs(row - median) > n_sigma * mad
        out[idx, outlier_mask] = np.nan
    return out


def _finite_summary(values: np.ndarray) -> tuple[int, int, float]:
    finite = int(np.isfinite(values).sum())
    total = int(values.size)
    fraction = (finite / total) if total else 0.0
    return finite, total, fraction


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
        "--filter-energy-min",
        type=float,
        default=None,
        help="Mask energies below this value (eV)",
    )
    parser.add_argument(
        "--filter-energy-max",
        type=float,
        default=None,
        help="Mask energies above this value (eV)",
    )
    parser.add_argument(
        "--filter-pitch-min",
        type=float,
        default=None,
        help="Mask pitches below this value (deg)",
    )
    parser.add_argument(
        "--filter-pitch-max",
        type=float,
        default=None,
        help="Mask pitches above this value (deg)",
    )
    parser.add_argument(
        "--mask-below",
        type=float,
        default=None,
        help="Mask normalized flux values below this threshold",
    )
    parser.add_argument(
        "--mask-above",
        type=float,
        default=None,
        help="Mask normalized flux values above this threshold",
    )
    parser.add_argument(
        "--row-min-finite",
        type=int,
        default=0,
        help="Mask entire energy rows with fewer than N finite values",
    )
    parser.add_argument(
        "--row-mad",
        type=float,
        default=0.0,
        help="Row-wise MAD clipping (mask values beyond N * MAD)",
    )
    parser.add_argument(
        "--smooth-pitch",
        type=int,
        default=0,
        help="Rolling-mean window along pitch axis (odd window size)",
    )
    parser.add_argument(
        "--smooth-energy",
        type=int,
        default=0,
        help="Rolling-mean window along energy axis (odd window size)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show raw vs filtered panels side-by-side",
    )
    parser.add_argument(
        "--filter-report",
        action="store_true",
        help="Print filtering summary",
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
        "--vcenter",
        type=float,
        default=1.0,
        help="Center value for diverging colormap (default: 1.0)",
    )
    parser.add_argument(
        "--no-center",
        action="store_true",
        help="Disable centered colormap scaling",
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
        "--fit-method",
        choices=["halekas", "lillis"],
        default=None,
        help="Loss-cone fitting method (defaults to config)",
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
        fit_method=args.fit_method,
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

    norm2d_raw = norm2d.copy()

    # Apply filters
    energy_mask = _mask_range_1d(
        energy_axis, args.filter_energy_min, args.filter_energy_max
    )
    if not np.all(energy_mask):
        norm2d = norm2d.copy()
        norm2d[~energy_mask, :] = np.nan

    pitch_mask = _mask_range_1d(
        pitch_axis, args.filter_pitch_min, args.filter_pitch_max
    )
    if not np.all(pitch_mask):
        norm2d = norm2d.copy()
        norm2d[:, ~pitch_mask] = np.nan

    norm2d = _mask_values(
        norm2d,
        mask_below=args.mask_below,
        mask_above=args.mask_above,
    )
    norm2d = _row_min_finite(norm2d, args.row_min_finite)
    norm2d = _row_mad_clip(norm2d, args.row_mad)
    norm2d = _apply_smooth(norm2d, axis=1, window=args.smooth_pitch)
    norm2d = _apply_smooth(norm2d, axis=0, window=args.smooth_energy)

    if args.filter_report:
        raw_finite, raw_total, raw_frac = _finite_summary(norm2d_raw)
        filt_finite, filt_total, filt_frac = _finite_summary(norm2d)
        print(
            "Filtering summary: "
            f"raw={raw_finite}/{raw_total} ({raw_frac:.1%}), "
            f"filtered={filt_finite}/{filt_total} ({filt_frac:.1%})"
        )

    # Determine color scale
    vmin = args.vmin if args.vmin is not None else 0.0

    if args.vmax is not None:
        vmax = args.vmax
    else:
        # Auto-scale: use 95th percentile or 2.0, whichever is larger
        if args.compare:
            combined = np.concatenate(
                [
                    norm2d_raw[np.isfinite(norm2d_raw)],
                    norm2d[np.isfinite(norm2d)],
                ]
            )
            finite_vals = combined if len(combined) > 0 else np.array([])
        else:
            finite_vals = norm2d[np.isfinite(norm2d)]
        vmax = (
            max(2.0, np.percentile(finite_vals, 95))
            if len(finite_vals) > 0
            else 2.0
        )

    norm = None
    if not args.no_center and vmin < args.vcenter < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=args.vcenter, vmax=vmax)

    def _plot_panel(ax, data: np.ndarray, panel_title: str) -> None:
        plot_kwargs = {
            "cmap": "RdBu_r",
            "shading": "auto",
        }
        if norm is None:
            plot_kwargs["vmin"] = vmin
            plot_kwargs["vmax"] = vmax
        else:
            plot_kwargs["norm"] = norm

        im = ax.pcolormesh(
            energy_axis,
            pitch_axis,
            data.T,  # (pitch, energy)
            **plot_kwargs,
        )
        if args.show_band:
            ax.axhline(
                args.pitch_min, color="green", linestyle="--", linewidth=1.5, alpha=0.8
            )
            ax.axhline(
                args.pitch_max, color="green", linestyle="--", linewidth=1.5, alpha=0.8
            )
            ax.fill_between(
                energy_axis,
                args.pitch_min,
                args.pitch_max,
                alpha=0.1,
                color="green",
            )
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Pitch Angle (°)")
        ax.set_xscale("log")
        ax.set_ylim(0, 180)
        ax.set_title(panel_title, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.4)
        return im

    filter_labels: list[str] = []
    if args.filter_energy_min is not None or args.filter_energy_max is not None:
        filter_labels.append("energy_mask")
    if args.filter_pitch_min is not None or args.filter_pitch_max is not None:
        filter_labels.append("pitch_mask")
    if args.mask_below is not None or args.mask_above is not None:
        filter_labels.append("value_mask")
    if args.row_min_finite > 0:
        filter_labels.append(f"row_min={args.row_min_finite}")
    if args.row_mad > 0:
        filter_labels.append(f"row_mad={args.row_mad:g}")
    if args.smooth_pitch > 1:
        filter_labels.append(f"smooth_pitch={_odd_window(args.smooth_pitch)}")
    if args.smooth_energy > 1:
        filter_labels.append(f"smooth_energy={_odd_window(args.smooth_energy)}")

    filters_note = ""
    if filter_labels:
        filters_note = f"\nfilters: {', '.join(filter_labels)}"

    if args.compare:
        fig, axes = plt.subplots(
            1, 2, figsize=(12, 5), dpi=args.dpi, sharey=True, constrained_layout=True
        )
        _plot_panel(
            axes[0],
            norm2d_raw,
            (
                f"Raw\nspec_no={chunk.spec_no}  |  {chunk.timestamp}"
                f"\nnorm={args.normalization}, incident={args.incident_stat}"
            ),
        )
        im = _plot_panel(
            axes[1],
            norm2d,
            (
                f"Filtered\nspec_no={chunk.spec_no}  |  {chunk.timestamp}"
                f"\nnorm={args.normalization}, incident={args.incident_stat}"
                f"{filters_note}"
            ),
        )
    else:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=args.dpi, constrained_layout=True)
        im = _plot_panel(
            ax,
            norm2d,
            (
                f"spec_no={chunk.spec_no}  |  {chunk.timestamp}\n"
                f"norm={args.normalization}, incident={args.incident_stat}"
                f"{filters_note}"
            ),
        )

    # Colorbar
    cbar = fig.colorbar(im, ax=axes if args.compare else ax, label="Normalized Flux")
    cbar.ax.axhline(args.vcenter, color="black", linestyle="-", linewidth=1)

    if not fig.get_constrained_layout():
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
