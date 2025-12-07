#!/usr/bin/env python
"""
Visualize observed vs. modelled loss-cone flux for a single measurement chunk.

Usage
-----
uv run python scripts/dev/losscone_plot_fit.py \
    --file data/1998/060_090MAR/3D980323.TAB \
    --chunk 0 \
    --theta-file data/theta.tab \
    --output scratch/losscone_chunk0.png

The script computes the normalized flux matrix used by LossConeFitter,
reconstructs the best-fit model for the specified chunk, and renders:
  (a) observed normalized flux,
  (b) synthetic loss-cone model (including beam),
  (c) residual (observed - model).
All panels share a logarithmic energy axis (x) and pitch-angle axis (y).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.flux import ERData, LossConeFitter
from src.model import synth_losscone


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot loss-cone fit for a chunk.")
    parser.add_argument(
        "--file",
        required=True,
        type=Path,
        help="Path to ER .TAB file.",
    )
    parser.add_argument(
        "--chunk",
        required=True,
        type=int,
        help="Chunk index (0-based) to visualize.",
    )
    parser.add_argument(
        "--theta-file",
        type=Path,
        default=config.DATA_DIR / config.THETA_FILE,
        help="Theta table used for pitch-angle calculations.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the figure; otherwise show interactively.",
    )
    parser.add_argument(
        "--dump-dir",
        type=Path,
        default=None,
        help="Optional directory to dump observed/model/residual grids for debugging.",
    )
    return parser.parse_args()


def _get_chunk_ranges(total_rows: int, chunk_idx: int) -> slice:
    start = chunk_idx * config.SWEEP_ROWS
    end = min((chunk_idx + 1) * config.SWEEP_ROWS, total_rows)
    if start >= total_rows:
        raise IndexError(f"Chunk {chunk_idx} out of range for {total_rows} rows.")
    return slice(start, end)


def main() -> int:
    args = _parse_args()

    er = ERData(str(args.file))
    fitter = LossConeFitter(er, str(args.theta_file))

    total_rows = len(er.data)
    chunk_slice = _get_chunk_ranges(total_rows, args.chunk)

    norm2d = fitter.build_norm2d(args.chunk)
    if np.isnan(norm2d).all():
        raise RuntimeError(f"Chunk {args.chunk} contains no valid flux.")

    # Fit to retrieve the best parameters.
    U_surface, bs_over_bm, beam_amp, chi2 = fitter._fit_surface_potential(args.chunk)
    print(
        f"Chunk {args.chunk}: U_surface={U_surface:.2f} V, Bs/Bm={bs_over_bm:.3f}, "
        f"beam_amp={beam_amp:.3f}, χ²={chi2:.3g}"
    )

    energies = er.data[config.ENERGY_COLUMN].to_numpy(dtype=np.float64)[chunk_slice]
    pitches = fitter.pitch_angle.pitch_angles[chunk_slice]

    beam_width = max(abs(U_surface) * fitter.beam_width_factor, config.EPS)
    model = synth_losscone(
        energies,
        pitches,
        U_surface,
        bs_over_bm,
        beam_width_eV=beam_width,
        beam_amp=beam_amp,
        beam_pitch_sigma_deg=fitter.beam_pitch_sigma_deg,
        background=fitter.background,
    )

    residual = norm2d - model
    residual_title = "Residual (normalized obs − model)"

    if args.dump_dir:
        args.dump_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            args.dump_dir / f"chunk{args.chunk}_energies.csv",
            energies.reshape(-1, 1),
            delimiter=",",
            header="energy_eV",
        )
        np.savetxt(
            args.dump_dir / f"chunk{args.chunk}_pitch.csv",
            pitches,
            delimiter=",",
            header="pitch_deg_channels",
        )
        np.savetxt(
            args.dump_dir / f"chunk{args.chunk}_observed_norm.csv",
            norm2d,
            delimiter=",",
        )
        np.savetxt(
            args.dump_dir / f"chunk{args.chunk}_model_norm.csv",
            model,
            delimiter=",",
        )
        np.savetxt(
            args.dump_dir / f"chunk{args.chunk}_residual_norm.csv",
            residual,
            delimiter=",",
        )
        print(f"Wrote debug grids to {args.dump_dir}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    titles = [
        f"Observed (chunk {args.chunk})",
        f"Model U_surface={U_surface:.1f}V, Bs/Bm={bs_over_bm:.2f}",
        residual_title,
    ]
    datasets = [norm2d, model, residual]

    # Build edges for log-energy (x) and pitch (y) axes.
    def _edge_array(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.size == 1:
            v = values[0]
            return np.array([0.9 * v, 1.1 * v])
        diffs = np.diff(values)
        edges = np.empty(values.size + 1, dtype=float)
        edges[1:-1] = values[:-1] + diffs / 2.0
        edges[0] = values[0] - diffs[0] / 2.0
        edges[-1] = values[-1] + diffs[-1] / 2.0
        return edges

    energy_edges = _edge_array(np.maximum(energies, config.EPS))
    pitch_per_channel = np.nanmedian(pitches, axis=0)
    pitch_edges = _edge_array(pitch_per_channel)

    energy_edges = np.clip(energy_edges, config.EPS, None)

    for ax, title, data in zip(axes, titles, datasets, strict=False):
        pcm = ax.pcolormesh(
            energy_edges,
            pitch_edges,
            data.T,
            shading="auto",
        )
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("Energy [eV]")
        ax.set_ylabel("Pitch angle [deg]")
        fig.colorbar(pcm, ax=ax, shrink=0.8)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=200)
        print(f"Saved plot to {args.output}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
