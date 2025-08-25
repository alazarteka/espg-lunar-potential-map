import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import src.config as config
from src.flux import ERData, PitchAngle
from src.potential_mapper.pipeline import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot flux vs energy (x) and pitch angle (y) for a given spectrum. "
            "Energy is shown using bin widths of ±0.25·E around the reported centers."
        )
    )
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--day", type=int, required=True)
    parser.add_argument("--spec-no", type=int, required=True, help="Spectrum number")
    parser.add_argument("--output", type=str, default=None, help="Path to save PNG")
    parser.add_argument("-d", "--display", action="store_true", default=False)
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    return parser.parse_args()


def select_day_file(year: int, month: int, day: int) -> Path:
    files = DataLoader.discover_flux_files(year=year, month=month, day=day)
    if not files:
        raise FileNotFoundError("No ER file found for the requested date")
    if len(files) > 1:
        print(f"Warning: multiple files matched; using {files[0]}")
    return files[0]


"""
This script plots flux as scatter points in energy–pitch space for a single spectrum.
Each point corresponds to one channel measurement at a given energy row, colored by flux.
"""


def extract_spectrum(er: ERData, spec_no: int) -> Tuple[np.ndarray, np.ndarray]:
    rows = er.data[er.data[config.SPEC_NO_COLUMN] == spec_no]
    if rows.empty:
        raise ValueError(f"Spectrum {spec_no} not found in file {er.er_data_file}")
    # Expect SWEEP_ROWS rows
    energies = rows[config.ENERGY_COLUMN].to_numpy(dtype=np.float64)  # (R,)
    flux_mat = rows[config.FLUX_COLS].to_numpy(dtype=np.float64)  # (R, C)
    return energies, flux_mat


def main() -> None:
    args = parse_args()
    day_file = select_day_file(args.year, args.month, args.day)
    er = ERData(str(day_file))

    # Compute pitch angles for the whole file, then slice the spectrum rows
    pa = PitchAngle(er, str(config.DATA_DIR / config.THETA_FILE))

    energies, flux_mat = extract_spectrum(er, args.spec_no)
    # Find the row indices for this spectrum
    mask = er.data[config.SPEC_NO_COLUMN] == args.spec_no
    idxs = np.nonzero(mask.to_numpy())[0]
    if idxs.size == 0:
        raise RuntimeError("Unable to locate spectrum rows after selection.")

    # Slice pitch angles for these rows
    pitch_mat = pa.pitch_angles[idxs, :]  # (R, C), degrees

    # Establish a consistent channel ordering by sorting pitch of first row
    order = np.argsort(pitch_mat[0, :])
    pitch_sorted = pitch_mat[:, order]
    flux_sorted = flux_mat[:, order]

    # Prepare scatter coordinates: repeat energies per channel, flatten pitch/flux
    R, C = flux_sorted.shape
    x_pts = np.repeat(energies, C)
    y_pts = pitch_sorted.reshape(-1)
    f_pts = flux_sorted.reshape(-1)

    # Plot
    fig, ax = plt.subplots(figsize=(5, 8), constrained_layout=True)

    # Black background and high-contrast styling
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    for s in ax.spines.values():
        s.set_color("white")
    ax.tick_params(colors="white")

    # Robust log scaling: percentile-based vmin/vmax on positive flux
    pos = flux_sorted[flux_sorted > 0]
    if pos.size:
        vmin = float(max(1e-3, np.percentile(pos, 1)))
        vmax = float(np.percentile(pos, 99))
        if vmin >= vmax:
            vmin = float(max(1e-3, np.min(pos)))
            vmax = float(np.max(pos))
    else:
        vmin, vmax = 1e-3, 1.0

    sc = ax.scatter(
        x_pts,
        y_pts,
        c=f_pts,
        s=14,
        cmap="inferno",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        linewidths=0,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(
        "Flux [particles cm$^{-2}$ s$^{-1}$ sr$^{-1}$ eV$^{-1}$]", color="white"
    )
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    ax.set_xscale("log")
    ax.set_xlabel("Energy (eV)", color="white")
    ax.set_ylabel("Pitch angle (deg)", color="white")
    ax.set_title(
        f"Flux vs Energy/Pitch — {args.year:04d}-{args.month:02d}-{args.day:02d} spec {args.spec_no}",
        color="white",
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180)
        if args.verbose:
            print(f"Saved plot to {out_path}")
    if args.display:
        plt.show()


if __name__ == "__main__":
    main()
