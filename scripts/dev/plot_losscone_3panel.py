#!/usr/bin/env python3
"""
Three-panel debug plot: raw data, normalized data, and model fit.

Shows the normalization transformation and how the model compares to normalized data.

Example:
    uv run python scripts/dev/plot_losscone_3panel.py \\
        --input data/1999/091_120APR/3D990429.TAB \\
        --spec-no 653 \\
        --output artifacts/debug_3panel.png \\
        --usc 11.0 \\
        --normalization global
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import from scripts/plots
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np

# Import interpolation function from the original script
from plots.plot_losscone_fit_paper import interpolate_to_regular_grid

from src import config
from src.flux import ERData, LossConeFitter, PitchAngle
from src.model import synth_losscone


def create_3panel_debug_plot(
    er_file: Path,
    spec_no: int,
    output_path: Path,
    theta_file: Path,
    usc: float,
    normalization: str,
    fixed_beam_amp: float | None,
    background: float,
    incident_flux_stat: str,
    dpi: int = 150,
) -> None:
    """
    Create three-panel debug plot: raw data, normalized data, model fit.

    Args:
        er_file: Path to ER .TAB file
        spec_no: Spectrum number to plot
        output_path: Where to save the figure
        theta_file: Path to theta table for pitch angle calculations
        usc: Spacecraft potential [V] applied to all rows
        normalization: Normalization mode ("global", "ratio", or "ratio2")
        fixed_beam_amp: If provided, fix Gaussian beam amplitude to this value
        background: Model background outside the loss cone
        incident_flux_stat: Statistic for incident flux ("mean" or "max")
        dpi: Resolution for output
    """
    print(f"Loading {er_file.name}...")
    er_data = ERData(str(er_file))
    pitch_angle = PitchAngle(er_data)

    # Create spacecraft potential array
    spacecraft_potential = np.full(len(er_data.data), usc)

    # Create fitter with spacecraft potential correction
    fitter = LossConeFitter(
        er_data,
        str(theta_file),
        pitch_angle,
        spacecraft_potential,
        normalization_mode=normalization,
        beam_amp_fixed=fixed_beam_amp,
        loss_cone_background=background,
        incident_flux_stat=incident_flux_stat,
    )

    # Validate spectrum number
    min_spec_no = er_data.data.iloc[0][config.SPEC_NO_COLUMN]
    max_spec_no = er_data.data.iloc[-1][config.SPEC_NO_COLUMN]
    print(f"Available spectrum numbers: {min_spec_no} to {max_spec_no}")

    if spec_no < min_spec_no or spec_no > max_spec_no:
        raise ValueError(
            f"Spectrum number {spec_no} out of range [{min_spec_no}, {max_spec_no}]"
        )

    # Find the rows corresponding to this spectrum number
    chunk_mask = er_data.data[config.SPEC_NO_COLUMN] == spec_no
    if not chunk_mask.any():
        raise ValueError(f"Spectrum number {spec_no} not found in data")

    chunk_data = er_data.data[chunk_mask]

    if len(chunk_data) != config.SWEEP_ROWS:
        print(
            f"Warning: Expected {config.SWEEP_ROWS} rows for spectrum {spec_no}, got {len(chunk_data)}"
        )

    # Fit the specified spectrum
    print(f"Fitting spectrum {spec_no}...")
    U_surface, bs_bm, beam_amp, chi2 = fitter._fit_surface_potential(spec_no)

    # Get raw flux data
    flux_data = chunk_data[config.FLUX_COLS].to_numpy(dtype=np.float64)
    energies = chunk_data[config.ENERGY_COLUMN].to_numpy(dtype=np.float64)

    # Get pitch angles
    chunk_indices = chunk_data.index.to_numpy()
    pitches = pitch_angle.pitch_angles[chunk_indices]

    timestamp = chunk_data.iloc[0][config.TIME_COLUMN]
    print(f"Spectrum timestamp: {timestamp}")
    print(f"Spectrum number: {chunk_data.iloc[0][config.SPEC_NO_COLUMN]}")

    # Replace zeros and negatives with NaN for raw data
    flux_data_clean = np.where(flux_data > 0, flux_data, np.nan)

    # Get normalized data (this applies the normalization mode)
    norm2d = fitter.build_norm2d(spec_no)

    # Convert to log10 flux for raw data display
    log_flux_data = np.log10(flux_data_clean)

    # Interpolate all three datasets onto regular grid
    energies_reg, pitches_reg, log_flux_reg = interpolate_to_regular_grid(
        energies, pitches, log_flux_data
    )
    _, _, norm_reg = interpolate_to_regular_grid(energies, pitches, norm2d)

    # Create model on irregular grid first
    beam_width = max(abs(U_surface) * config.LOSS_CONE_BEAM_WIDTH_FACTOR, 1.0)
    model_irregular = synth_losscone(
        energies,
        pitches,
        U_surface,
        U_spacecraft=usc,
        bs_over_bm=bs_bm,
        beam_width_eV=beam_width,
        beam_amp=beam_amp,
        beam_pitch_sigma_deg=config.LOSS_CONE_BEAM_PITCH_SIGMA_DEG,
        background=fitter.background,
    )

    # Interpolate model onto regular grid
    _, _, model_reg = interpolate_to_regular_grid(energies, pitches, model_irregular)

    # Calculate loss cone boundary (with spacecraft potential correction)
    loss_cone_angle = []
    for E in energies:
        if E <= 0:
            loss_cone_angle.append(np.nan)
            continue
        # Apply spacecraft potential correction: E_plasma = E_measured - U_spacecraft
        E_corrected = E - usc
        if E_corrected <= 0:
            loss_cone_angle.append(np.nan)
            continue
        x = bs_bm * (1.0 + U_surface / E_corrected)
        if x <= 0:
            ac = 0.0
        elif x >= 1:
            ac = 90.0
        else:
            ac = np.degrees(np.arcsin(np.sqrt(x)))
        loss_cone_angle.append(180 - ac)

    loss_cone_angle = np.array(loss_cone_angle)

    # Create figure with three side-by-side heatmaps
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(18, 5), constrained_layout=True, dpi=dpi
    )

    # Panel 1: Raw observed data (log scale)
    im1 = ax1.pcolormesh(
        energies_reg,
        pitches_reg,
        log_flux_reg.T,
        cmap="viridis",
        shading="auto",
    )
    ax1.plot(energies, loss_cone_angle, "w-", linewidth=2, label="Loss Cone")
    ax1.set_xlabel("Energy (eV)")
    ax1.set_ylabel("Pitch Angle (°)")
    ax1.set_title("Raw Electron Flux")
    ax1.set_xscale("log")
    ax1.set_ylim(0, 180)
    ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.4)
    cbar1 = fig.colorbar(im1, ax=ax1, label="log₁₀(Flux) [#/cm²/s/sr/eV]")

    # Panel 2: Normalized data
    # Determine color scale based on normalization mode
    if normalization == "ratio2":
        # ratio2: incident=1, reflected=ratio (can be > 1)
        vmin2, vmax2 = 0, np.nanmax([2.0, np.nanpercentile(norm_reg, 99)])
    elif normalization in ("global", "ratio_rescaled"):
        # global and ratio_rescaled: scaled to [0, 1]
        vmin2, vmax2 = 0, 1
    else:  # ratio
        # ratio: per-energy, typically near 1 but can exceed
        vmin2, vmax2 = 0, np.nanmax([2.0, np.nanpercentile(norm_reg, 99)])

    im2 = ax2.pcolormesh(
        energies_reg,
        pitches_reg,
        norm_reg.T,
        cmap="viridis",
        shading="auto",
        vmin=vmin2,
        vmax=vmax2,
    )
    ax2.plot(energies, loss_cone_angle, "w-", linewidth=2, label="Loss Cone")
    ax2.set_xlabel("Energy (eV)")
    ax2.set_ylabel("Pitch Angle (°)")
    ax2.set_title(f"Normalized Flux ({normalization})")
    ax2.set_xscale("log")
    ax2.set_ylim(0, 180)
    ax2.grid(True, alpha=0.3, linestyle="--", linewidth=0.4)
    cbar2 = fig.colorbar(im2, ax=ax2, label="Normalized Flux")

    # Panel 3: Model
    im3 = ax3.pcolormesh(
        energies_reg,
        pitches_reg,
        model_reg.T,
        cmap="viridis",
        shading="auto",
        vmin=0,
        vmax=1,
    )
    ax3.plot(energies, loss_cone_angle, "w-", linewidth=2, label="Loss Cone")
    ax3.set_xlabel("Energy (eV)")
    ax3.set_ylabel("Pitch Angle (°)")
    ax3.set_title("Model Fit")
    ax3.set_xscale("log")
    ax3.set_ylim(0, 180)
    ax3.grid(True, alpha=0.3, linestyle="--", linewidth=0.4)
    cbar3 = fig.colorbar(im3, ax=ax3, label="Model Flux")

    # Add text box with fit parameters
    textstr = (
        f"UTC: {timestamp}\n"
        f"U_surface = {U_surface:.1f} V\n"
        f"Bₛ/Bₘ = {bs_bm:.3f}\n"
        f"Beam amp = {beam_amp:.3f}\n"
        f"USC = {usc:.1f} V\n"
        f"bg = {fitter.background:.3f}, inc={incident_flux_stat}\n"
        f"norm = {normalization}\n"
        f"χ² = {chi2:.2f}"
    )
    ax3.text(
        0.98,
        0.02,
        textstr,
        transform=ax3.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved to {output_path}")
    print(
        f"Fit parameters: U_surface={U_surface:.1f}V, Bₛ/Bₘ={bs_bm:.3f}, beam={beam_amp:.3f}, χ²={chi2:.2f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create 3-panel debug plot (raw, normalized, model)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="ER .TAB file to process",
    )
    parser.add_argument(
        "--spec-no",
        type=int,
        required=True,
        help="Spectrum number to plot (from SPEC_NO column in data)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for PNG file",
    )
    parser.add_argument(
        "--theta-file",
        type=Path,
        default=config.DATA_DIR / config.THETA_FILE,
        help="Theta table for pitch angle calculations",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output resolution",
    )
    parser.add_argument(
        "--usc",
        type=float,
        default=11.0,
        help="Spacecraft potential [V] applied to all rows",
    )
    parser.add_argument(
        "--normalization",
        choices=["global", "ratio", "ratio2", "ratio_rescaled"],
        default="global",
        help="Normalization mode",
    )
    parser.add_argument(
        "--background",
        type=float,
        default=config.LOSS_CONE_BACKGROUND,
        help="Model background level outside the loss cone",
    )
    parser.add_argument(
        "--incident-flux-stat",
        choices=["mean", "max"],
        default="mean",
        help="Statistic to use for incident flux normalization",
    )
    parser.add_argument(
        "--fixed-beam-amp",
        type=float,
        default=None,
        help="Fix beam amplitude instead of fitting it",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        print(f"Error: Input file {args.input} not found")
        return 1

    if not args.theta_file.exists():
        print(f"Error: Theta file {args.theta_file} not found")
        return 1

    create_3panel_debug_plot(
        args.input,
        args.spec_no,
        args.output,
        args.theta_file,
        args.usc,
        args.normalization,
        args.fixed_beam_amp,
        args.background,
        args.incident_flux_stat,
        args.dpi,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
