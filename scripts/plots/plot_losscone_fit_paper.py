#!/usr/bin/env python3
"""
Create matplotlib figure showing measured vs model loss-cone fit for paper.

Side-by-side heatmaps with consistent visual style matching plot_daily_measurements.py.

Example:
    uv run python scripts/plots/plot_losscone_fit_paper.py \\
        --input data/1998/091_120APR/3D980415.TAB \\
        --spec-no 50 \\
        --output plots/publish/losscone_fit_comparison.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.flux import ERData, LossConeFitter, PitchAngle
from src.model import synth_losscone

try:
    from src.model_torch import LossConeFitterTorch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
from src.visualization import style, utils


def interpolate_to_regular_grid(
    energies: np.ndarray,
    pitches: np.ndarray,
    flux_data: np.ndarray,
    n_pitch_bins: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate irregular 2D (energy, pitch) grid onto regular grid.

    Args:
        energies: 1D array of energy values (n_energies,)
        pitches: 2D array of pitch angles (n_energies, n_channels)
        flux_data: 2D array of flux values (n_energies, n_channels)
        n_pitch_bins: Number of regular pitch angle bins

    Returns:
        energies_reg: Regular energy grid (same as input)
        pitches_reg: Regular pitch angle grid (n_pitch_bins,)
        flux_reg: Interpolated flux on regular grid (n_energies, n_pitch_bins)
    """
    # Create regular pitch grid
    pitch_min = np.nanmin(pitches)
    pitch_max = np.nanmax(pitches)
    pitches_reg = np.linspace(pitch_min, pitch_max, n_pitch_bins)

    # Initialize regular grid
    flux_reg = np.zeros((len(energies), n_pitch_bins))

    # Interpolate for each energy separately
    for i in range(len(energies)):
        valid_mask = np.isfinite(flux_data[i]) & np.isfinite(pitches[i])

        if np.sum(valid_mask) > 1:
            pitch_pts = pitches[i, valid_mask]
            flux_pts = flux_data[i, valid_mask]

            # Sort by pitch angle
            sort_idx = np.argsort(pitch_pts)
            pitch_pts_sorted = pitch_pts[sort_idx]
            flux_pts_sorted = flux_pts[sort_idx]

            # Interpolate onto regular pitch grid
            flux_reg[i] = np.interp(
                pitches_reg,
                pitch_pts_sorted,
                flux_pts_sorted,
                left=np.nan,
                right=np.nan,
            )
        else:
            flux_reg[i] = np.nan

    return energies, pitches_reg, flux_reg


def create_losscone_comparison_plot(
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
    title: str | None = None,
    use_torch: bool = False,
) -> None:
    """
    Create side-by-side comparison of measured vs model loss-cone.

    Args:
        er_file: Path to ER .TAB file
        spec_no: Spectrum number to plot
        output_path: Where to save the figure
        theta_file: Path to theta table for pitch angle calculations
        usc: Spacecraft potential [V] applied to all rows
        normalization: Loss cone normalization mode ("global" or "ratio")
        fixed_beam_amp: If provided, fix Gaussian beam amplitude to this value
        background: Model background outside the loss cone
        incident_flux_stat: Statistic for incident flux ("mean" or "max")
        dpi: Resolution for output
        title: Optional override for the figure suptitle
    """
    print(f"Loading {er_file.name}...")
    er_data = ERData(str(er_file))
    pitch_angle = PitchAngle(er_data)

    # Create spacecraft potential array (constant USC per Halekas et al.)
    spacecraft_potential = np.full(len(er_data.data), usc)

    # Create fitter with spacecraft potential correction
    if use_torch and HAS_TORCH:
        print("Using PyTorch-accelerated fitter (~5x faster)")
        fitter = LossConeFitterTorch(
            er_data,
            str(theta_file),
            pitch_angle,
            spacecraft_potential,
            normalization_mode=normalization,
            beam_amp_fixed=fixed_beam_amp,
            loss_cone_background=background,
            incident_flux_stat=incident_flux_stat,
            device="cpu",
        )
    else:
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

    # Validate and find the spectrum
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
    if use_torch and HAS_TORCH:
        U_surface, bs_bm, beam_amp, chi2 = fitter._fit_surface_potential_torch(spec_no)
    else:
        U_surface, bs_bm, beam_amp, chi2 = fitter._fit_surface_potential(spec_no)
    # U_surface, bs_bm, beam_amp, chi2 = -160.0, 0.975, 0.5, 1.23  # Example fixed values for paper plot

    # chunk_data already retrieved above
    flux_data = chunk_data[config.FLUX_COLS].to_numpy(dtype=np.float64)
    energies = chunk_data[config.ENERGY_COLUMN].to_numpy(dtype=np.float64)

    # Get the row indices for pitch angle lookup
    chunk_indices = chunk_data.index.to_numpy()
    pitches = pitch_angle.pitch_angles[chunk_indices]

    timestamp = chunk_data.iloc[0][config.TIME_COLUMN]
    print(f"Spectrum timestamp: {timestamp}")
    print(f"Spectrum number: {chunk_data.iloc[0][config.SPEC_NO_COLUMN]}")

    # Replace zeros and negatives with NaN
    flux_data = np.where(flux_data > 0, flux_data, np.nan)

    # Convert to log10 flux for observed data
    log_flux_data = np.log10(flux_data)

    # Interpolate observed data onto regular grid
    energies_reg, pitches_reg, log_flux_reg = interpolate_to_regular_grid(
        energies, pitches, log_flux_data
    )

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

    # Create figure with side-by-side heatmaps
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 5), constrained_layout=True, dpi=dpi
    )

    # Plot observed data
    im1 = ax1.pcolormesh(
        energies_reg,
        pitches_reg,
        log_flux_reg.T,
        cmap="viridis",
        shading="auto",
    )
    ax1.plot(energies, loss_cone_angle, "w-", linewidth=2, label="Loss Cone")
    ax1.set_xlabel("Energy (eV)", fontsize=style.FONT_SIZE_LABEL)
    ax1.set_ylabel("Pitch Angle (°)", fontsize=style.FONT_SIZE_LABEL)
    ax1.set_title("Observed Electron Flux", fontsize=style.FONT_SIZE_TITLE)
    ax1.set_xscale("log")
    ax1.set_ylim(0, 180)
    style.apply_paper_style(ax1)

    # Plot model
    im2 = ax2.pcolormesh(
        energies_reg,
        pitches_reg,
        model_reg.T,
        cmap="viridis",
        shading="auto",
        vmin=0,
        vmax=1,
    )
    ax2.plot(energies, loss_cone_angle, "w-", linewidth=2, label="Loss Cone")
    ax2.set_xlabel("Energy (eV)", fontsize=style.FONT_SIZE_LABEL)
    ax2.set_ylabel("Pitch Angle (°)", fontsize=style.FONT_SIZE_LABEL)
    ax2.set_title("Model Fit", fontsize=style.FONT_SIZE_TITLE)
    ax2.set_xscale("log")
    ax2.set_ylim(0, 180)
    style.apply_paper_style(ax2)

    # Add colorbars
    cbar1 = fig.colorbar(im1, ax=ax1, label="log₁₀(Flux) [#/cm²/s/sr/eV]")
    cbar1.ax.tick_params(labelsize=style.FONT_SIZE_TEXT)
    cbar2 = fig.colorbar(im2, ax=ax2, label="Normalized Flux")
    cbar2.ax.tick_params(labelsize=style.FONT_SIZE_TEXT)

    # Add text box with fit parameters
    textstr = (
        f"UTC: {timestamp}\nU_surface = {U_surface:.1f} V\n"
        f"Bₛ/Bₘ = {bs_bm:.3f}\nBeam amp = {beam_amp:.3f}\n"
        f"USC = {usc:.1f} V ({normalization})\n"
        f"bg = {fitter.background:.3f}, inc={incident_flux_stat}\n"
        f"χ² = {chi2:.2f}"
    )
    utils.add_stats_box(ax2, textstr, loc="lower right")

    # Add figure title if provided
    if title:
        fig.suptitle(title, fontsize=style.FONT_SIZE_TITLE + 1, y=1.02)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved to {output_path}")
    print(
        f"Fit parameters: U_surface={U_surface:.1f}V, Bₛ/Bₘ={bs_bm:.3f}, beam={beam_amp:.3f}, χ²={chi2:.2f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create loss-cone comparison plot for paper",
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
        default=-11.0,
        help="Spacecraft potential [V] applied to all rows",
    )
    parser.add_argument(
        "--normalization",
        choices=["global", "ratio", "ratio2", "ratio_rescaled"],
        default="global",
        help="Normalization mode: global (max incident), ratio (per-energy), ratio2 (pairwise), ratio_rescaled (ratio then rescale to [0,1])",
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
    parser.add_argument(
        "--paper-mode",
        action="store_true",
        help="Use Halekas Figure 5 settings (USC=+11V, ratio normalization, fixed beam amp=1)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Override figure title",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use PyTorch-accelerated fitter (~5x faster)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.paper_mode:
        # Halekas Figure 5: USC=+11 V, ratio normalization, fixed beam amplitude
        args.usc = 11.0
        args.normalization = "ratio"
        if args.fixed_beam_amp is None:
            args.fixed_beam_amp = 1.0

    if not args.input.exists():
        print(f"Error: Input file {args.input} not found")
        return 1

    if not args.theta_file.exists():
        print(f"Error: Theta file {args.theta_file} not found")
        return 1

    create_losscone_comparison_plot(
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
        title=args.title,
        use_torch=args.fast,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
