#!/usr/bin/env python3
"""
Create Halekas-style loss cone fit visualization.
Shows observed normalized flux vs best-fit model with loss cone boundary.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import config
from src.flux import ERData, PitchAngle, LossConeFitter
from src.model import synth_losscone


def plot_loss_cone_comparison(er_file: Path, chunk_idx: int = 10, output_path: Path = None):
    """
    Create side-by-side plot of observed vs model loss cone.

    Args:
        er_file: Path to ER .TAB file
        chunk_idx: Which 15-row chunk to visualize
        output_path: Where to save plot (None = show instead)
    """
    # Load data
    print(f"Loading {er_file.name}...")
    er_data = ERData(str(er_file))
    pitch_angle = PitchAngle(er_data, str(config.DATA_DIR / config.THETA_FILE))

    # Create fitter
    fitter = LossConeFitter(er_data, str(config.DATA_DIR / config.THETA_FILE), pitch_angle)

    # Fit loss cone for all chunks (but catch failures)
    print("Fitting loss cones...")
    n_chunks = len(er_data.data) // config.SWEEP_ROWS
    results_list = []

    for i in range(n_chunks):
        try:
            U_surface, bs_bm, beam_amp, chi2 = fitter._fit_surface_potential(i)
            results_list.append([U_surface, bs_bm, beam_amp, chi2, i])
        except Exception as e:
            print(f"  Chunk {i} failed: {e}")
            continue

    if not results_list:
        print("No valid fits found!")
        return

    results = np.array(results_list)
    print(f"Successfully fit {len(results)}/{n_chunks} chunks")

    # Pick a chunk
    if chunk_idx >= len(results):
        chunk_idx = len(results) // 2
        print(f"Requested chunk {chunk_idx} not available, using {chunk_idx}")

    U_surface, bs_bm, beam_amp, chi2, chunk_idx_actual = results[chunk_idx]
    chunk_idx = int(chunk_idx_actual)  # Use the actual chunk index from results

    # Get the data for this chunk
    start_row = chunk_idx * config.SWEEP_ROWS
    end_row = min((chunk_idx + 1) * config.SWEEP_ROWS, len(er_data.data))

    chunk_data = er_data.data.iloc[start_row:end_row]
    flux_data = chunk_data[config.FLUX_COLS].to_numpy(dtype=np.float64)
    energies = chunk_data[config.ENERGY_COLUMN].to_numpy(dtype=np.float64)
    pitches = pitch_angle.pitch_angles[start_row:end_row]

    # Normalize by dividing by incident flux (pitch > 90°)
    # Average flux in incident hemisphere per energy
    incident_mask = pitches > 90
    incident_avg = np.nanmean(flux_data * incident_mask, axis=1, keepdims=True)
    incident_avg = np.where(incident_avg > 0, incident_avg, 1.0)  # avoid div by zero

    normalized_obs = flux_data / incident_avg

    # Create model
    beam_width = max(abs(U_surface) * config.LOSS_CONE_BEAM_WIDTH_FACTOR, 1.0)
    model = synth_losscone(
        energies,
        pitches,
        U_surface,
        U_spacecraft=0.0,
        bs_over_bm=bs_bm,
        beam_width_eV=beam_width,
        beam_amp=beam_amp,
        beam_pitch_sigma_deg=config.LOSS_CONE_BEAM_PITCH_SIGMA_DEG,
    )

    # Calculate loss cone boundary for plotting
    loss_cone_angle = []
    for E in energies:
        if E <= 0:
            loss_cone_angle.append(np.nan)
            continue
        x = bs_bm * (1.0 + U_surface / E)
        if x <= 0:
            ac = 0.0
        elif x >= 1:
            ac = 90.0
        else:
            ac = np.degrees(np.arcsin(np.sqrt(x)))
        # Store as 180 - ac (the cutoff pitch angle)
        loss_cone_angle.append(180 - ac)

    loss_cone_angle = np.array(loss_cone_angle)

    # Get timestamp for title
    timestamp = chunk_data.iloc[0][config.TIME_COLUMN]

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Common parameters
    vmin, vmax = 0.1, 1.0
    extent = [energies.min(), energies.max(), pitches.min(), pitches.max()]

    # Left: Observed
    im0 = axes[0].imshow(
        normalized_obs.T,
        aspect='auto',
        origin='lower',
        extent=extent,
        cmap='viridis',
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest'
    )
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Energy [eV]')
    axes[0].set_ylabel('Pitch Angle [deg]')
    axes[0].set_title(f'Observed (chunk {chunk_idx})\n{timestamp}')
    axes[0].plot(energies, loss_cone_angle, 'w-', linewidth=2, label='Loss Cone')
    plt.colorbar(im0, ax=axes[0], label='Normalized Flux')

    # Middle: Model
    im1 = axes[1].imshow(
        model.T,
        aspect='auto',
        origin='lower',
        extent=extent,
        cmap='viridis',
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest'
    )
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Energy [eV]')
    axes[1].set_ylabel('Pitch Angle [deg]')
    axes[1].set_title(f'Best Fit\nU_surface={U_surface:.1f}V, Bs/Bm={bs_bm:.2f}')
    axes[1].plot(energies, loss_cone_angle, 'w-', linewidth=2, label='Loss Cone')
    plt.colorbar(im1, ax=axes[1], label='Normalized Flux')

    # Right: Residual
    residual = normalized_obs - model
    vmax_res = max(abs(np.nanpercentile(residual, 5)), abs(np.nanpercentile(residual, 95)))
    im2 = axes[2].imshow(
        residual.T,
        aspect='auto',
        origin='lower',
        extent=extent,
        cmap='RdBu_r',
        vmin=-vmax_res,
        vmax=vmax_res,
        interpolation='nearest'
    )
    axes[2].set_xscale('log')
    axes[2].set_xlabel('Energy [eV]')
    axes[2].set_ylabel('Pitch Angle [deg]')
    axes[2].set_title(f'Residual (obs - model)\nχ²={chi2:.1e}')
    axes[2].plot(energies, loss_cone_angle, 'k-', linewidth=1, alpha=0.5)
    plt.colorbar(im2, ax=axes[2], label='Residual')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    else:
        plt.show()

    plt.close()

    # Print fit results
    print(f"\nFit Results for chunk {chunk_idx}:")
    print(f"  U_surface = {U_surface:.2f} V")
    print(f"  Bs/Bm = {bs_bm:.3f}")
    print(f"  Beam amplitude = {beam_amp:.2f}")
    print(f"  χ² = {chi2:.2e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot loss cone fit comparison")
    parser.add_argument("er_file", type=Path, help="Path to ER .TAB file")
    parser.add_argument("--chunk", type=int, default=10, help="Chunk index to plot")
    parser.add_argument("--output", type=Path, help="Output file path")

    args = parser.parse_args()

    plot_loss_cone_comparison(args.er_file, args.chunk, args.output)
