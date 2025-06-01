#!/usr/bin/env python3
"""
Loss Cone Visualizer

Plots energy vs pitch angle vs normalized flux for a specific sweep (spec_no).
This helps visualize the loss cone structure in the data.

Usage:
    python loss_cone_visualizer.py <flux_file> <theta_file> <spec_no>

Example:
    python loss_cone_visualizer.py data/1998/091_120APR/3D980429.TAB data/theta.tab 657
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.interpolate import griddata

# Import our modules
from flux import ERData, PitchAngle
import config


def load_sweep_data(flux_file: str, theta_file: str, spec_no: int):
    """Load and process data for a specific sweep."""

    # Load ER data
    er_data = ERData(flux_file)
    if er_data.data is None:
        raise ValueError(f"Could not load flux data from {flux_file}")

    # Filter for the specific spec_no
    sweep_data = er_data.data[er_data.data['spec_no'] == spec_no].copy()
    if len(sweep_data) == 0:
        raise ValueError(f"No data found for spec_no {spec_no}")

    if len(sweep_data) != config.SWEEP_ROWS:
        print(f"Warning: Expected {config.SWEEP_ROWS} rows, got {len(sweep_data)}")

    # Create a temporary ERData object with just this sweep
    temp_er_data = ERData.__new__(ERData)
    temp_er_data.data = sweep_data.reset_index(drop=True)
    temp_er_data.er_data_file = flux_file

    # Calculate pitch angles
    pitch_angle = PitchAngle(temp_er_data, theta_file)

    return temp_er_data, pitch_angle


def calculate_normalized_flux(er_data, pitch_angles):
    """Calculate incident-normalized flux exactly like LossConeFitter does."""

    # Create a temporary LossConeFitter to use its exact normalization method
    from flux import LossConeFitter

    temp_fitter = LossConeFitter.__new__(LossConeFitter)
    temp_fitter.er_data = er_data
    temp_fitter.pitch_angle = pitch_angles
    temp_fitter.thetas = pitch_angles.thetas

    # Build the 2D normalized flux array (this is what the fitter sees)
    norm2d = temp_fitter.build_norm2d(0)  # measurement_chunk = 0 since we only have one sweep

    energies = er_data.data['energy'].to_numpy(dtype=np.float64)

    return norm2d, energies


def create_loss_cone_plot(energies, pitch_angles, normalized_flux, spec_no, flux_file):
    """Create the loss cone visualization plot with both scatter and interpolated views."""

    # Create subplots - scatter and interpolated
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Get pitch angle data
    pitch_grid = pitch_angles.pitch_angles  # Shape: (n_energy, n_channels)

    # Collect data points
    energy_points = []
    pitch_points = []
    flux_points = []

    for i, energy in enumerate(energies):
        if i < len(pitch_grid) and i < len(normalized_flux):
            angles = pitch_grid[i]
            fluxes = normalized_flux[i]

            # Only collect valid data points
            valid_mask = ~np.isnan(angles) & ~np.isnan(fluxes) & (fluxes > 0)

            if valid_mask.any():
                energy_points.extend([energy] * valid_mask.sum())
                pitch_points.extend(angles[valid_mask])
                flux_points.extend(fluxes[valid_mask])

    if len(flux_points) == 0:
        print("Warning: No valid data points to plot")
        return fig

    energy_points = np.array(energy_points)
    pitch_points = np.array(pitch_points)
    flux_points = np.array(flux_points)

    # Plot 1: Scatter plot (original view)
    scatter = ax1.scatter(energy_points, pitch_points, c=np.log10(flux_points + config.EPS),
                         cmap='inferno', s=2, alpha=0.7)

    ax1.set_xlabel('Energy (eV)')
    ax1.set_ylabel('Pitch Angle (degrees)')
    ax1.set_title(f'Raw Data Points\nSpec No: {spec_no}')
    ax1.axhline(y=90, color='red', linestyle='--', alpha=0.7, linewidth=2, label='90° boundary')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 180)
    ax1.set_xlim(energies.min() * 0.9, energies.max() * 1.1)

    # Plot 2: Interpolated heatmap
    from scipy.interpolate import griddata

    # Create regular grid for interpolation
    energy_grid = np.linspace(energies.min(), energies.max(), 50)
    pitch_grid_reg = np.linspace(0, 180, 100)
    energy_mesh, pitch_mesh = np.meshgrid(energy_grid, pitch_grid_reg)

    # Interpolate flux values onto regular grid
    flux_interp = griddata(
        (energy_points, pitch_points), 
        np.log10(flux_points + config.EPS),
        (energy_mesh, pitch_mesh), 
        method='linear',
        fill_value=np.nan
    )

    # Create heatmap
    im = ax2.imshow(flux_interp, 
                    aspect='auto', 
                    origin='lower',
                    cmap='inferno',
                    extent=[energies.min(), energies.max(), 0, 180])

    ax2.set_xlabel('Energy (eV)')
    ax2.set_ylabel('Pitch Angle (degrees)')
    ax2.set_title(f'Interpolated Heatmap\nSpec No: {spec_no}')
    ax2.axhline(y=90, color='red', linestyle='--', alpha=0.7, linewidth=2, label='90° boundary')
    ax2.legend()

    # Color bars
    plt.colorbar(scatter, ax=ax1, label='Log10(Normalized Flux)', shrink=0.8)
    plt.colorbar(im, ax=ax2, label='Log10(Normalized Flux)', shrink=0.8)

    plt.suptitle(f'Loss Cone Visualization - {Path(flux_file).name}')
    plt.tight_layout()

    return fig


def main():
    """Main function."""

    if len(sys.argv) != 4:
        print("Usage: python loss_cone_visualizer.py <flux_file> <theta_file> <spec_no>")
        print("Example: python loss_cone_visualizer.py data/1998/091_120APR/3D980429.TAB data/theta.tab 657")
        sys.exit(1)

    flux_file = sys.argv[1]
    theta_file = sys.argv[2]

    try:
        spec_no = int(sys.argv[3])
    except ValueError:
        print(f"Error: spec_no must be an integer, got '{sys.argv[3]}'")
        sys.exit(1)

    # Validate files exist
    if not Path(flux_file).exists():
        print(f"Error: Flux file {flux_file} not found")
        sys.exit(1)

    if not Path(theta_file).exists():
        print(f"Error: Theta file {theta_file} not found")
        sys.exit(1)

    try:
        print(f"Loading data for spec_no {spec_no} from {flux_file}...")
        er_data, pitch_angle = load_sweep_data(flux_file, theta_file, spec_no)

        print("Calculating normalized flux...")
        normalized_flux, energies = calculate_normalized_flux(er_data, pitch_angle)

        print("Creating visualization...")
        fig = create_loss_cone_plot(energies, pitch_angle, normalized_flux, spec_no, flux_file)

        # Show additional info
        print(f"\nSweep Information:")
        print(f"  Spec No: {spec_no}")
        print(f"  Energy range: {energies.min():.1f} - {energies.max():.1f} eV")
        print(f"  Number of energy steps: {len(energies)}")
        print(f"  Pitch angle range: {np.nanmin(pitch_angle.pitch_angles):.1f} - {np.nanmax(pitch_angle.pitch_angles):.1f} degrees")

        # Show flux statistics
        valid_flux = normalized_flux[~np.isnan(normalized_flux)]
        if len(valid_flux) > 0:
            print(f"  Flux range: {valid_flux.min():.2e} - {valid_flux.max():.2e}")

        # Show what the fitting algorithm would see
        print(f"\nFitting Algorithm View:")
        print(f"  Data shape: {normalized_flux.shape}")
        print(f"  Valid data points: {(~np.isnan(normalized_flux)).sum()}")
        print(f"  NaN data points: {np.isnan(normalized_flux).sum()}")

        plt.show()

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
