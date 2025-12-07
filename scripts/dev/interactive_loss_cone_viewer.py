#!/usr/bin/env python3
"""
Interactive loss cone viewer using Plotly.
Browse through different spec_no chunks and view raw (un-normalized) data.
"""
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import config
from src.flux import ERData, LossConeFitter, PitchAngle
from src.model import synth_losscone


def interpolate_to_regular_grid(energies, pitches, flux_data, n_pitch_bins=100):
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

    # Interpolate for each energy separately to maintain energy structure
    for i in range(len(energies)):
        # Get valid (non-NaN) data points for this energy
        valid_mask = np.isfinite(flux_data[i]) & np.isfinite(pitches[i])

        if np.sum(valid_mask) > 1:  # Need at least 2 points to interpolate
            pitch_pts = pitches[i, valid_mask]
            flux_pts = flux_data[i, valid_mask]

            # CRITICAL: Sort by pitch angle before interpolating
            # np.interp requires sorted x-values
            sort_idx = np.argsort(pitch_pts)
            pitch_pts_sorted = pitch_pts[sort_idx]
            flux_pts_sorted = flux_pts[sort_idx]

            # Interpolate onto regular pitch grid
            flux_reg[i] = np.interp(pitches_reg, pitch_pts_sorted, flux_pts_sorted,
                                    left=np.nan, right=np.nan)
        else:
            # Not enough data, fill with NaN
            flux_reg[i] = np.nan

    return energies, pitches_reg, flux_reg


def create_interactive_viewer(er_file: Path, output_path: Path = None):
    """
    Create interactive Plotly HTML viewer for loss cone data.

    Args:
        er_file: Path to ER .TAB file
        output_path: Where to save HTML (default: scratch/loss_cone_viewer.html)
    """
    if output_path is None:
        output_path = Path("scratch/loss_cone_viewer.html")

    # Load data
    print(f"Loading {er_file.name}...")
    er_data = ERData(str(er_file))
    pitch_angle = PitchAngle(er_data, str(config.DATA_DIR / config.THETA_FILE))

    # Create fitter
    fitter = LossConeFitter(er_data, str(config.DATA_DIR / config.THETA_FILE), pitch_angle)

    # Fit loss cone for all chunks
    print("Fitting loss cones...")
    n_chunks = len(er_data.data) // config.SWEEP_ROWS
    results_list = []

    for i in range(n_chunks):
        try:
            U_surface, bs_bm, beam_amp, chi2 = fitter._fit_surface_potential(i)
            results_list.append([U_surface, bs_bm, beam_amp, chi2, i])
        except Exception:
            # Skip failed fits
            continue

    if not results_list:
        print("No valid fits found!")
        return

    results = np.array(results_list)
    print(f"Successfully fit {len(results)}/{n_chunks} chunks")

    # Create frames for each chunk
    frames = []

    for idx, (U_surface, bs_bm, beam_amp, chi2, chunk_idx) in enumerate(results):
        chunk_idx = int(chunk_idx)

        # Get data for this chunk
        start_row = chunk_idx * config.SWEEP_ROWS
        end_row = min((chunk_idx + 1) * config.SWEEP_ROWS, len(er_data.data))

        chunk_data = er_data.data.iloc[start_row:end_row]
        flux_data = chunk_data[config.FLUX_COLS].to_numpy(dtype=np.float64)
        energies = chunk_data[config.ENERGY_COLUMN].to_numpy(dtype=np.float64)
        pitches = pitch_angle.pitch_angles[start_row:end_row]
        timestamp = chunk_data.iloc[0][config.TIME_COLUMN]

        # Replace zeros and negatives with NaN for better visualization
        flux_data = np.where(flux_data > 0, flux_data, np.nan)

        # Convert to log10 flux
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
            U_spacecraft=0.0,
            bs_over_bm=bs_bm,
            beam_width_eV=beam_width,
            beam_amp=beam_amp,
            beam_pitch_sigma_deg=config.LOSS_CONE_BEAM_PITCH_SIGMA_DEG,
            background=fitter.background,
        )

        # Interpolate model onto regular grid
        _, _, model_reg = interpolate_to_regular_grid(
            energies, pitches, model_irregular
        )

        # Calculate loss cone boundary
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
            loss_cone_angle.append(180 - ac)

        loss_cone_angle = np.array(loss_cone_angle)

        # Create frame with regular grid data
        frame_name = f"chunk_{chunk_idx}"
        frames.append({
            'name': frame_name,
            'data': log_flux_reg,  # Regular grid, log10 flux
            'model': model_reg,  # Regular grid
            'energies': energies_reg,
            'pitches': pitches_reg,  # Now 1D regular grid
            'loss_cone': loss_cone_angle,
            'U_surface': U_surface,
            'bs_bm': bs_bm,
            'beam_amp': beam_amp,
            'chi2': chi2,
            'timestamp': timestamp,
            'chunk_idx': chunk_idx,
        })

    # Create initial plot (first chunk)
    first_frame = frames[0]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Observed Flux', 'Model Flux'),
        horizontal_spacing=0.12
    )

    # Observed data
    fig.add_trace(
        go.Heatmap(
            z=first_frame['data'].T,
            x=first_frame['energies'],
            y=first_frame['pitches'],  # Now 1D regular grid
            colorscale='Viridis',
            name='Observed',
            colorbar=dict(x=0.45, len=0.9, title='log₁₀(Flux)<br>[#/cm²/s/sr/eV]'),
        ),
        row=1, col=1
    )

    # Loss cone boundary on observed
    fig.add_trace(
        go.Scatter(
            x=first_frame['energies'],
            y=first_frame['loss_cone'],
            mode='lines',
            line=dict(color='white', width=2),
            name='Loss Cone',
            showlegend=False,
        ),
        row=1, col=1
    )

    # Model
    fig.add_trace(
        go.Heatmap(
            z=first_frame['model'].T,
            x=first_frame['energies'],
            y=first_frame['pitches'],  # Now 1D regular grid
            colorscale='Viridis',
            name='Model',
            colorbar=dict(x=1.02, len=0.9, title='Normalized'),
            zmin=0,
            zmax=1,
        ),
        row=1, col=2
    )

    # Loss cone boundary on model
    fig.add_trace(
        go.Scatter(
            x=first_frame['energies'],
            y=first_frame['loss_cone'],
            mode='lines',
            line=dict(color='white', width=2),
            name='Loss Cone',
            showlegend=False,
        ),
        row=1, col=2
    )

    # Update layout
    fig.update_xaxes(type='log', title_text='Energy [eV]', row=1, col=1)
    fig.update_xaxes(type='log', title_text='Energy [eV]', row=1, col=2)
    fig.update_yaxes(title_text='Pitch Angle [deg]', row=1, col=1)
    fig.update_yaxes(title_text='Pitch Angle [deg]', row=1, col=2)

    fig.update_layout(
        title=dict(
            text=f"Chunk {first_frame['chunk_idx']}: {first_frame['timestamp']}<br>"
                 f"U_surface = {first_frame['U_surface']:.1f} V, Bs/Bm = {first_frame['bs_bm']:.2f}, "
                 f"Beam Amp = {first_frame['beam_amp']:.1f}, χ² = {first_frame['chi2']:.1e}",
            x=0.5,
            xanchor='center',
        ),
        height=600,
        showlegend=False,
    )

    # Create slider steps
    steps = []
    for idx, frame in enumerate(frames):
        step = dict(
            method="update",
            args=[
                {
                    "z": [
                        frame['data'].T,
                        None,  # scatter trace
                        frame['model'].T,
                        None,  # scatter trace
                    ],
                    "x": [
                        frame['energies'],
                        frame['energies'],
                        frame['energies'],
                        frame['energies'],
                    ],
                    "y": [
                        frame['pitches'],  # Now 1D regular grid
                        frame['loss_cone'],
                        frame['pitches'],  # Now 1D regular grid
                        frame['loss_cone'],
                    ],
                },
                {
                    "title": dict(
                        text=f"Chunk {frame['chunk_idx']}: {frame['timestamp']}<br>"
                             f"U_surface = {frame['U_surface']:.1f} V, Bs/Bm = {frame['bs_bm']:.2f}, "
                             f"Beam Amp = {frame['beam_amp']:.1f}, χ² = {frame['chi2']:.1e}",
                        x=0.5,
                        xanchor='center',
                    ),
                },
            ],
            label=f"{frame['chunk_idx']}",
        )
        steps.append(step)

    sliders = [dict(
        active=0,
        yanchor="top",
        y=-0.15,
        xanchor="left",
        currentvalue=dict(
            prefix="Chunk: ",
            visible=True,
            xanchor="right"
        ),
        pad=dict(b=10, t=50),
        len=0.9,
        x=0.05,
        steps=steps
    )]

    fig.update_layout(sliders=sliders)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"\nSaved interactive viewer to: {output_path}")
    print(f"Open in browser to navigate through {len(frames)} chunks")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create interactive loss cone viewer")
    parser.add_argument("er_file", type=Path, help="Path to ER .TAB file")
    parser.add_argument("--output", type=Path, help="Output HTML file path")

    args = parser.parse_args()

    create_interactive_viewer(args.er_file, args.output)
