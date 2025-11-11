"""
Interactive Plotly visualization of time-dependent lunar surface potential.

Creates a 3D sphere with temporal animation/slider showing smooth interpolation
of the spherical harmonic coefficients over time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import interp1d

from src.temporal import load_temporal_coefficients, reconstruct_global_map


def latlon_to_xyz(lat: np.ndarray, lon: np.ndarray, radius: float = 1.0) -> tuple:
    """Convert lat/lon to Cartesian coordinates on sphere."""
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    
    return x, y, z


def create_interactive_sphere(
    times: np.ndarray,
    coeffs: np.ndarray,
    lmax: int,
    n_interp_frames: int = 100,
    output_path: Path | None = None,
) -> go.Figure:
    """
    Create interactive 3D sphere with temporal animation.
    
    Args:
        times: Array of datetime64 timestamps
        coeffs: Array of shape (n_windows, n_coeffs) with spherical harmonic coefficients
        lmax: Maximum spherical harmonic degree
        n_interp_frames: Number of interpolated frames for smooth animation
        output_path: Optional path to save HTML file
    """
    n_windows = len(times)
    
    # Convert times to hours since start for interpolation
    t0 = times[0]
    hours = (times - t0) / np.timedelta64(1, 'h')
    hours_interp = np.linspace(hours[0], hours[-1], n_interp_frames)
    
    # Interpolate coefficients for smooth temporal evolution
    print(f"Interpolating {coeffs.shape[1]} coefficients across {n_interp_frames} frames...")
    coeffs_interp = np.zeros((n_interp_frames, coeffs.shape[1]), dtype=np.complex128)
    
    for i in range(coeffs.shape[1]):
        # Interpolate real and imaginary parts separately
        f_real = interp1d(hours, coeffs[:, i].real, kind='cubic', fill_value='extrapolate')
        f_imag = interp1d(hours, coeffs[:, i].imag, kind='cubic', fill_value='extrapolate')
        coeffs_interp[:, i] = f_real(hours_interp) + 1j * f_imag(hours_interp)
    
    # Generate potential maps for each interpolated time
    print("Reconstructing potential maps on sphere...")
    latitudes, longitudes, potential_0 = reconstruct_global_map(
        coeffs_interp[0], lmax, lat_steps=n_lat, lon_steps=n_lon
    )
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    x, y, z = latlon_to_xyz(lat_grid, lon_grid)
    
    # Determine global color scale from all frames
    print("Computing global color range...")
    all_potentials = []
    for i in range(min(20, n_interp_frames)):  # Sample to save time
        _, _, pot = reconstruct_global_map(
            coeffs_interp[i], lmax, lat_steps=n_lat, lon_steps=n_lon
        )
        all_potentials.append(pot)
    
    vmin = np.percentile(np.concatenate([p.ravel() for p in all_potentials]), 1)
    vmax = np.percentile(np.concatenate([p.ravel() for p in all_potentials]), 99)
    
    print(f"Color scale: {vmin:.0f} to {vmax:.0f} V")
    
    # Create frames for animation
    print("Creating animation frames...")
    frames = []
    
    for i in range(n_interp_frames):
        _, _, potential = reconstruct_global_map(
            coeffs_interp[i], lmax, lat_steps=n_lat, lon_steps=n_lon
        )
        
        # Convert interpolated time back to datetime
        t_interp = t0 + np.timedelta64(int(hours_interp[i] * 3600), 's')
        
        frame = go.Frame(
            data=[go.Surface(
                x=x, y=y, z=z,
                surfacecolor=potential,
                colorscale='RdBu_r',
                cmin=vmin,
                cmax=vmax,
                showscale=True,
                colorbar=dict(
                    title="Potential (V)",
                    x=1.02,
                ),
            )],
            name=str(i),
            layout=go.Layout(
                title_text=f"Lunar Surface Potential - {np.datetime_as_string(t_interp, unit='D')}"
            ),
        )
        frames.append(frame)
        
        if (i + 1) % 20 == 0:
            print(f"  Created {i+1}/{n_interp_frames} frames")
    
    # Create initial figure
    fig = go.Figure(
        data=[go.Surface(
            x=x, y=y, z=z,
            surfacecolor=potential_0,
            colorscale='RdBu_r',
            cmin=vmin,
            cmax=vmax,
            showscale=True,
            colorbar=dict(
                title="Potential (V)",
                x=1.02,
            ),
        )],
        frames=frames,
    )
    
    # Add slider and play/pause buttons
    fig.update_layout(
        title=f"Lunar Surface Potential Evolution (lmax={lmax})",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.5),
            ),
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {
                            "frame": {"duration": 100, "redraw": True},
                            "fromcurrent": True,
                            "mode": "immediate",
                            "transition": {"duration": 50},
                        }],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        }],
                    ),
                ],
                x=0.1,
                y=0,
                xanchor="left",
                yanchor="top",
            ),
        ],
        sliders=[
            dict(
                active=0,
                steps=[
                    dict(
                        args=[
                            [str(i)],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        method="animate",
                        label=np.datetime_as_string(
                            t0 + np.timedelta64(int(hours_interp[i] * 3600), 's'),
                            unit='D',
                        ),
                    )
                    for i in range(0, n_interp_frames, max(1, n_interp_frames // 20))
                ],
                x=0.1,
                y=0,
                len=0.9,
                xanchor="left",
                yanchor="top",
                pad=dict(b=10, t=50),
                currentvalue=dict(
                    visible=True,
                    prefix="Date: ",
                    xanchor="right",
                    font=dict(size=16),
                ),
            ),
        ],
        width=1200,
        height=800,
    )
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        print(f"\nSaved interactive visualization to {output_path}")
        print(f"File size: {output_path.stat().st_size / 1e6:.1f} MB")
    
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create interactive 3D visualization of temporal lunar potential",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="NPZ file with temporal coefficients",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/interactive_temporal_sphere.html"),
        help="Output HTML file path",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="Number of interpolated frames for smooth animation",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=91,
        help="Latitude resolution (higher = smoother but slower)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    if not args.input.exists():
        print(f"Error: {args.input} not found")
        return 1
    
    print(f"Loading temporal coefficients from {args.input}")
    dataset = load_temporal_coefficients(args.input)
    times = dataset.times
    coeffs = dataset.coeffs
    lmax = dataset.lmax
    
    print(f"\nDataset info:")
    print(f"  Date range: {np.datetime_as_string(times[0], unit='D')} to {np.datetime_as_string(times[-1], unit='D')}")
    print(f"  Time windows: {len(times)}")
    print(f"  lmax: {lmax}")
    print(f"  Coefficients per window: {coeffs.shape[1]}")
    print(f"  Animation frames: {args.frames}")
    print()
    
    fig = create_interactive_sphere(
        times,
        coeffs,
        lmax,
        n_interp_frames=args.frames,
        output_path=args.output,
    )
    
    print("\nDone! Open the HTML file in a browser to explore the interactive visualization.")
    print("Use the play button or slider to animate through time.")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
