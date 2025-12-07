#!/usr/bin/env python3
"""Create Plotly time series of lunar surface potential."""

import argparse

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def main():
    parser = argparse.ArgumentParser(description="Plot surface potential time series")
    parser.add_argument("--input", required=True, help="NPZ potential cache file")
    parser.add_argument(
        "--output",
        default="artifacts/plots/potential_timeseries.html",
        help="Output HTML file",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1000,
        help="Downsample to N points for faster rendering",
    )
    args = parser.parse_args()

    print(f"Loading data from {args.input}")
    data = np.load(args.input)

    # Extract data
    times = data["rows_time"]  # Unix timestamps
    proj_potential = data["rows_projected_potential"]
    proj_lat = data["rows_projection_latitude"]
    proj_lon = data["rows_projection_longitude"]
    in_sun = data["rows_projection_in_sun"]

    # Convert to datetime
    datetimes = [np.datetime64(int(float(t)), "s") for t in times]

    # Filter valid potentials
    valid = np.isfinite(proj_potential) & np.isfinite(proj_lat)

    datetimes = np.array(datetimes)[valid]
    proj_potential = proj_potential[valid]
    proj_lat = proj_lat[valid]
    proj_lon = proj_lon[valid]
    in_sun = in_sun[valid]

    print(f"Valid measurements: {len(proj_potential):,}")

    # Downsample if requested
    if args.sample and len(proj_potential) > args.sample:
        indices = np.linspace(0, len(proj_potential) - 1, args.sample, dtype=int)
        datetimes = datetimes[indices]
        proj_potential = proj_potential[indices]
        proj_lat = proj_lat[indices]
        proj_lon = proj_lon[indices]
        in_sun = in_sun[indices]
        print(f"Downsampled to {args.sample} points")

    # Separate day and night
    day_mask = in_sun
    night_mask = ~in_sun

    # Create figure with subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Surface Potential vs Time", "Spatial Coverage"),
        vertical_spacing=0.12,
        row_heights=[0.7, 0.3],
    )

    # Time series plot
    if np.any(day_mask):
        fig.add_trace(
            go.Scattergl(
                x=datetimes[day_mask],
                y=proj_potential[day_mask],
                mode="markers",
                marker=dict(size=2, color="orange", opacity=0.5),
                name="Dayside",
                hovertemplate="%{x}<br>Potential: %{y:.1f} V<extra></extra>",
            ),
            row=1,
            col=1,
        )

    if np.any(night_mask):
        fig.add_trace(
            go.Scattergl(
                x=datetimes[night_mask],
                y=proj_potential[night_mask],
                mode="markers",
                marker=dict(size=2, color="darkblue", opacity=0.5),
                name="Nightside",
                hovertemplate="%{x}<br>Potential: %{y:.1f} V<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Spatial coverage (lat/lon scatter)
    fig.add_trace(
        go.Scattergl(
            x=proj_lon,
            y=proj_lat,
            mode="markers",
            marker=dict(
                size=2,
                color=proj_potential,
                colorscale="RdBu_r",
                showscale=True,
                colorbar=dict(title="Potential (V)", y=0.2, len=0.3),
                opacity=0.6,
            ),
            name="Coverage",
            hovertemplate="Lat: %{y:.1f}°<br>Lon: %{x:.1f}°<br>Potential: %{marker.color:.1f} V<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Update axes
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text="Surface Potential (V)", row=1, col=1)

    fig.update_xaxes(title_text="Longitude (°)", row=2, col=1)
    fig.update_yaxes(title_text="Latitude (°)", row=2, col=1)

    # Update layout
    fig.update_layout(
        height=900,
        title_text="Lunar Surface Potential Time Series",
        showlegend=True,
        hovermode="closest",
    )

    # Save
    fig.write_html(args.output, include_plotlyjs="cdn")
    print(f"Saved to {args.output}")

    # Print stats
    print("\nStatistics:")
    print(f"  Date range: {datetimes[0]} to {datetimes[-1]}")
    print(
        f"  Potential range: {np.nanmin(proj_potential):.1f} to {np.nanmax(proj_potential):.1f} V"
    )
    print(f"  Mean potential: {np.nanmean(proj_potential):.1f} V")
    print(f"  Dayside fraction: {100 * np.sum(day_mask) / len(day_mask):.1f}%")


if __name__ == "__main__":
    main()
