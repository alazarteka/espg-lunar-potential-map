"""Interactive energy–pitch explorer for a day's spectra."""

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

import src.config as config
from src.flux import ERData, PitchAngle
from src.utils.flux_files import select_flux_day_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive energy–pitch scatter with a slider over spectrum numbers. "
            "Each frame shows one spectrum's channels as points colored by flux."
        )
    )
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--day", type=int, required=True)
    parser.add_argument(
        "--output", type=str, default=None, help="Save HTML to this path"
    )
    parser.add_argument(
        "-d", "--display", action="store_true", default=False, help="Open in browser"
    )
    parser.add_argument(
        "--include-zeros",
        action="store_true",
        default=False,
        help="Plot non-positive flux points in gray for context",
    )
    parser.add_argument(
        "--mode",
        choices=["points", "bars"],
        default="bars",
        help="Render as points (scatter) or bars spanning [0.75E, 1.25E]",
    )
    parser.add_argument(
        "--bar-width-deg",
        type=float,
        default=2.0,
        help="Override bar thickness in degrees for bars mode (default: 2 degrees)",
    )
    return parser.parse_args()


def _pitch_edges_from_centers(centers: np.ndarray) -> np.ndarray:
    edges = np.empty(centers.size + 1, dtype=np.float64)
    if centers.size > 1:
        edges[1:-1] = 0.5 * (centers[1:] + centers[:-1])
        edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
        edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    else:
        edges[:] = [centers[0] - 0.5, centers[0] + 0.5]
    return edges


def build_spec_points(
    er: ERData, pa: PitchAngle, spec_no: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = er.data[config.SPEC_NO_COLUMN] == spec_no
    idxs = np.nonzero(mask.to_numpy())[0]
    if idxs.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    energies = er.data.loc[mask, config.ENERGY_COLUMN].to_numpy(dtype=np.float64)
    flux_mat = er.data.loc[mask, config.FLUX_COLS].to_numpy(dtype=np.float64)
    pitch_mat = pa.pitch_angles[idxs, :]

    # Sort channels by the first row's pitch to keep a stable order
    order = np.argsort(pitch_mat[0, :])
    pitch_sorted = pitch_mat[:, order]
    flux_sorted = flux_mat[:, order]

    # Compute per-point pitch thickness using row-wise edges
    R, C = flux_sorted.shape
    y_widths = np.zeros_like(pitch_sorted)
    for i in range(R):
        y_row = pitch_sorted[i, :]
        local_order = np.argsort(y_row)
        edges_sorted = _pitch_edges_from_centers(y_row[local_order])
        widths_sorted = edges_sorted[1:] - edges_sorted[:-1]
        widths_row = np.empty_like(y_row)
        widths_row[local_order] = widths_sorted
        # Ensure non-negative bar thickness; zero is allowed but avoid negatives
        y_widths[i, :] = np.abs(widths_row)

    # Flatten to point lists
    x_pts = np.repeat(energies, C)
    y_pts = pitch_sorted.reshape(-1)
    f_pts = flux_sorted.reshape(-1)
    yw_pts = y_widths.reshape(-1)
    # Keep only finite values with positive energy and finite flux
    mask_valid = (
        np.isfinite(x_pts) & (x_pts > 0) & np.isfinite(y_pts) & np.isfinite(f_pts)
    )
    return x_pts[mask_valid], y_pts[mask_valid], f_pts[mask_valid], yw_pts[mask_valid]


def build_all_frames(
    er: ERData, pa: PitchAngle
) -> tuple[
    dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    float,
    float,
    np.ndarray,
    tuple[float, float],
    tuple[float, float],
    int | None,
]:
    spec_nos = er.data[config.SPEC_NO_COLUMN].unique()
    data_by_spec: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    all_pos = []
    x_min, x_max = np.inf, -np.inf
    y_min, y_max = np.inf, -np.inf
    first_nonempty: int | None = None

    for sn in spec_nos:
        x, y, f, yw = build_spec_points(er, pa, int(sn))
        data_by_spec[int(sn)] = (x, y, f, yw)
        if f.size:
            # Collect positive flux only for color scaling
            pos = f[f > 0]
            if pos.size:
                all_pos.append(pos)
            x_min = float(min(x_min, float(np.min(x))))
            x_max = float(max(x_max, float(np.max(x))))
            y_min = float(min(y_min, float(np.min(y))))
            y_max = float(max(y_max, float(np.max(y))))
            if first_nonempty is None:
                first_nonempty = int(sn)

    if all_pos:
        concat = np.concatenate(all_pos)
        vmin = float(max(1e-3, np.percentile(concat, 1)))
        vmax = float(np.percentile(concat, 99))
        if vmin >= vmax:
            vmin = float(max(1e-3, float(np.min(concat))))
            vmax = float(np.max(concat))
    else:
        vmin, vmax = 1e-3, 1.0

    # Provide axis ranges with small padding
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        x_min, x_max = 1e0, 1e1
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        y_min, y_max = 0.0, 180.0

    # Add slight padding for aesthetics
    x_range = (x_min / 1.3, x_max * 1.3)
    y_pad = max(2.0, 0.05 * (y_max - y_min))
    y_range = (max(0.0, y_min - y_pad), min(180.0, y_max + y_pad))

    return data_by_spec, vmin, vmax, spec_nos, x_range, y_range, first_nonempty


def make_figure(
    data_by_spec: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    spec_nos: np.ndarray,
    vmin: float,
    vmax: float,
    title: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    first_nonempty: int | None,
    include_zeros: bool,
    mode: str,
    bar_width_deg: float | None,
) -> go.Figure:
    # Initial spectrum: pick first with data, else first entry
    initial_sn = int(first_nonempty if first_nonempty is not None else int(spec_nos[0]))
    x0, y0, f0, yw0 = data_by_spec.get(
        initial_sn, (np.array([]), np.array([]), np.array([]), np.array([]))
    )
    # Common log-color mapping setup
    eps = 1e-300
    cmin_log = np.log10(max(eps, vmin))
    cmax_log = np.log10(max(eps, vmax))

    traces = []
    if mode == "points":
        # Split into positive and non-positive for plotting
        pos_mask0 = f0 > 0
        x0_pos, y0_pos, f0_pos = x0[pos_mask0], y0[pos_mask0], f0[pos_mask0]
        x0_non, y0_non = x0[~pos_mask0], y0[~pos_mask0]

        color_pos0 = np.log10(np.clip(f0_pos, eps, None))
        traces.append(
            go.Scatter(
                x=x0_pos,
                y=y0_pos,
                mode="markers",
                marker=dict(
                    size=9,
                    color=color_pos0,
                    coloraxis="coloraxis",
                    line=dict(color="#222", width=0.6),
                ),
                customdata=f0_pos,
                hovertemplate=(
                    "Energy: %{x:.3g} eV<br>Pitch: %{y:.1f}°<br>Flux: %{customdata:.3g}<extra></extra>"
                ),
                name="flux>0",
            )
        )
        if include_zeros and x0_non.size:
            traces.append(
                go.Scatter(
                    x=x0_non,
                    y=y0_non,
                    mode="markers",
                    marker=dict(
                        size=7, color="#C0C0C0", line=dict(color="#666", width=0.4)
                    ),
                    hoverinfo="skip",
                    name="flux<=0",
                )
            )
    else:  # bars
        # Build bar base/length from energy width and use pitch thickness for bar height
        base0 = 0.75 * x0
        width0 = 0.5 * x0
        pos_mask0 = f0 > 0

        # Positive flux bars with log color
        traces.append(
            go.Bar(
                x=width0[pos_mask0],
                y=y0[pos_mask0],
                base=base0[pos_mask0],
                orientation="h",
                width=(bar_width_deg if bar_width_deg is not None else yw0[pos_mask0]),
                marker=dict(
                    color=np.log10(np.clip(f0[pos_mask0], eps, None)),
                    coloraxis="coloraxis",
                    line=dict(color="#222", width=0.2),
                ),
                customdata=np.stack(
                    [
                        base0[pos_mask0],
                        base0[pos_mask0] + width0[pos_mask0],
                        f0[pos_mask0],
                        0.5 * yw0[pos_mask0],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "Energy: %{customdata[0]:.3g}–%{customdata[1]:.3g} eV"
                    "<br>Pitch: %{y:.1f}° (±%{customdata[3]:.1f}°)"
                    "<br>Flux: %{customdata[2]:.3g}<extra></extra>"
                ),
                name="flux>0",
            )
        )
        if include_zeros:
            # Non-positive flux bars in gray
            traces.append(
                go.Bar(
                    x=width0[~pos_mask0],
                    y=y0[~pos_mask0],
                    base=base0[~pos_mask0],
                    orientation="h",
                    width=(
                        bar_width_deg if bar_width_deg is not None else yw0[~pos_mask0]
                    ),
                    marker=dict(color="#C0C0C0", line=dict(color="#666", width=0.2)),
                    hoverinfo="skip",
                    name="flux<=0",
                )
            )

    frames = []
    slider_steps = []
    for sn in spec_nos:
        sn_int = int(sn)
        x, y, f, yw = data_by_spec[sn_int]
        pos_mask = f > 0
        if mode == "points":
            x_pos, y_pos, f_pos = x[pos_mask], y[pos_mask], f[pos_mask]
            x_non, y_non = x[~pos_mask], y[~pos_mask]
            color_pos = np.log10(np.clip(f_pos, eps, None))

            frame_traces = [
                go.Scatter(
                    x=x_pos,
                    y=y_pos,
                    mode="markers",
                    marker=dict(
                        size=9,
                        color=color_pos,
                        coloraxis="coloraxis",
                        line=dict(color="#222", width=0.6),
                    ),
                    customdata=f_pos,
                    hovertemplate=(
                        "Energy: %{x:.3g} eV<br>Pitch: %{y:.1f}°<br>Flux: %{customdata:.3g}<extra></extra>"
                    ),
                    name="flux>0",
                )
            ]
            if include_zeros:
                frame_traces.append(
                    go.Scatter(
                        x=x[~pos_mask],
                        y=y[~pos_mask],
                        mode="markers",
                        marker=dict(
                            size=7, color="#C0C0C0", line=dict(color="#666", width=0.4)
                        ),
                        hoverinfo="skip",
                        name="flux<=0",
                    )
                )
        else:
            base = 0.75 * x
            width = 0.5 * x
            frame_traces = [
                go.Bar(
                    x=width[pos_mask],
                    y=y[pos_mask],
                    base=base[pos_mask],
                    orientation="h",
                    width=(
                        bar_width_deg if bar_width_deg is not None else yw[pos_mask]
                    ),
                    marker=dict(
                        color=np.log10(np.clip(f[pos_mask], eps, None)),
                        coloraxis="coloraxis",
                        line=dict(color="#222", width=0.2),
                    ),
                    customdata=np.stack(
                        [
                            base[pos_mask],
                            base[pos_mask] + width[pos_mask],
                            f[pos_mask],
                            0.5 * yw[pos_mask],
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "Energy: %{customdata[0]:.3g}–%{customdata[1]:.3g} eV"
                        "<br>Pitch: %{y:.1f}° (±%{customdata[3]:.1f}°)"
                        "<br>Flux: %{customdata[2]:.3g}<extra></extra>"
                    ),
                    name="flux>0",
                )
            ]
            if include_zeros:
                frame_traces.append(
                    go.Bar(
                        x=width[~pos_mask],
                        y=y[~pos_mask],
                        base=base[~pos_mask],
                        orientation="h",
                        width=(
                            bar_width_deg
                            if bar_width_deg is not None
                            else yw[~pos_mask]
                        ),
                        marker=dict(
                            color="#C0C0C0", line=dict(color="#666", width=0.2)
                        ),
                        hoverinfo="skip",
                        name="flux<=0",
                    )
                )
        frames.append(go.Frame(name=str(sn_int), data=frame_traces))
        slider_steps.append(
            {
                "args": [
                    [str(sn_int)],
                    {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    },
                ],
                "label": str(sn_int),
                "method": "animate",
            }
        )

    # Configure a shared coloraxis with a visible colorbar on the right
    tick_lo = int(np.floor(cmin_log))
    tick_hi = int(np.ceil(cmax_log))
    # Prefer decade ticks if the span is modest; otherwise fallback to 3 anchors
    if tick_hi - tick_lo <= 6:
        tickvals = list(range(tick_lo, tick_hi + 1))
    else:
        tick_mid = int(np.floor((tick_lo + tick_hi) / 2))
        tickvals = [tick_lo, tick_mid, tick_hi]
    ticktext = [f"1e{t}" for t in tickvals]
    coloraxis = dict(
        colorscale="Turbo",
        cmin=cmin_log,
        cmax=cmax_log,
        colorbar=dict(
            title="Flux [cm⁻² s⁻¹ sr⁻¹ eV⁻¹]",
            tickvals=tickvals,
            ticktext=ticktext,
            x=1.02,
        ),
    )

    layout = go.Layout(
        title=title,
        template="plotly_white",
        xaxis=dict(
            title="Energy (eV)",
            type="log",
            range=[np.log10(x_range[0]), np.log10(x_range[1])],
        ),
        yaxis=dict(title="Pitch angle (deg)", range=[y_range[0], y_range[1]]),
        coloraxis=coloraxis,
        sliders=[
            {
                "active": 0,
                "y": 0,
                "x": 0.1,
                "xanchor": "left",
                "yanchor": "top",
                "len": 0.9,
                "pad": {"t": 30, "b": 10},
                "currentvalue": {"prefix": "Spectrum: ", "visible": True},
                "steps": slider_steps,
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.1,
                "y": 1.15,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": True,
                                "frame": {"duration": 100, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                            },
                        ],
                    },
                ],
            }
        ],
        margin=dict(l=60, r=20, t=60, b=60),
        barmode="overlay" if mode == "bars" else None,
    )

    fig = go.Figure(data=traces, layout=layout, frames=frames)
    return fig


def main() -> None:
    args = parse_args()
    day_file = select_flux_day_file(args.year, args.month, args.day)
    er = ERData(str(day_file))
    pa = PitchAngle(er, str(config.DATA_DIR / config.THETA_FILE))

    data_by_spec, vmin, vmax, spec_nos, x_range, y_range, first_nonempty = (
        build_all_frames(er, pa)
    )

    title = f"Flux vs Energy/Pitch — {args.year:04d}-{args.month:02d}-{args.day:02d}"
    fig = make_figure(
        data_by_spec,
        spec_nos,
        vmin,
        vmax,
        title,
        x_range,
        y_range,
        first_nonempty,
        include_zeros=args.include_zeros,
        mode=args.mode,
        bar_width_deg=args.bar_width_deg,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import plotly.io as pio

        pio.write_html(fig, file=str(out_path), include_plotlyjs="cdn", auto_open=False)
        print(f"Saved interactive HTML to {out_path}")

    if args.display:
        fig.show()


if __name__ == "__main__":
    main()
