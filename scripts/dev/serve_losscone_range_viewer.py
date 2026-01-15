#!/usr/bin/env python3
"""
Serve an interactive loss-cone viewer for a single sweep across U_surface range.

Usage:
  uv run python scripts/dev/serve_losscone_range_viewer.py \
    data/1999/091_120APR/3D990429.TAB --chunk 0 --u-min -2000 --u-max 0 \
    --u-steps 81 --fast
  # Override U_sc with a constant
  uv run python scripts/dev/serve_losscone_range_viewer.py \
    data/1999/091_120APR/3D990429.TAB --chunk 0 --u-values -2000,-1000,0 \
    --u-spacecraft 0
  # Optimize Bs/Bm per U_surface across a grid
  uv run python scripts/dev/serve_losscone_range_viewer.py \
    data/1999/091_120APR/3D990429.TAB --chunk 0 --u-steps 41 \
    --bs-min 0.3 --bs-max 1.1 --bs-steps 17
"""

from __future__ import annotations

import argparse
import functools
import http.server
from pathlib import Path
from typing import Iterable

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src import config
from src.flux import ERData, LossConeFitter, PitchAngle
from src.model import synth_losscone_batch
from src.potential_mapper.spice import load_spice_files
from src.spacecraft_potential import calculate_potential

try:
    import torch

    from src.model_torch import HAS_TORCH, _auto_detect_dtype, synth_losscone_batch_torch
except Exception:  # pragma: no cover - optional GPU path
    HAS_TORCH = False
    torch = None  # type: ignore[assignment]
    synth_losscone_batch_torch = None  # type: ignore[assignment]
    _auto_detect_dtype = None  # type: ignore[assignment]


def parse_values(
    values: str | None, v_min: float, v_max: float, v_steps: int, name: str
) -> np.ndarray:
    if values:
        return np.array([float(x) for x in values.split(",")], dtype=float)
    if v_steps < 2:
        raise ValueError(f"{name}_steps must be >= 2 when using a range")
    return np.linspace(v_min, v_max, v_steps, dtype=float)


def interpolate_to_regular_grid(
    energies: np.ndarray,
    pitches: np.ndarray,
    flux_data: np.ndarray,
    n_pitch_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate irregular (energy, pitch) grid onto a regular pitch grid."""
    pitch_min = np.nanmin(pitches)
    pitch_max = np.nanmax(pitches)
    pitches_reg = np.linspace(pitch_min, pitch_max, n_pitch_bins)

    flux_reg = np.zeros((len(energies), n_pitch_bins))
    for i in range(len(energies)):
        valid_mask = np.isfinite(flux_data[i]) & np.isfinite(pitches[i])
        if np.sum(valid_mask) > 1:
            pitch_pts = pitches[i, valid_mask]
            flux_pts = flux_data[i, valid_mask]
            sort_idx = np.argsort(pitch_pts)
            pitch_pts_sorted = pitch_pts[sort_idx]
            flux_pts_sorted = flux_pts[sort_idx]
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


def compute_loss_cone_boundary(
    energies: np.ndarray,
    U_surface: float,
    bs_over_bm: float,
    u_spacecraft: float,
) -> np.ndarray:
    """Compute loss-cone boundary angle (pitch <= boundary is inside cone)."""
    E_corr = np.maximum(energies - u_spacecraft, config.EPS)
    x = bs_over_bm * (1.0 + U_surface / E_corr)
    x_clipped = np.clip(x, 0.0, 1.0)
    ac_deg = np.degrees(np.arcsin(np.sqrt(x_clipped)))
    return 180.0 - ac_deg


def build_models(
    energies: np.ndarray,
    pitches: np.ndarray,
    u_values: np.ndarray,
    bs_over_bm: float,
    beam_amp: float,
    beam_width: float,
    u_spacecraft: float,
    background: float,
    use_torch: bool,
) -> np.ndarray:
    n_params = len(u_values)
    bs_arr = np.full(n_params, bs_over_bm, dtype=float)
    beam_amp_arr = np.full(n_params, beam_amp, dtype=float)
    beam_width_arr = np.full(n_params, beam_width, dtype=float)
    background_arr = np.full(n_params, background, dtype=float)

    if use_torch:
        if not HAS_TORCH or synth_losscone_batch_torch is None or torch is None:
            raise ImportError("PyTorch is required for --fast mode.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = _auto_detect_dtype(device) if _auto_detect_dtype else torch.float32

        energies_t = torch.tensor(energies, device=device, dtype=dtype)
        pitches_t = torch.tensor(pitches, device=device, dtype=dtype)
        u_t = torch.tensor(u_values, device=device, dtype=dtype)
        bs_t = torch.tensor(bs_arr, device=device, dtype=dtype)
        beam_amp_t = torch.tensor(beam_amp_arr, device=device, dtype=dtype)
        beam_width_t = torch.tensor(beam_width_arr, device=device, dtype=dtype)
        background_t = torch.tensor(background_arr, device=device, dtype=dtype)

        models_t = synth_losscone_batch_torch(
            energy_grid=energies_t,
            pitch_grid=pitches_t,
            U_surface=u_t,
            U_spacecraft=float(u_spacecraft),
            bs_over_bm=bs_t,
            beam_width_eV=beam_width_t,
            beam_amp=beam_amp_t,
            beam_pitch_sigma_deg=config.LOSS_CONE_BEAM_PITCH_SIGMA_DEG,
            background=background_t,
        )
        return models_t.detach().cpu().numpy()

    return synth_losscone_batch(
        energy_grid=energies,
        pitch_grid=pitches,
        U_surface=u_values,
        U_spacecraft=float(u_spacecraft),
        bs_over_bm=bs_arr,
        beam_width_eV=beam_width_arr,
        beam_amp=beam_amp_arr,
        beam_pitch_sigma_deg=config.LOSS_CONE_BEAM_PITCH_SIGMA_DEG,
        background=background_arr,
    )


def serve_html(path: Path, port: int) -> None:
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(path.parent)
    )
    server = http.server.ThreadingHTTPServer(("localhost", port), handler)
    url = f"http://localhost:{port}/{path.name}"
    print(f"Serving {path} at {url}")
    print("Press Ctrl+C to stop the server")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Serve an interactive loss-cone viewer for a U_surface range.",
    )
    parser.add_argument("er_file", type=Path, help="Path to ER .TAB file")
    parser.add_argument("--chunk", type=int, default=0, help="0-based sweep index")
    parser.add_argument("--spec-no", type=int, default=None, help="Spec number")
    parser.add_argument(
        "--u-min", type=float, default=-2000.0, help="Minimum U_surface [V]"
    )
    parser.add_argument(
        "--u-max", type=float, default=0.0, help="Maximum U_surface [V]"
    )
    parser.add_argument(
        "--u-steps", type=int, default=81, help="Number of U_surface samples"
    )
    parser.add_argument(
        "--u-values",
        type=str,
        default=None,
        help="Comma-separated U_surface values (overrides range)",
    )
    parser.add_argument(
        "--bs-over-bm",
        type=float,
        default=None,
        help="Fixed B_s/B_m ratio (skip optimization)",
    )
    parser.add_argument(
        "--bs-min",
        type=float,
        default=None,
        help="Minimum B_s/B_m (enables slider)",
    )
    parser.add_argument(
        "--bs-max",
        type=float,
        default=None,
        help="Maximum B_s/B_m (enables slider)",
    )
    parser.add_argument(
        "--bs-steps",
        type=int,
        default=1,
        help="Number of B_s/B_m samples",
    )
    parser.add_argument(
        "--bs-values",
        type=str,
        default=None,
        help="Comma-separated B_s/B_m values (overrides range)",
    )
    parser.add_argument(
        "--beam-amp", type=float, default=1.0, help="Fixed beam amplitude"
    )
    parser.add_argument(
        "--beam-width",
        type=float,
        default=None,
        help="Beam width [eV] (default: fitter config)",
    )
    parser.add_argument(
        "--u-spacecraft",
        type=float,
        default=None,
        help="Spacecraft potential [V] (constant override)",
    )
    parser.add_argument(
        "--normalization",
        choices=["global", "ratio", "ratio2", "ratio_rescaled"],
        default="ratio",
        help="Flux normalization mode for data",
    )
    parser.add_argument(
        "--incident-stat",
        choices=["mean", "max"],
        default="mean",
        help="Incident flux statistic for normalization",
    )
    parser.add_argument(
        "--n-pitch-bins",
        type=int,
        default=100,
        help="Number of pitch bins for interpolation",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Use log10 scale for both data and model",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use PyTorch-accelerated loss-cone model",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scratch/loss_cone_range_viewer.html"),
        help="Output HTML path",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port for local HTTP server"
    )
    parser.add_argument(
        "--no-serve",
        action="store_true",
        help="Write HTML but do not start a server",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    u_values = parse_values(args.u_values, args.u_min, args.u_max, args.u_steps, "u")
    if len(u_values) == 0:
        raise ValueError("No U_surface values provided")
    if args.bs_over_bm is None:
        bs_min = (
            config.LOSS_CONE_BS_OVER_BM_MIN if args.bs_min is None else args.bs_min
        )
        bs_max = (
            config.LOSS_CONE_BS_OVER_BM_MAX if args.bs_max is None else args.bs_max
        )
        bs_values = parse_values(
            args.bs_values, bs_min, bs_max, args.bs_steps, "bs"
        )
    else:
        bs_values = np.array([args.bs_over_bm], dtype=float)
    if len(bs_values) == 0:
        raise ValueError("No bs_over_bm values provided")
    if args.n_pitch_bins < 2:
        raise ValueError("n_pitch_bins must be >= 2")
    if len(u_values) * len(bs_values) > 800:
        print("Warning: large U/bs grid may generate a heavy HTML file.")

    er_data = ERData(str(args.er_file))
    pitch_angle = PitchAngle(er_data)

    if args.spec_no is not None:
        spec_vals = er_data.data[config.SPEC_NO_COLUMN].to_numpy()
        matches = np.where(spec_vals == args.spec_no)[0]
        if len(matches) == 0:
            raise ValueError(f"Spec number {args.spec_no} not found")
        args.chunk = int(matches[0] // config.SWEEP_ROWS)

    n_rows = len(er_data.data)
    start = args.chunk * config.SWEEP_ROWS
    end = min(start + config.SWEEP_ROWS, n_rows)
    if start >= n_rows:
        raise ValueError(f"Chunk {args.chunk} out of range (rows={n_rows})")

    if args.u_spacecraft is None:
        spec_no = int(er_data.data.iloc[start][config.SPEC_NO_COLUMN])
        try:
            load_spice_files()
            sc_result = calculate_potential(er_data, spec_no)
        except Exception as exc:  # pragma: no cover - depends on SPICE availability
            sc_result = None
            print(f"Warning: failed to compute spacecraft potential ({exc})")
        if sc_result is None:
            print("Warning: using U_spacecraft=0.0; pass --u-spacecraft to override")
            u_spacecraft = 0.0
        else:
            _params, u_sc = sc_result
            u_spacecraft = float(u_sc.magnitude)
    else:
        u_spacecraft = float(args.u_spacecraft)

    spacecraft_potential = np.full(n_rows, u_spacecraft, dtype=float)
    fitter = LossConeFitter(
        er_data,
        str(config.DATA_DIR / config.THETA_FILE),
        pitch_angle=pitch_angle,
        spacecraft_potential=spacecraft_potential,
        normalization_mode=args.normalization,
        incident_flux_stat=args.incident_stat,
    )

    norm2d = fitter.build_norm2d(args.chunk)
    norm2d = norm2d[: end - start]
    data = np.where(np.isfinite(norm2d) & (norm2d > 0), norm2d, np.nan)
    if not np.isfinite(data).any():
        raise ValueError("No finite normalized data available for this chunk")

    energies = er_data.data[config.ENERGY_COLUMN].to_numpy(dtype=float)[start:end]
    pitches = pitch_angle.pitch_angles[start:end]
    timestamp = er_data.data.iloc[start][config.TIME_COLUMN]

    energies_reg, pitches_reg, data_reg = interpolate_to_regular_grid(
        energies, pitches, data, args.n_pitch_bins
    )
    if args.log_scale:
        eps = config.EPS
        data_reg_plot = np.log10(np.maximum(data_reg, eps))
    else:
        data_reg_plot = data_reg

    beam_width = (
        args.beam_width if args.beam_width is not None else fitter.beam_width_ev
    )
    data_mask = np.isfinite(data) & (data > 0)
    log_data = np.zeros_like(data)
    log_data[data_mask] = np.log(data[data_mask] + config.EPS)
    data_mask_3d = data_mask[None, :, :]

    models_by_bs: list[np.ndarray] = []
    chi2_grid = np.full((len(bs_values), len(u_values)), np.nan, dtype=float)

    for bs_idx, bs_val in enumerate(bs_values):
        models = build_models(
            energies=energies,
            pitches=pitches,
            u_values=u_values,
            bs_over_bm=float(bs_val),
            beam_amp=args.beam_amp,
            beam_width=beam_width,
            u_spacecraft=u_spacecraft,
            background=fitter.background,
            use_torch=args.fast,
        )
        models_by_bs.append(models)
        log_models = np.log(models + config.EPS)
        diff = (log_data[None, :, :] - log_models) * data_mask_3d
        chi2 = np.sum(diff * diff, axis=(1, 2))
        chi2[~np.isfinite(chi2)] = 1e30
        chi2_grid[bs_idx] = chi2

    best_bs_idx = np.argmin(chi2_grid, axis=0)

    model_frames: list[dict] = []
    for u_idx, u_val in enumerate(u_values):
        bs_idx = int(best_bs_idx[u_idx])
        bs_val = float(bs_values[bs_idx])
        model = models_by_bs[bs_idx][u_idx]
        _, _, model_reg = interpolate_to_regular_grid(
            energies, pitches, model, args.n_pitch_bins
        )
        if args.log_scale:
            eps = config.EPS
            model_reg_plot = np.log10(np.maximum(model_reg, eps))
        else:
            model_reg_plot = model_reg
        loss_cone = compute_loss_cone_boundary(
            energies, float(u_val), bs_val, u_spacecraft
        )
        model_frames.append(
            {
                "u_val": float(u_val),
                "bs_val": bs_val,
                "model": model_reg_plot,
                "loss_cone": loss_cone,
                "data": data_reg_plot,
                "chi2": float(chi2_grid[bs_idx, u_idx]),
            }
        )

    if not model_frames:
        raise ValueError("No model frames generated")

    first = model_frames[0]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Observed (normalized)", "Model"),
        horizontal_spacing=0.12,
    )

    fig.add_trace(
        go.Heatmap(
            z=first["data"].T,
            x=energies_reg,
            y=pitches_reg,
            colorscale="Viridis",
            name="Observed",
            colorbar=dict(x=0.45, len=0.9, title="Data"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=energies,
            y=first["loss_cone"],
            mode="lines",
            line=dict(color="white", width=2),
            name="Loss Cone",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=first["model"].T,
            x=energies_reg,
            y=pitches_reg,
            colorscale="Viridis",
            name="Model",
            colorbar=dict(x=1.02, len=0.9, title="Model"),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=energies,
            y=first["loss_cone"],
            mode="lines",
            line=dict(color="white", width=2),
            name="Loss Cone",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(type="log", title_text="Energy [eV]", row=1, col=1)
    fig.update_xaxes(type="log", title_text="Energy [eV]", row=1, col=2)
    fig.update_yaxes(title_text="Pitch Angle [deg]", row=1, col=1)
    fig.update_yaxes(title_text="Pitch Angle [deg]", row=1, col=2)

    fig.update_layout(
        title=dict(
            text=(
                f"Chunk {args.chunk}: {timestamp}<br>"
                f"U_surface = {first['u_val']:.1f} V, Bs/Bm = {first['bs_val']:.2f}, "
                f"Beam Amp = {args.beam_amp:.1f}, U_sc = {u_spacecraft:.1f} V"
            ),
            x=0.5,
            xanchor="center",
        ),
        height=600,
        showlegend=False,
    )

    steps = []
    for frame in model_frames:
        steps.append(
            dict(
                method="update",
                args=[
                    {
                        "z": [frame["data"].T, None, frame["model"].T, None],
                        "x": [energies_reg, energies, energies_reg, energies],
                        "y": [
                            pitches_reg,
                            frame["loss_cone"],
                            pitches_reg,
                            frame["loss_cone"],
                        ],
                    },
                    {
                        "title": dict(
                            text=(
                                f"Chunk {args.chunk}: {timestamp}<br>"
                                f"U_surface = {frame['u_val']:.1f} V, "
                                f"Bs/Bm = {frame['bs_val']:.2f}, "
                                f"Beam Amp = {args.beam_amp:.1f}, "
                                f"U_sc = {u_spacecraft:.1f} V"
                            ),
                            x=0.5,
                            xanchor="center",
                        )
                    },
                ],
                label=f"{frame['u_val']:.0f}",
            )
        )

    fig.update_layout(
        sliders=[
            dict(
                active=0,
                yanchor="top",
                y=-0.15,
                xanchor="left",
                currentvalue=dict(prefix="U_surface: ", visible=True, xanchor="right"),
                pad=dict(b=10, t=50),
                len=0.9,
                x=0.05,
                steps=steps,
            )
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(args.output), include_plotlyjs="cdn")
    print(f"Wrote {args.output}")
    if args.fast and not HAS_TORCH:
        print("Warning: --fast requested but torch not available; used CPU path.")

    if not args.no_serve:
        serve_html(args.output, args.port)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
