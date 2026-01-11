#!/usr/bin/env python3
"""
Serve an interactive viewer for optimal B_s/B_m vs U_surface across spec_nos.

Usage:
  uv run python scripts/dev/serve_losscone_sc_bs_viewer.py \
    data/1999/091_120APR/3D990429.TAB --u-sc-min -500 --u-sc-max 100 \
    --u-sc-steps 61 --u-min -2000 --u-max 0 --u-steps 61 --fast
  # Limit to a few spectra
  uv run python scripts/dev/serve_losscone_sc_bs_viewer.py \
    data/1999/091_120APR/3D990429.TAB --spec-nos 650,651,652 --no-serve
"""

from __future__ import annotations

import argparse
import functools
import http.server
from pathlib import Path
from typing import Iterable

import numpy as np
import plotly.graph_objects as go

from src import config
from src.flux import ERData, LossConeFitter, PitchAngle
from src.model import synth_losscone_batch

try:
    import torch

    from src.model_torch import (
        HAS_TORCH,
        _auto_detect_dtype,
        compute_chi2_batch_torch,
        synth_losscone_batch_torch,
    )
except Exception:  # pragma: no cover - optional GPU path
    HAS_TORCH = False
    torch = None  # type: ignore[assignment]
    synth_losscone_batch_torch = None  # type: ignore[assignment]
    compute_chi2_batch_torch = None  # type: ignore[assignment]
    _auto_detect_dtype = None  # type: ignore[assignment]


def parse_values(
    values: str | None, v_min: float, v_max: float, v_steps: int, name: str
) -> np.ndarray:
    if values:
        return np.array([float(x) for x in values.split(",")], dtype=float)
    if v_steps < 2:
        raise ValueError(f"{name}_steps must be >= 2 when using a range")
    return np.linspace(v_min, v_max, v_steps, dtype=float)


def compute_best_bs_curve(
    energies: np.ndarray,
    pitches: np.ndarray,
    data: np.ndarray,
    u_sc_values: np.ndarray,
    u_values: np.ndarray,
    bs_values: np.ndarray,
    beam_amp: float,
    beam_width: float,
    background: float,
    use_torch: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_mask = np.isfinite(data) & (data > 0)
    if not data_mask.any():
        raise ValueError("No finite normalized data available for this spectrum")

    n_u = len(u_values)
    n_bs = len(bs_values)

    u_grid, bs_grid = np.meshgrid(u_values, bs_values, indexing="ij")
    u_flat = u_grid.ravel()
    bs_flat = bs_grid.ravel()

    beam_amp_arr = np.full_like(u_flat, beam_amp, dtype=float)
    beam_width_arr = np.full_like(u_flat, beam_width, dtype=float)
    background_arr = np.full_like(u_flat, background, dtype=float)

    best_bs = np.full(n_u, np.nan, dtype=float)
    best_u_sc = np.full(n_u, np.nan, dtype=float)
    best_chi2 = np.full(n_u, np.inf, dtype=float)

    # TODO: consider a batched U_sc path if the model supports per-parameter U_sc.
    if use_torch:
        if (
            not HAS_TORCH
            or synth_losscone_batch_torch is None
            or compute_chi2_batch_torch is None
            or torch is None
        ):
            raise ImportError("PyTorch is required for --fast mode.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = _auto_detect_dtype(device) if _auto_detect_dtype else torch.float32

        energies_t = torch.tensor(energies, device=device, dtype=dtype)
        pitches_t = torch.tensor(pitches, device=device, dtype=dtype)
        data_t = torch.tensor(data, device=device, dtype=dtype)
        data_mask_t = torch.tensor(data_mask, device=device, dtype=torch.bool)

        n_params = u_flat.size
        nE, nPitch = pitches.shape
        bytes_per = {torch.float16: 2, torch.float32: 4, torch.float64: 8}.get(
            dtype, 4
        )
        target_bytes = 200 * 1024 * 1024
        max_params = max(1024, int(target_bytes / (nE * nPitch * bytes_per)))
        param_batch = min(n_params, max_params)

        for u_sc_val in u_sc_values:
            u_sc_scalar = float(u_sc_val)
            chi2_all = np.empty(n_params, dtype=np.float64)
            for start in range(0, n_params, param_batch):
                end = min(start + param_batch, n_params)
                u_t = torch.tensor(u_flat[start:end], device=device, dtype=dtype)
                bs_t = torch.tensor(bs_flat[start:end], device=device, dtype=dtype)
                beam_amp_t = torch.tensor(
                    beam_amp_arr[start:end], device=device, dtype=dtype
                )
                beam_width_t = torch.tensor(
                    beam_width_arr[start:end], device=device, dtype=dtype
                )
                background_t = torch.tensor(
                    background_arr[start:end], device=device, dtype=dtype
                )

                models_t = synth_losscone_batch_torch(
                    energy_grid=energies_t,
                    pitch_grid=pitches_t,
                    U_surface=u_t,
                    U_spacecraft=u_sc_scalar,
                    bs_over_bm=bs_t,
                    beam_width_eV=beam_width_t,
                    beam_amp=beam_amp_t,
                    beam_pitch_sigma_deg=config.LOSS_CONE_BEAM_PITCH_SIGMA_DEG,
                    background=background_t,
                )
                chi2_t = compute_chi2_batch_torch(
                    models_t, data_t, data_mask_t, eps=config.EPS
                )
                chi2_all[start:end] = chi2_t.detach().cpu().numpy().astype(
                    np.float64, copy=False
                )

            chi2_all = np.nan_to_num(chi2_all, nan=1e30, posinf=1e30, neginf=1e30)
            chi2_grid = chi2_all.reshape(n_u, n_bs)
            bs_idx = np.argmin(chi2_grid, axis=1)
            candidate_chi2 = chi2_grid[np.arange(n_u), bs_idx]
            improve = candidate_chi2 < best_chi2
            if np.any(improve):
                best_chi2[improve] = candidate_chi2[improve]
                best_bs[improve] = bs_values[bs_idx[improve]]
                best_u_sc[improve] = u_sc_scalar

        return best_bs, best_u_sc, best_chi2

    log_data = np.zeros_like(data, dtype=float)
    log_data[data_mask] = np.log(data[data_mask] + config.EPS)
    data_mask_3d = data_mask[None, :, :]

    n_params = u_flat.size
    nE, nPitch = pitches.shape
    target_bytes = 200 * 1024 * 1024
    bytes_per = np.dtype(float).itemsize
    max_params = max(512, int(target_bytes / (nE * nPitch * bytes_per)))
    param_batch = min(n_params, max_params)

    for u_sc_val in u_sc_values:
        u_sc_scalar = float(u_sc_val)
        chi2_all = np.empty(n_params, dtype=np.float64)
        for start in range(0, n_params, param_batch):
            end = min(start + param_batch, n_params)
            models = synth_losscone_batch(
                energy_grid=energies,
                pitch_grid=pitches,
                U_surface=u_flat[start:end],
                U_spacecraft=u_sc_scalar,
                bs_over_bm=bs_flat[start:end],
                beam_width_eV=beam_width_arr[start:end],
                beam_amp=beam_amp_arr[start:end],
                beam_pitch_sigma_deg=config.LOSS_CONE_BEAM_PITCH_SIGMA_DEG,
                background=background_arr[start:end],
            )
            log_models = np.log(models + config.EPS)
            diff = (log_data[None, :, :] - log_models) * data_mask_3d
            chi2 = np.sum(diff * diff, axis=(1, 2)).astype(np.float64, copy=False)
            chi2_all[start:end] = chi2

        chi2_all = np.nan_to_num(chi2_all, nan=1e30, posinf=1e30, neginf=1e30)
        chi2_grid = chi2_all.reshape(n_u, n_bs)
        bs_idx = np.argmin(chi2_grid, axis=1)
        candidate_chi2 = chi2_grid[np.arange(n_u), bs_idx]
        improve = candidate_chi2 < best_chi2
        if np.any(improve):
            best_chi2[improve] = candidate_chi2[improve]
            best_bs[improve] = bs_values[bs_idx[improve]]
            best_u_sc[improve] = u_sc_scalar

    return best_bs, best_u_sc, best_chi2


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


def _parse_spec_nos(spec_values: np.ndarray) -> list[int]:
    spec_nos: list[int] = []
    seen: set[int] = set()
    for value in spec_values:
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value_f):
            continue
        spec_no = int(value_f)
        if spec_no in seen:
            continue
        seen.add(spec_no)
        spec_nos.append(spec_no)
    return spec_nos


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Serve an interactive viewer of optimal B_s/B_m vs U_surface across "
            "spectra."
        ),
    )
    parser.add_argument("er_file", type=Path, help="Path to ER .TAB file")
    parser.add_argument(
        "--spec-nos",
        type=str,
        default=None,
        help="Comma-separated spec numbers (defaults to all specs in file)",
    )
    parser.add_argument(
        "--spec-step",
        type=int,
        default=1,
        help="Stride for spec numbers (1 = keep all)",
    )
    parser.add_argument(
        "--spec-limit",
        type=int,
        default=None,
        help="Limit number of spectra to process",
    )
    parser.add_argument(
        "--u-sc-min",
        type=float,
        default=-1500.0,
        help="Minimum U_sc [V] (fit grid)",
    )
    parser.add_argument(
        "--u-sc-max",
        type=float,
        default=0.0,
        help="Maximum U_sc [V] (fit grid)",
    )
    parser.add_argument(
        "--u-sc-steps", type=int, default=61, help="Number of U_sc samples (fit grid)"
    )
    parser.add_argument(
        "--u-sc-values",
        type=str,
        default=None,
        help="Comma-separated U_sc values (overrides range, fit grid)",
    )
    parser.add_argument(
        "--u-min", type=float, default=-2000.0, help="Minimum U_surface [V]"
    )
    parser.add_argument(
        "--u-max", type=float, default=0.0, help="Maximum U_surface [V]"
    )
    parser.add_argument(
        "--u-steps", type=int, default=61, help="Number of U_surface samples"
    )
    parser.add_argument(
        "--u-values",
        type=str,
        default=None,
        help="Comma-separated U_surface values (overrides range)",
    )
    parser.add_argument(
        "--bs-min",
        type=float,
        default=config.LOSS_CONE_BS_OVER_BM_MIN,
        help="Minimum B_s/B_m",
    )
    parser.add_argument(
        "--bs-max",
        type=float,
        default=config.LOSS_CONE_BS_OVER_BM_MAX,
        help="Maximum B_s/B_m",
    )
    parser.add_argument(
        "--bs-steps",
        type=int,
        default=17,
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
        "--fast",
        action="store_true",
        help="Use PyTorch-accelerated loss-cone model",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scratch/loss_cone_sc_bs_viewer.html"),
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

    use_torch = args.fast
    if args.fast and not HAS_TORCH:
        print("Warning: --fast requested but torch not available; using CPU path.")
        use_torch = False

    u_sc_values = parse_values(
        args.u_sc_values, args.u_sc_min, args.u_sc_max, args.u_sc_steps, "u_sc"
    )
    if len(u_sc_values) == 0:
        raise ValueError("No U_sc values provided")

    u_values = parse_values(args.u_values, args.u_min, args.u_max, args.u_steps, "u")
    if len(u_values) == 0:
        raise ValueError("No U_surface values provided")

    bs_values = parse_values(
        args.bs_values, args.bs_min, args.bs_max, args.bs_steps, "bs"
    )
    if len(bs_values) == 0:
        raise ValueError("No bs_over_bm values provided")

    if args.spec_step < 1:
        raise ValueError("spec_step must be >= 1")
    if len(u_sc_values) * len(u_values) * len(bs_values) > 50000:
        print("Warning: large U_sc/U_surface/Bs grid may be slow to compute.")

    er_data = ERData(str(args.er_file))
    pitch_angle = PitchAngle(er_data, str(config.DATA_DIR / config.THETA_FILE))
    fitter = LossConeFitter(
        er_data,
        str(config.DATA_DIR / config.THETA_FILE),
        pitch_angle=pitch_angle,
        normalization_mode=args.normalization,
        incident_flux_stat=args.incident_stat,
    )

    spec_vals = er_data.data[config.SPEC_NO_COLUMN].to_numpy()
    spec_nos = _parse_spec_nos(spec_vals)
    if args.spec_nos:
        requested = [int(x) for x in args.spec_nos.split(",") if x.strip()]
        available = set(spec_nos)
        missing = [sn for sn in requested if sn not in available]
        if missing:
            print(f"Warning: spec numbers not found: {missing}")
        spec_nos = [sn for sn in requested if sn in available]
    if args.spec_step > 1:
        spec_nos = spec_nos[:: args.spec_step]
    if args.spec_limit is not None:
        spec_nos = spec_nos[: args.spec_limit]
    if not spec_nos:
        raise ValueError("No spec numbers selected")

    first_idx_by_spec: dict[int, int] = {}
    target_specs = set(spec_nos)
    for idx, value in enumerate(spec_vals):
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value_f):
            continue
        spec_no = int(value_f)
        if spec_no not in target_specs or spec_no in first_idx_by_spec:
            continue
        first_idx_by_spec[spec_no] = idx
        if len(first_idx_by_spec) == len(target_specs):
            break

    frames: list[dict] = []
    for spec_no in spec_nos:
        if spec_no not in first_idx_by_spec:
            print(f"Skipping spec {spec_no}: no row index found")
            continue
        start = int(first_idx_by_spec[spec_no])
        chunk_idx = int(start // config.SWEEP_ROWS)
        end = min(start + config.SWEEP_ROWS, len(er_data.data))
        if start >= len(er_data.data):
            print(f"Skipping spec {spec_no}: out of range")
            continue

        norm2d = fitter.build_norm2d(chunk_idx)
        norm2d = norm2d[: end - start]
        data = np.where(np.isfinite(norm2d) & (norm2d > 0), norm2d, np.nan)
        if not np.isfinite(data).any():
            print(f"Skipping spec {spec_no}: no finite normalized data")
            continue

        energies = er_data.data[config.ENERGY_COLUMN].to_numpy(dtype=float)[start:end]
        pitches = pitch_angle.pitch_angles[start:end]
        timestamp = er_data.data.iloc[start][config.TIME_COLUMN]

        beam_width = (
            args.beam_width if args.beam_width is not None else fitter.beam_width_ev
        )

        best_bs, best_u_sc, best_chi2 = compute_best_bs_curve(
            energies=energies,
            pitches=pitches,
            data=data,
            u_sc_values=u_sc_values,
            u_values=u_values,
            bs_values=bs_values,
            beam_amp=args.beam_amp,
            beam_width=beam_width,
            background=fitter.background,
            use_torch=use_torch,
        )

        frames.append(
            {
                "spec_no": spec_no,
                "timestamp": timestamp,
                "u_surface": u_values,
                "best_bs": best_bs,
                "best_u_sc": best_u_sc,
                "best_chi2": best_chi2,
            }
        )

    if not frames:
        raise ValueError("No valid spectra available to plot")

    first = frames[0]
    customdata = np.column_stack([first["best_u_sc"], first["best_chi2"]])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=first["u_surface"],
            y=first["best_bs"],
            mode="lines+markers",
            name="Best B_s/B_m",
            customdata=customdata,
            hovertemplate=(
                "U_surface=%{x:.1f} V<br>"
                "Best B_s/B_m=%{y:.3f}<br>"
                "Best U_sc=%{customdata[0]:.1f} V<br>"
                "chi2=%{customdata[1]:.3g}<extra></extra>"
            ),
        )
    )

    fig.update_xaxes(
        title_text="U_surface [V]", range=[u_values.min(), u_values.max()]
    )
    fig.update_yaxes(
        title_text="Best B_s/B_m", range=[bs_values.min(), bs_values.max()]
    )

    fig.update_layout(
        title=dict(
            text=f"Spec {first['spec_no']}: {first['timestamp']}",
            x=0.5,
            xanchor="center",
        ),
        height=600,
        showlegend=False,
    )

    steps = []
    for frame in frames:
        customdata = np.column_stack([frame["best_u_sc"], frame["best_chi2"]])
        steps.append(
            dict(
                method="update",
                args=[
                    {
                        "x": [frame["u_surface"]],
                        "y": [frame["best_bs"]],
                        "customdata": [customdata],
                    },
                    {
                        "title": dict(
                            text=f"Spec {frame['spec_no']}: {frame['timestamp']}",
                            x=0.5,
                            xanchor="center",
                        )
                    },
                ],
                label=str(frame["spec_no"]),
            )
        )

    fig.update_layout(
        sliders=[
            dict(
                active=0,
                yanchor="top",
                y=-0.15,
                xanchor="left",
                currentvalue=dict(prefix="Spec: ", visible=True, xanchor="right"),
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

    if not args.no_serve:
        serve_html(args.output, args.port)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
