#!/usr/bin/env python3
"""
Compare loss-cone fits from the legacy (Halekas) and Lillis-style fitters.

This script samples random chunks across the dataset and writes side-by-side
visualizations: observed normalized flux, model fit, and residual for each
fitter.

Example:
    uv run python scripts/diagnostics/losscone_fit_compare.py \\
        --data-root data \\
        --samples 6 \\
        --seed 42 \\
        --outdir artifacts/losscone_fit_compare

    uv run python scripts/diagnostics/losscone_fit_compare.py \\
        --file data/1998/060_090MAR/3D980323.TAB \\
        --samples 3 \\
        --outdir artifacts/losscone_fit_compare_single
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.flux import ERData, LossConeFitter, PitchAngle
from src.model import synth_losscone


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample random chunks and compare Halekas vs Lillis loss-cone fits."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=config.DATA_DIR,
        help="Root directory to search for ER .TAB files (ignored with --file).",
    )
    parser.add_argument(
        "--file",
        type=Path,
        action="append",
        help="Specific ER .TAB file(s) to sample from (repeatable).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=6,
        help="Number of random samples to render.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for file/chunk selection.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Cap number of files to consider when scanning data-root.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=20,
        help="Max chunk attempts per file before moving on.",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="ratio",
        choices=["global", "ratio", "ratio2", "ratio_rescaled"],
        help="Normalization mode for fitting.",
    )
    parser.add_argument(
        "--incident-flux-stat",
        type=str,
        default="mean",
        choices=["mean", "max"],
        help="Incident flux statistic for normalization.",
    )
    parser.add_argument(
        "--background",
        type=float,
        default=config.LOSS_CONE_BACKGROUND,
        help="Background model level outside loss cone.",
    )
    parser.add_argument(
        "--u-spacecraft",
        type=float,
        default=0.0,
        help="Constant spacecraft potential [V] applied to all rows.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("artifacts/losscone_fit_compare"),
        help="Output directory for rendered figures.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional CSV path for fit parameter summaries.",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=None,
        help="Render a specific chunk index (requires exactly one --file).",
    )
    parser.add_argument(
        "--label-mode",
        type=str,
        choices=["method", "relative"],
        default="method",
        help='Row label style: "method" (Halekas/Lillis) or "relative" (Previous/Current).',
    )
    return parser.parse_args()


def _edge_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        v = values[0]
        return np.array([0.9 * v, 1.1 * v])
    diffs = np.diff(values)
    edges = np.empty(values.size + 1, dtype=float)
    edges[1:-1] = values[:-1] + diffs / 2.0
    edges[0] = values[0] - diffs[0] / 2.0
    edges[-1] = values[-1] + diffs[-1] / 2.0
    return edges


def _finite_range(
    data: np.ndarray, fallback: tuple[float, float], pct: tuple[float, float]
) -> tuple[float, float]:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return fallback
    vmin = float(np.nanpercentile(finite, pct[0]))
    vmax = float(np.nanpercentile(finite, pct[1]))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return fallback
    return vmin, vmax


def _compute_loss_cone_boundary(
    energies: np.ndarray,
    u_surface: float,
    bs_over_bm: float,
    u_spacecraft: float,
) -> np.ndarray:
    e_corr = energies - u_spacecraft
    loss_cone = np.full_like(energies, np.nan, dtype=float)
    valid = e_corr > 0
    if not np.any(valid):
        return loss_cone
    x = bs_over_bm * (1.0 + u_surface / np.maximum(e_corr, config.EPS))
    x = np.clip(x, 0.0, 1.0)
    ac_deg = np.degrees(np.arcsin(np.sqrt(x)))
    loss_cone[valid] = 180.0 - ac_deg[valid]
    return loss_cone


def _chunk_slice(total_rows: int, chunk_idx: int) -> slice:
    start = chunk_idx * config.SWEEP_ROWS
    end = min((chunk_idx + 1) * config.SWEEP_ROWS, total_rows)
    if start >= total_rows:
        raise IndexError(f"Chunk {chunk_idx} out of range for {total_rows} rows.")
    return slice(start, end)


def _collect_files(
    data_root: Path, explicit_files: list[Path] | None, max_files: int | None, rng
) -> list[Path]:
    if explicit_files:
        files = [p for p in explicit_files if p.exists()]
    else:
        files = list(data_root.rglob("*.TAB"))

    if not files:
        raise RuntimeError("No ER .TAB files found.")

    if max_files is not None and len(files) > max_files:
        indices = rng.choice(len(files), size=max_files, replace=False)
        files = [files[int(i)] for i in indices]

    return files


def _build_fitters(
    er: ERData,
    pitch_angle: PitchAngle,
    normalization_mode: str,
    incident_flux_stat: str,
    background: float,
    spacecraft_potential: np.ndarray | None,
) -> tuple[LossConeFitter, LossConeFitter]:
    fitter_halekas = LossConeFitter(
        er,
        pitch_angle=pitch_angle,
        spacecraft_potential=spacecraft_potential,
        normalization_mode=normalization_mode,
        incident_flux_stat=incident_flux_stat,
        loss_cone_background=background,
        fit_method="halekas",
    )
    fitter_lillis = LossConeFitter(
        er,
        pitch_angle=pitch_angle,
        spacecraft_potential=spacecraft_potential,
        normalization_mode=normalization_mode,
        incident_flux_stat=incident_flux_stat,
        loss_cone_background=background,
        fit_method="lillis",
    )
    return fitter_halekas, fitter_lillis


def _render_comparison(
    output_path: Path,
    energies: np.ndarray,
    pitches: np.ndarray,
    norm2d: np.ndarray,
    halekas: dict[str, float | np.ndarray],
    lillis: dict[str, float | np.ndarray],
    title: str,
    row_labels: tuple[str, str],
) -> None:
    vmin, vmax = _finite_range(norm2d, (0.0, 1.0), (5.0, 95.0))
    vmax = max(vmax, 1.0)

    residuals = np.concatenate(
        [
            np.ravel(halekas["residual"]),
            np.ravel(lillis["residual"]),
        ]
    )
    res_max = _finite_range(np.abs(residuals), (0.1, 0.5), (95.0, 99.0))[1]
    res_max = max(res_max, 0.1)

    energy_edges = _edge_array(np.maximum(energies, config.EPS))
    pitch_per_channel = np.nanmedian(pitches, axis=0)
    pitch_edges = _edge_array(pitch_per_channel)

    fig, axes = plt.subplots(2, 3, figsize=(18, 8), constrained_layout=True)

    for row_idx, label, payload in [
        (0, row_labels[0], halekas),
        (1, row_labels[1], lillis),
    ]:
        loss_cone = payload["loss_cone"]
        model = payload["model"]
        residual = payload["residual"]

        obs_ax = axes[row_idx, 0]
        mod_ax = axes[row_idx, 1]
        res_ax = axes[row_idx, 2]

        obs_im = obs_ax.pcolormesh(
            energy_edges,
            pitch_edges,
            norm2d.T,
            shading="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        obs_ax.plot(energies, loss_cone, "w-", linewidth=1.5, alpha=0.9)
        obs_ax.set_xscale("log")
        obs_ax.set_title(f"{label} Observed")
        obs_ax.set_xlabel("Energy [eV]")
        obs_ax.set_ylabel("Pitch [deg]")
        fig.colorbar(obs_im, ax=obs_ax, shrink=0.85)

        mod_im = mod_ax.pcolormesh(
            energy_edges,
            pitch_edges,
            model.T,
            shading="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        mod_ax.plot(energies, loss_cone, "w-", linewidth=1.5, alpha=0.9)
        mod_ax.set_xscale("log")
        mod_ax.set_title(
            f"{label} Model\n"
            f"U={payload['u_surface']:.1f}V, Bs/Bm={payload['bs_over_bm']:.2f}, "
            f"beam={payload['beam_amp']:.2f}"
        )
        mod_ax.set_xlabel("Energy [eV]")
        mod_ax.set_ylabel("Pitch [deg]")
        fig.colorbar(mod_im, ax=mod_ax, shrink=0.85)

        res_im = res_ax.pcolormesh(
            energy_edges,
            pitch_edges,
            residual.T,
            shading="auto",
            cmap="RdBu_r",
            vmin=-res_max,
            vmax=res_max,
        )
        res_ax.plot(energies, loss_cone, "k-", linewidth=1.0, alpha=0.6)
        res_ax.set_xscale("log")
        res_ax.set_title(f"{label} Residual\nchi2={payload['chi2']:.3g}")
        res_ax.set_xlabel("Energy [eV]")
        res_ax.set_ylabel("Pitch [deg]")
        fig.colorbar(res_im, ax=res_ax, shrink=0.85)

    fig.suptitle(title, fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)

    if args.chunk is not None:
        if not args.file or len(args.file) != 1:
            raise RuntimeError("--chunk requires exactly one --file entry.")

    files = _collect_files(args.data_root, args.file, args.max_files, rng)

    summaries: list[dict[str, object]] = []
    samples_done = 0
    file_queue = list(files)
    rng.shuffle(file_queue)

    if args.label_mode == "relative":
        row_labels = ("Previous", "Current")
    else:
        row_labels = ("Halekas", "Lillis")

    while samples_done < args.samples and file_queue:
        file_path = file_queue.pop()
        er = ERData(str(file_path))
        if er.data.empty:
            continue

        total_chunks = len(er.data) // config.SWEEP_ROWS
        if total_chunks == 0:
            continue

        spacecraft_potential = None
        if args.u_spacecraft != 0.0:
            spacecraft_potential = np.full(len(er.data), args.u_spacecraft)

        pitch_angle = PitchAngle(er)
        fitter_halekas, fitter_lillis = _build_fitters(
            er,
            pitch_angle,
            args.normalization,
            args.incident_flux_stat,
            args.background,
            spacecraft_potential,
        )

        for _ in range(args.max_attempts):
            if args.chunk is not None:
                chunk_idx = int(args.chunk)
            else:
                chunk_idx = int(rng.integers(0, total_chunks))
            norm2d = fitter_halekas.build_norm2d(chunk_idx)
            if np.isnan(norm2d).all():
                if args.chunk is not None:
                    break
                continue

            halekas_params = fitter_halekas._fit_surface_potential(chunk_idx)
            lillis_params = fitter_lillis._fit_surface_potential(chunk_idx)
            if not np.all(np.isfinite(halekas_params)) or not np.all(
                np.isfinite(lillis_params)
            ):
                if args.chunk is not None:
                    break
                continue

            chunk_slice = _chunk_slice(len(er.data), chunk_idx)
            energies = er.data[config.ENERGY_COLUMN].to_numpy(dtype=np.float64)[
                chunk_slice
            ]
            pitches = pitch_angle.pitch_angles[chunk_slice]
            spec_no = int(er.data.iloc[chunk_slice.start][config.SPEC_NO_COLUMN])
            timestamp = str(er.data.iloc[chunk_slice.start][config.TIME_COLUMN])

            hal_u, hal_bs, hal_beam, hal_chi2 = halekas_params
            lil_u, lil_bs, lil_beam, lil_chi2 = lillis_params

            hal_model = synth_losscone(
                energy_grid=energies,
                pitch_grid=pitches,
                U_surface=hal_u,
                U_spacecraft=float(args.u_spacecraft),
                bs_over_bm=hal_bs,
                beam_width_eV=fitter_halekas.beam_width_ev,
                beam_amp=hal_beam,
                beam_pitch_sigma_deg=fitter_halekas.beam_pitch_sigma_deg,
                background=fitter_halekas.background,
            )
            lil_model = synth_losscone(
                energy_grid=energies,
                pitch_grid=pitches,
                U_surface=lil_u,
                U_spacecraft=float(args.u_spacecraft),
                bs_over_bm=lil_bs,
                beam_width_eV=fitter_lillis.beam_width_ev,
                beam_amp=lil_beam,
                beam_pitch_sigma_deg=fitter_lillis.beam_pitch_sigma_deg,
                background=fitter_lillis.background,
            )

            hal_loss = _compute_loss_cone_boundary(
                energies, hal_u, hal_bs, float(args.u_spacecraft)
            )
            lil_loss = _compute_loss_cone_boundary(
                energies, lil_u, lil_bs, float(args.u_spacecraft)
            )

            output_path = (
                args.outdir / f"{file_path.stem}_chunk{chunk_idx:04d}.png"
            )
            title = (
                f"{file_path.name} | chunk {chunk_idx} | "
                f"spec {spec_no} | {timestamp}"
            )

            _render_comparison(
                output_path,
                energies,
                pitches,
                norm2d,
                {
                    "u_surface": hal_u,
                    "bs_over_bm": hal_bs,
                    "beam_amp": hal_beam,
                    "chi2": hal_chi2,
                    "model": hal_model,
                    "residual": norm2d - hal_model,
                    "loss_cone": hal_loss,
                },
                {
                    "u_surface": lil_u,
                    "bs_over_bm": lil_bs,
                    "beam_amp": lil_beam,
                    "chi2": lil_chi2,
                    "model": lil_model,
                    "residual": norm2d - lil_model,
                    "loss_cone": lil_loss,
                },
                title,
                row_labels,
            )

            summaries.append(
                {
                    "file": file_path.name,
                    "chunk": chunk_idx,
                    "spec_no": spec_no,
                    "timestamp": timestamp,
                    "halekas_u_surface": hal_u,
                    "halekas_bs_over_bm": hal_bs,
                    "halekas_beam_amp": hal_beam,
                    "halekas_chi2": hal_chi2,
                    "lillis_u_surface": lil_u,
                    "lillis_bs_over_bm": lil_bs,
                    "lillis_beam_amp": lil_beam,
                    "lillis_chi2": lil_chi2,
                }
            )

            samples_done += 1
            print(f"Saved {output_path}")
            break

        if args.chunk is not None:
            break

    if args.summary_csv and summaries:
        import csv

        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
        print(f"Wrote summary CSV to {args.summary_csv}")

    if args.chunk is None and samples_done < args.samples:
        print(f"Generated {samples_done}/{args.samples} samples.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
