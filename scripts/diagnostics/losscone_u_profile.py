#!/usr/bin/env python3
"""
Compute a 1D "identifiability" profile for U_surface by profiling over Bs/Bm + beam_amp.

This is a diagnostic tool: for each spectrum, we evaluate a grid of candidate U_surface
values and, for each U, minimize the Lillis-style chi2 over Bs/Bm and beam amplitude.
The resulting chi2(U) curve is a loose analogue of a confidence profile: broad / flat
profiles indicate weak constraints on U_surface.

Notes
-----
- Currently supports `fit_method=lillis` only.
- Beam amplitude is minimized analytically for each (U, Bs/Bm) point since the model is
  linear in beam_amp when beam width is fixed.
- The analysis expects a batch output NPZ from `src.potential_mapper.batch` so we can
  reuse the row-level polarity array without re-running SPICE.

Examples
--------
Compute profile metrics + plot for Apr 29, 1999 (U_sc forced to 0 in the batch run):

  uv run python scripts/diagnostics/losscone_u_profile.py \\
    --year 1999 --month 4 --day 29 \\
    --batch-npz artifacts/potential_cache/1999_04_29_u0_lillis/potential_batch_1999_04_29.npz \\
    --fit-method lillis \\
    --u-spacecraft 0 \\
    --u-count 61 --bs-count 41 \\
    --delta-reduced 0.001 --delta-reduced 0.002 --delta-reduced 0.005
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src import config
from src.losscone.cpu import LossConeFitter, PitchAngle
from src.losscone.masks import build_lillis_mask
from src.losscone.model import _compute_beam, _compute_loss_cone_angle
from src.losscone.types import FitMethod, parse_fit_method
from src.potential_mapper.pipeline import DataLoader, load_all_data


@dataclass(frozen=True)
class ProfileGrids:
    u_grid: np.ndarray  # (n_u,)
    bs_grid: np.ndarray  # (n_bs,)


def _parse_utc(ts: str) -> datetime:
    ts = str(ts).strip()
    if not ts:
        return datetime.fromtimestamp(0, tz=UTC).replace(tzinfo=None)
    try:
        return datetime.fromtimestamp(float(ts), tz=UTC).replace(tzinfo=None)
    except ValueError:
        pass

    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    dt = dt.astimezone(UTC)
    return dt.replace(tzinfo=None)


def _beam_template(
    *,
    u_surface_grid: np.ndarray,  # (n_u,)
    energies: np.ndarray,  # (nE,)
    pitches: np.ndarray,  # (nE, nPitch)
    u_spacecraft: float,
    beam_width_ev: float,
    beam_pitch_sigma_deg: float,
) -> np.ndarray:
    """Beam template for beam_amp=1.0, shape (n_u, nE, nPitch)."""
    u_surface_grid = np.asarray(u_surface_grid, dtype=np.float64)
    n_u = u_surface_grid.size
    return _compute_beam(
        energies[None, :, None],
        pitches[None, :, :],
        u_surface_grid.reshape(n_u, 1, 1),
        np.asarray(u_spacecraft, dtype=np.float64).reshape(1, 1, 1),
        np.ones((n_u, 1, 1), dtype=np.float64),
        np.full((n_u, 1, 1), float(beam_width_ev), dtype=np.float64),
        float(beam_pitch_sigma_deg),
    )


def _profile_lillis_u(
    *,
    norm2d: np.ndarray,  # (nE, nPitch)
    energies: np.ndarray,  # (nE,)
    pitches: np.ndarray,  # (nE, nPitch)
    mask: np.ndarray,  # (nE, nPitch)
    grids: ProfileGrids,
    u_spacecraft: float,
    background: float,
    beam_width_ev: float,
    beam_pitch_sigma_deg: float,
    beam_amp_min: float,
    beam_amp_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (chi2_profile_raw, bs_best, beam_amp_best), each shaped (n_u,).
    """
    if not mask.any():
        n_u = int(grids.u_grid.size)
        nan = np.full(n_u, np.nan, dtype=np.float64)
        return nan, nan, nan

    beam_u = _beam_template(
        u_surface_grid=grids.u_grid,
        energies=energies,
        pitches=pitches,
        u_spacecraft=u_spacecraft,
        beam_width_ev=beam_width_ev,
        beam_pitch_sigma_deg=beam_pitch_sigma_deg,
    )  # (n_u, nE, nPitch)

    # IMPORTANT: avoid `x * mask` with NaNs present; 0 * NaN => NaN.
    mask_u = mask[None, None, :, :]  # (1, 1, nE, nPitch)
    denom = np.sum(np.where(mask[None, :, :], beam_u * beam_u, 0.0), axis=(1, 2))

    # Loss-cone base model for all (U, bs) combinations via library angle formula.
    n_u = int(grids.u_grid.size)
    n_bs = int(grids.bs_grid.size)
    u_flat = np.repeat(grids.u_grid, n_bs)
    bs_flat = np.tile(grids.bs_grid, n_u)
    ac_deg = _compute_loss_cone_angle(
        energies[None, :, None],
        u_flat.reshape(-1, 1, 1),
        np.asarray(u_spacecraft, dtype=np.float64).reshape(1, 1, 1),
        bs_flat.reshape(-1, 1, 1),
    ).reshape(n_u, n_bs, energies.size, 1)

    inside = pitches[None, None, :, :] <= (180.0 - ac_deg)
    base = float(background) + (1.0 - float(background)) * inside.astype(np.float64)

    # Analytical beam_amp*(U, bs) via least squares projection (with bounds).
    residual = norm2d[None, None, :, :] - base  # (n_u, n_bs, nE, nPitch)
    beam = beam_u[:, None, :, :]  # (n_u, 1, nE, nPitch)

    numer = np.sum(np.where(mask_u, residual * beam, 0.0), axis=(2, 3))
    denom_safe = np.where(denom > 0.0, denom, np.nan)  # (n_u,)
    beam_amp = numer / denom_safe[:, None]  # (n_u, n_bs)
    beam_amp = np.clip(beam_amp, float(beam_amp_min), float(beam_amp_max))
    beam_amp = np.where(np.isfinite(beam_amp), beam_amp, 0.0)

    model = base + beam_amp[:, :, None, None] * beam  # (n_u, n_bs, nE, nPitch)
    diff = norm2d[None, None, :, :] - model
    chi2_raw = np.sum(np.where(mask_u, diff * diff, 0.0), axis=(2, 3))  # (n_u, n_bs)

    finite = np.isfinite(chi2_raw)
    any_finite = finite.any(axis=1)
    chi2_safe = np.where(finite, chi2_raw, np.inf)
    best_idx = np.argmin(chi2_safe, axis=1)  # (n_u,)

    best_chi2 = chi2_safe[np.arange(best_idx.size), best_idx].astype(np.float64)
    best_bs = grids.bs_grid[best_idx].astype(np.float64)
    best_beam = beam_amp[np.arange(best_idx.size), best_idx].astype(np.float64)

    best_chi2[~any_finite] = np.nan
    best_bs[~any_finite] = np.nan
    best_beam[~any_finite] = np.nan
    return (
        best_chi2,
        best_bs,
        best_beam,
    )


def _compute_width(
    u_grid: np.ndarray, chi2_reduced: np.ndarray, delta_reduced: float
) -> float:
    finite = np.isfinite(chi2_reduced)
    if not finite.any():
        return float("nan")
    chi2_min = float(np.nanmin(chi2_reduced))
    keep = chi2_reduced <= (chi2_min + float(delta_reduced))
    if not np.any(keep):
        return 0.0
    return float(np.max(u_grid[keep]) - np.min(u_grid[keep]))


def _infer_grids(
    *,
    u_min: float,
    u_max: float,
    u_count: int,
    bs_min: float,
    bs_max: float,
    bs_count: int,
) -> ProfileGrids:
    if u_count < 3:
        raise ValueError("--u-count must be >= 3")
    if bs_count < 3:
        raise ValueError("--bs-count must be >= 3")
    u_grid = np.linspace(float(u_min), float(u_max), int(u_count), dtype=np.float64)
    bs_grid = np.linspace(float(bs_min), float(bs_max), int(bs_count), dtype=np.float64)
    return ProfileGrids(u_grid=u_grid, bs_grid=bs_grid)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile chi2(U_surface) over Bs/Bm + beam_amp (Lillis only)."
    )

    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--er-file", type=Path, help="Path to ER .TAB file")
    src_group.add_argument("--year", type=int, help="Year (for DataLoader discovery)")

    parser.add_argument("--month", type=int, help="Month (1-12)")
    parser.add_argument("--day", type=int, help="Day (1-31)")

    parser.add_argument(
        "--batch-npz",
        type=Path,
        required=True,
        help="Batch output NPZ from src.potential_mapper.batch (provides polarity + time)",
    )
    parser.add_argument(
        "--fit-method",
        choices=["lillis"],
        default="lillis",
        help="Fit method (only lillis currently supported)",
    )
    parser.add_argument(
        "--normalization",
        choices=["ratio", "ratio2"],
        default="ratio",
        help="Loss-cone normalization mode",
    )
    parser.add_argument(
        "--incident-stat",
        choices=["mean", "max"],
        default="mean",
        help="Incident flux statistic (ratio mode)",
    )
    parser.add_argument(
        "--u-spacecraft",
        type=float,
        default=0.0,
        help="Spacecraft potential override [V] (used for model + valid-energy mask)",
    )

    parser.add_argument(
        "--u-min",
        type=float,
        default=config.LOSS_CONE_U_SURFACE_MIN,
        help="U_surface grid minimum [V]",
    )
    parser.add_argument(
        "--u-max",
        type=float,
        default=config.LOSS_CONE_U_SURFACE_MAX,
        help="U_surface grid maximum [V]",
    )
    parser.add_argument(
        "--u-count",
        type=int,
        default=61,
        help="Number of U_surface grid points",
    )
    parser.add_argument(
        "--bs-min",
        type=float,
        default=config.LOSS_CONE_BS_OVER_BM_MIN,
        help="Bs/Bm grid minimum",
    )
    parser.add_argument(
        "--bs-max",
        type=float,
        default=config.LOSS_CONE_BS_OVER_BM_MAX,
        help="Bs/Bm grid maximum",
    )
    parser.add_argument(
        "--bs-count",
        type=int,
        default=41,
        help="Number of Bs/Bm grid points",
    )
    parser.add_argument(
        "--delta-reduced",
        type=float,
        action="append",
        default=[0.001, 0.002, 0.005],
        help="Reduced-chi2 thresholds (repeatable) for U-width metrics",
    )

    parser.add_argument(
        "--output-npz",
        type=Path,
        default=None,
        help="Output NPZ path (defaults under artifacts/potential_cache/daily)",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Output plot path (defaults under artifacts/potential_cache/plots)",
    )
    parser.add_argument(
        "--augment-batch-npz",
        type=Path,
        default=None,
        help=(
            "Optional: write a copy of the input batch NPZ augmented with "
            "spec/row U-width + identifiability QC fields"
        ),
    )
    parser.add_argument(
        "--augment-delta-reduced",
        type=float,
        default=0.001,
        help="Delta reduced-chi2 used for U-width QC in the augmented batch NPZ",
    )
    parser.add_argument(
        "--augment-width-max",
        type=float,
        default=200.0,
        help="Max U-width [V] considered identifiable in augmented batch QC",
    )
    parser.add_argument(
        "--augment-no-rows",
        action="store_true",
        help="Do not add row-level U-width/QC arrays to the augmented batch NPZ",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite outputs")

    return parser.parse_args()


def _resolve_er_files(args: argparse.Namespace) -> list[Path]:
    if args.er_file is not None:
        return [Path(args.er_file)]
    if args.month is None or args.day is None:
        raise ValueError("--month and --day are required with --year")
    return DataLoader.discover_flux_files(
        year=int(args.year), month=int(args.month), day=int(args.day)
    )


def main() -> int:
    args = parse_args()

    fit_method = parse_fit_method(args.fit_method)
    if fit_method != FitMethod.LILLIS:
        raise ValueError("Only fit_method='lillis' is supported in this script.")

    batch_npz = np.load(args.batch_npz, allow_pickle=False)
    rows_polarity = batch_npz["rows_projection_polarity"].astype(np.int8, copy=False)
    spec_nos_npz = batch_npz["spec_spec_no"].astype(np.int64, copy=False)
    spec_time_npz = batch_npz["spec_time_start"].astype("U64", copy=False)
    spec_u_fit_npz = batch_npz["spec_u_surface"].astype(np.float64, copy=False)
    spec_bs_fit_npz = batch_npz["spec_bs_over_bm"].astype(np.float64, copy=False)
    spec_chi2_fit_npz = batch_npz["spec_fit_chi2"].astype(np.float64, copy=False)
    rows_beam_amp = batch_npz["rows_beam_amp"].astype(np.float64, copy=False)

    spec_idx_by_no: dict[int, int] = {
        int(spec_no): int(i) for i, spec_no in enumerate(spec_nos_npz)
    }

    er_files = _resolve_er_files(args)
    if not er_files:
        raise FileNotFoundError("No ER files found for the requested date.")

    er_data = load_all_data([Path(p) for p in er_files])
    if er_data.data.empty:
        raise RuntimeError("Merged ERData is empty.")

    n_rows = len(er_data.data)
    if n_rows != rows_polarity.shape[0]:
        raise ValueError(
            f"rows_projection_polarity has {rows_polarity.shape[0]} rows but ERData has {n_rows}"
        )

    pitch_angle = PitchAngle(er_data, polarity=rows_polarity)

    spacecraft_potential = np.full(n_rows, float(args.u_spacecraft), dtype=np.float64)
    fitter = LossConeFitter(
        er_data,
        pitch_angle=pitch_angle,
        spacecraft_potential=spacecraft_potential,
        normalization_mode=str(args.normalization),
        incident_flux_stat=str(args.incident_stat),
        fit_method=fit_method,
    )

    nE = int(config.SWEEP_ROWS)
    nP = int(config.CHANNELS)
    n_chunks = n_rows // nE

    flux_all = er_data.data[config.FLUX_COLS].to_numpy(dtype=np.float64, copy=False)
    raw_flux = flux_all[: n_chunks * nE].reshape(n_chunks, nE, nP)
    energies = (
        er_data.data[config.ENERGY_COLUMN]
        .to_numpy(dtype=np.float64, copy=False)[: n_chunks * nE]
        .reshape(n_chunks, nE)
    )
    pitches = pitch_angle.pitch_angles[: n_chunks * nE].reshape(n_chunks, nE, nP)

    norm2d = fitter.build_norm2d_batch(list(range(n_chunks)))
    if norm2d.shape != (n_chunks, nE, nP):
        raise RuntimeError(f"Unexpected norm2d shape {norm2d.shape}")

    data_mask = build_lillis_mask(raw_flux, pitches)
    valid_energy = energies >= float(args.u_spacecraft)
    valid_energy_mask = np.broadcast_to(valid_energy[:, :, None], (n_chunks, nE, nP))
    combined_mask = data_mask & valid_energy_mask
    n_valid = np.count_nonzero(combined_mask, axis=(1, 2)).astype(np.int64)
    dof = np.maximum(n_valid - 3, 1).astype(np.int64)

    grids = _infer_grids(
        u_min=float(args.u_min),
        u_max=float(args.u_max),
        u_count=int(args.u_count),
        bs_min=float(args.bs_min),
        bs_max=float(args.bs_max),
        bs_count=int(args.bs_count),
    )

    profile_chi2_red = np.full((n_chunks, grids.u_grid.size), np.nan, dtype=np.float64)
    profile_bs_best = np.full_like(profile_chi2_red, np.nan)
    profile_beam_best = np.full_like(profile_chi2_red, np.nan)
    profile_u_best = np.full(n_chunks, np.nan, dtype=np.float64)
    profile_chi2_min_red = np.full(n_chunks, np.nan, dtype=np.float64)

    widths: dict[float, np.ndarray] = {
        float(delta): np.full(n_chunks, np.nan, dtype=np.float64)
        for delta in (float(x) for x in args.delta_reduced)
    }

    # Align NPZ fit outputs to chunk order via spec_no.
    spec_no_rows = er_data.data[config.SPEC_NO_COLUMN].to_numpy(
        dtype=np.int64, copy=False
    )
    spec_no_chunk = spec_no_rows[: n_chunks * nE : nE].astype(np.int64, copy=False)
    time_chunk = np.full(n_chunks, "", dtype="U64")
    u_fit = np.full(n_chunks, np.nan, dtype=np.float64)
    bs_fit = np.full(n_chunks, np.nan, dtype=np.float64)
    chi2_fit = np.full(n_chunks, np.nan, dtype=np.float64)
    beam_fit = rows_beam_amp[: n_chunks * nE : nE].astype(np.float64, copy=False)

    for i, spec_no in enumerate(spec_no_chunk):
        j = spec_idx_by_no.get(int(spec_no))
        if j is None:
            continue
        time_chunk[i] = spec_time_npz[j]
        u_fit[i] = spec_u_fit_npz[j]
        bs_fit[i] = spec_bs_fit_npz[j]
        chi2_fit[i] = spec_chi2_fit_npz[j]

    background = float(fitter.background)
    beam_width_ev = float(fitter.beam_width_ev)
    beam_pitch_sigma = float(fitter.beam_pitch_sigma_deg)
    beam_amp_min = float(fitter.beam_amp_min)
    beam_amp_max = float(fitter.beam_amp_max)

    for idx in tqdm(range(n_chunks), desc="Profiling spectra", unit="spec"):
        if n_valid[idx] < int(config.LILLIS_MIN_VALID_BINS):
            continue

        chi2_raw_u, bs_best_u, beam_best_u = _profile_lillis_u(
            norm2d=norm2d[idx],
            energies=energies[idx],
            pitches=pitches[idx],
            mask=combined_mask[idx],
            grids=grids,
            u_spacecraft=float(args.u_spacecraft),
            background=background,
            beam_width_ev=beam_width_ev,
            beam_pitch_sigma_deg=beam_pitch_sigma,
            beam_amp_min=beam_amp_min,
            beam_amp_max=beam_amp_max,
        )
        red = chi2_raw_u / float(dof[idx])
        profile_chi2_red[idx] = red
        profile_bs_best[idx] = bs_best_u
        profile_beam_best[idx] = beam_best_u

        finite = np.isfinite(red)
        if finite.any():
            j_min = int(np.nanargmin(red))
            profile_u_best[idx] = float(grids.u_grid[j_min])
            profile_chi2_min_red[idx] = float(red[j_min])
            for delta, out in widths.items():
                out[idx] = _compute_width(grids.u_grid, red, float(delta))

    # Resolve output paths.
    if args.output_npz is None:
        stamp = (
            f"{args.year:04d}_{args.month:02d}_{args.day:02d}"
            if args.year is not None
            else "erfile"
        )
        out_npz = (
            Path("artifacts/potential_cache/daily")
            / f"{stamp}_u_profile_{args.fit_method}_u_sc_{float(args.u_spacecraft):g}.npz"
        )
    else:
        out_npz = Path(args.output_npz)

    if out_npz.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {out_npz} (use --overwrite)")

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        spec_spec_no=spec_no_chunk,
        spec_time_start=time_chunk,
        spec_u_fit=u_fit,
        spec_bs_fit=bs_fit,
        spec_beam_fit=beam_fit,
        spec_chi2_fit=chi2_fit,
        spec_n_valid=n_valid,
        spec_dof=dof,
        u_grid=grids.u_grid,
        bs_grid=grids.bs_grid,
        profile_chi2_reduced=profile_chi2_red,
        profile_bs_best=profile_bs_best,
        profile_beam_best=profile_beam_best,
        profile_u_best=profile_u_best,
        profile_chi2_min_reduced=profile_chi2_min_red,
        **{
            f"u_width_dchi2red_{str(delta).replace('.', 'p')}": arr
            for delta, arr in widths.items()
        },
    )

    if args.plot is None:
        stamp = (
            f"{args.year:04d}_{args.month:02d}_{args.day:02d}"
            if args.year is not None
            else "erfile"
        )
        out_plot = (
            Path("artifacts/potential_cache/plots")
            / f"{stamp}_u_profile_{args.fit_method}_u_sc_{float(args.u_spacecraft):g}.png"
        )
    else:
        out_plot = Path(args.plot)

    out_plot.parent.mkdir(parents=True, exist_ok=True)

    # Plot: U_fit vs time (colored by width) + width vs time.
    plot_written = False
    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt

        times = np.array(
            [_parse_utc(t) if str(t).strip() else None for t in time_chunk],
            dtype=object,
        )
        has_time = np.array([t is not None for t in times], dtype=bool)
        delta_for_plot = float(sorted(widths.keys())[0])
        width_plot = widths[delta_for_plot]

        ok = has_time & np.isfinite(u_fit) & np.isfinite(width_plot)
        if ok.any():
            fig, (ax_u, ax_w) = plt.subplots(
                2, 1, figsize=(14, 8), sharex=True, constrained_layout=True
            )

            c = np.log10(np.clip(width_plot[ok], 1.0, None))
            sc = ax_u.scatter(
                times[ok],
                u_fit[ok],
                c=c,
                s=9,
                cmap="viridis",
                alpha=0.8,
                linewidths=0.0,
            )
            ax_u.set_ylabel("U_surface fit [V]")
            ax_u.grid(True, alpha=0.2)
            cb = fig.colorbar(sc, ax=ax_u, pad=0.01)
            cb.set_label(f"log10 width [V] (Δχ²_red≤{delta_for_plot:g})")

            ok_w = has_time & np.isfinite(width_plot) & (width_plot > 0)
            ax_w.scatter(times[ok_w], width_plot[ok_w], s=9, alpha=0.8, linewidths=0.0)
            ax_w.set_ylabel(f"U width [V] (Δχ²_red≤{delta_for_plot:g})")
            ax_w.set_yscale("log")
            ax_w.grid(True, alpha=0.2)
            ax_w.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax_w.set_xlabel("UTC")

            fig.suptitle(
                f"U identifiability profile ({args.fit_method}, U_sc={float(args.u_spacecraft):g} V)"
            )
            fig.savefig(out_plot, dpi=200)
            plt.close(fig)
            plot_written = True
    except Exception as exc:
        plot_written = False
        print(f"Plot generation failed: {exc}")

    print(f"Saved profile cache: {out_npz}")
    if plot_written and out_plot.exists():
        print(f"Saved plot: {out_plot}")
    else:
        print("Plot was not generated (matplotlib error or no finite points).")

    if args.augment_batch_npz is not None:
        from src.potential_mapper.u_profile_qc import augment_batch_npz_with_u_width

        augment_batch_npz_with_u_width(
            batch_npz_path=Path(args.batch_npz),
            profile_npz_path=out_npz,
            out_npz_path=Path(args.augment_batch_npz),
            delta_reduced=float(args.augment_delta_reduced),
            identifiable_width_max_v=float(args.augment_width_max),
            include_rows=not bool(args.augment_no_rows),
            overwrite=bool(args.overwrite),
        )
        print(f"Saved augmented batch NPZ: {Path(args.augment_batch_npz)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
