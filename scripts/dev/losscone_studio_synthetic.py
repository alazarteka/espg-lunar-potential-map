#!/usr/bin/env python3
"""
Launch Loss Cone Studio with a synthetic ER dataset.

Usage:
  uv run python scripts/dev/losscone_studio_synthetic.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from src import config
from src.flux import ERData, PitchAngle
from src.model import synth_losscone
from src.utils.synthetic import prepare_phis


def _load_losscone_studio():
    module_path = Path(__file__).resolve().parents[1] / "diagnostics" / "losscone_studio.py"
    spec = importlib.util.spec_from_file_location("losscone_studio", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Loss Cone Studio from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _perturb_flux(
    flux: np.ndarray,
    noise_level: float,
    noise_mode: str,
    seed: int,
    background: float,
) -> np.ndarray:
    if noise_level <= 0:
        return np.clip(flux, background, None)

    rng = np.random.default_rng(seed)
    if noise_mode == "log":
        noise = rng.normal(0.0, noise_level, size=flux.shape)
        perturbed = flux * np.exp(noise)
    elif noise_mode == "linear":
        noise = rng.normal(0.0, noise_level, size=flux.shape)
        perturbed = flux + noise
    else:
        raise ValueError(f"Unknown noise_mode: {noise_mode}")

    return np.clip(perturbed, background, None)


def _build_er_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    n_rows = config.SWEEP_ROWS
    energies = np.geomspace(args.energy_min, args.energy_max, n_rows)
    phis, _solid_angles = prepare_phis()

    df = pd.DataFrame(columns=config.ALL_COLS)
    df[config.ENERGY_COLUMN] = energies
    df[config.SPEC_NO_COLUMN] = args.spec_no
    df[config.UTC_COLUMN] = args.utc
    df[config.TIME_COLUMN] = (
        pd.to_datetime(df[config.UTC_COLUMN]).astype("int64") // 10**9
    )

    mag = np.array([1.0, 0.0, 0.0])
    df[config.MAG_COLS] = np.tile(mag, (n_rows, 1))

    phis_row = np.array(phis, dtype=float)
    df[config.PHI_COLS] = np.repeat(phis_row[None, :], n_rows, axis=0)

    df[config.FLUX_COLS] = np.full(
        (n_rows, config.CHANNELS), args.background, dtype=float
    )

    er_temp = ERData.from_dataframe(df.copy(), "synthetic_pitch")
    pitch = PitchAngle(er_temp).pitch_angles

    flux = synth_losscone(
        energy_grid=energies,
        pitch_grid=pitch,
        U_surface=args.u_surface,
        U_spacecraft=args.u_spacecraft,
        bs_over_bm=args.bs_over_bm,
        beam_width_eV=args.beam_width,
        beam_amp=args.beam_amp,
        beam_pitch_sigma_deg=args.beam_pitch_sigma,
        background=args.background,
    )

    flux = _perturb_flux(
        flux=flux,
        noise_level=args.noise_level,
        noise_mode=args.noise_mode,
        seed=args.seed,
        background=args.background,
    )

    df[config.FLUX_COLS] = flux
    return df[config.ALL_COLS]


def _write_er_file(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep=" ", header=False, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch Loss Cone Studio with a synthetic ER dataset."
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        default=Path("scratch/losscone_studio/synthetic_er.tab"),
        help="Output ER file path",
    )
    parser.add_argument("--energy-min", type=float, default=20.0)
    parser.add_argument("--energy-max", type=float, default=20000.0)
    parser.add_argument("--spec-no", type=int, default=1)
    parser.add_argument("--utc", type=str, default="1998-04-01T00:00:00")
    parser.add_argument("--u-surface", type=float, default=-160.0)
    parser.add_argument("--bs-over-bm", type=float, default=0.95)
    parser.add_argument("--u-spacecraft", type=float, default=10.0)
    parser.add_argument("--beam-amp", type=float, default=0.0)
    parser.add_argument("--beam-width", type=float, default=config.LOSS_CONE_BEAM_WIDTH_EV)
    parser.add_argument(
        "--beam-pitch-sigma",
        type=float,
        default=config.LOSS_CONE_BEAM_PITCH_SIGMA_DEG,
    )
    parser.add_argument("--background", type=float, default=config.LOSS_CONE_BACKGROUND)
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument("--noise-mode", choices=("log", "linear"), default="log")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--theta-file",
        type=Path,
        default=config.DATA_DIR / config.THETA_FILE,
        help="Theta file for pitch-angle calculations",
    )
    parser.add_argument(
        "--normalization",
        choices=["global", "ratio", "ratio2", "ratio_rescaled"],
        default="ratio",
    )
    parser.add_argument(
        "--incident-stat",
        choices=["mean", "max"],
        default="mean",
    )
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--n-pitch-bins", type=int, default=100)
    parser.add_argument("--port", type=int, default=5006)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--no-open", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    df = _build_er_dataframe(args)
    _write_er_file(df, args.out_file)

    studio = _load_losscone_studio()
    studio_args = argparse.Namespace(
        er_file=args.out_file,
        theta_file=args.theta_file,
        normalization=args.normalization,
        incident_stat=args.incident_stat,
        background=args.background,
        fast=args.fast,
        chunk=args.chunk,
        u_surface=args.u_surface,
        bs_over_bm=args.bs_over_bm,
        beam_amp=args.beam_amp,
        beam_width=args.beam_width,
        beam_pitch_sigma=args.beam_pitch_sigma,
        u_spacecraft=args.u_spacecraft,
        n_pitch_bins=args.n_pitch_bins,
        port=args.port,
        no_open=args.no_open,
    )

    studio.pn.extension()
    app = studio.build_app(studio_args)
    studio.pn.serve(
        {"losscone_studio": app},
        port=args.port,
        show=not args.no_open,
        websocket_origin=[f"localhost:{args.port}"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
