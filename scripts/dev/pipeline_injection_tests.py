"""
End-to-end synthetic injection test for the potential-mapping pipeline.

This script:
1) Generates a synthetic loss-cone spectrum with known parameters.
2) Perturbs the spectrum (optionally in log space).
3) Builds an ERData-compatible dataset.
4) Runs the full pipeline (with geometry + SC potential mocked).

Usage:
    uv run python scripts/dev/pipeline_injection_tests.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

import src.spacecraft_potential as spacecraft_potential
from src import config
from src.flux import ERData, PitchAngle
from src.model import synth_losscone
from src.potential_mapper import pipeline
from src.utils.synthetic import prepare_phis
from src.utils.units import ureg


@dataclass(frozen=True)
class InjectionConfig:
    u_surface: float = -160.0
    bs_over_bm: float = 0.95
    u_spacecraft: float = 10.0
    beam_amp: float = 0.0
    noise_level: float = 0.1
    noise_mode: str = "log"
    seed: int = 42
    spec_no: int = 1
    utc: str = "1998-04-01T00:00:00"


def _perturb_flux(
    flux: np.ndarray,
    noise_level: float,
    noise_mode: str,
    seed: int,
    background: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if noise_level <= 0:
        return np.clip(flux, background, 1.0)

    if noise_mode == "log":
        noise = rng.normal(0.0, noise_level, size=flux.shape)
        perturbed = flux * np.exp(noise)
    elif noise_mode == "linear":
        noise = rng.normal(0.0, noise_level, size=flux.shape)
        perturbed = flux + noise
    else:
        raise ValueError(f"Unknown noise_mode: {noise_mode}")

    return np.clip(perturbed, background, 1.0)


def _build_er_dataframe(cfg: InjectionConfig) -> pd.DataFrame:
    n_rows = config.SWEEP_ROWS
    energies = np.geomspace(20.0, 20000.0, n_rows)
    phis, _solid_angles = prepare_phis()

    df = pd.DataFrame(columns=config.ALL_COLS)
    df[config.ENERGY_COLUMN] = energies
    df[config.SPEC_NO_COLUMN] = cfg.spec_no
    df[config.UTC_COLUMN] = cfg.utc
    df[config.TIME_COLUMN] = (
        pd.to_datetime(df[config.UTC_COLUMN]).astype("int64") // 10**9
    )

    mag = np.array([1.0, 0.0, 0.0])
    df[config.MAG_COLS] = np.tile(mag, (n_rows, 1))

    phis_row = np.array(phis, dtype=float)
    df[config.PHI_COLS] = np.repeat(phis_row[None, :], n_rows, axis=0)

    # Placeholder flux; real values will be generated from the forward model.
    df[config.FLUX_COLS] = np.full(
        (n_rows, config.CHANNELS), config.LOSS_CONE_BACKGROUND
    )

    # Use the real pitch angles implied by the ER geometry.
    # Match pipeline polarity (+1 for Moonward along +B in patched geometry).
    er_temp = ERData.from_dataframe(df.copy(), "synthetic_pitch")
    polarity = np.ones(n_rows, dtype=np.int8)
    pitch = PitchAngle(er_temp, polarity=polarity).pitch_angles

    flux = synth_losscone(
        energy_grid=energies,
        pitch_grid=pitch,
        U_surface=cfg.u_surface,
        U_spacecraft=cfg.u_spacecraft,
        bs_over_bm=cfg.bs_over_bm,
        beam_width_eV=config.LOSS_CONE_BEAM_WIDTH_EV,
        beam_amp=cfg.beam_amp,
        beam_pitch_sigma_deg=config.LOSS_CONE_BEAM_PITCH_SIGMA_DEG,
        background=config.LOSS_CONE_BACKGROUND,
    )

    flux = _perturb_flux(
        flux=flux,
        noise_level=cfg.noise_level,
        noise_mode=cfg.noise_mode,
        seed=cfg.seed,
        background=config.LOSS_CONE_BACKGROUND,
    )

    df[config.FLUX_COLS] = flux
    return df


@contextmanager
def _patched_pipeline(u_spacecraft: float):
    original = {}

    def _patch(module, name, value):
        original[(module, name)] = getattr(module, name)
        setattr(module, name, value)

    class _FakeCoordinateCalculator:
        def __init__(self, *_args, **_kwargs):
            pass

        def calculate_coordinate_transformation(self, er_data: ERData):
            n_rows = len(er_data.data)
            radius = config.LUNAR_RADIUS.to(ureg.kilometer).magnitude
            lp_positions = np.tile(np.array([radius, 0.0, 0.0]), (n_rows, 1))
            vec_to_sun = np.tile(np.array([1.0, 0.0, 0.0]), (n_rows, 1))
            return SimpleNamespace(
                lp_positions=lp_positions,
                lp_vectors_to_sun=vec_to_sun,
                moon_vectors_to_sun=vec_to_sun,
            )

    def _fake_project_magnetic_fields(er_data: ERData, _coord_arrays):
        n_rows = len(er_data.data)
        return np.tile(np.array([1.0, 0.0, 0.0]), (n_rows, 1))

    def _fake_find_surface_intersection_with_polarity(
        _coord_arrays, projected_b: np.ndarray
    ):
        n_rows = projected_b.shape[0]
        radius = config.LUNAR_RADIUS.to(ureg.kilometer).magnitude
        points = np.tile(np.array([radius, 0.0, 0.0]), (n_rows, 1))
        mask = np.ones(n_rows, dtype=bool)
        polarity = np.ones(n_rows, dtype=np.int8)
        return points, mask, polarity

    def _fake_get_intersections_or_none_batch(*_args, **kwargs):
        pos = kwargs.get("pos")
        n_rows = pos.shape[0] if pos is not None else 0
        points = np.full((n_rows, 3), np.nan, dtype=float)
        return points, np.zeros(n_rows, dtype=bool)

    def _fake_load_attitude_data(*_args, **_kwargs):
        return np.array([0.0]), np.array([0.0]), np.array([0.0])

    def _fake_calculate_potential(_er_data, _spec_no):
        return object(), u_spacecraft * ureg.volt

    _patch(pipeline, "CoordinateCalculator", _FakeCoordinateCalculator)
    _patch(pipeline, "project_magnetic_fields", _fake_project_magnetic_fields)
    _patch(
        pipeline,
        "find_surface_intersection_with_polarity",
        _fake_find_surface_intersection_with_polarity,
    )
    _patch(
        pipeline,
        "get_intersections_or_none_batch",
        _fake_get_intersections_or_none_batch,
    )
    _patch(pipeline, "load_attitude_data", _fake_load_attitude_data)
    _patch(spacecraft_potential, "calculate_potential", _fake_calculate_potential)

    try:
        yield
    finally:
        for (module, name), value in original.items():
            setattr(module, name, value)


def run_pipeline_injection_test(cfg: InjectionConfig) -> pipeline.PotentialResults:
    df = _build_er_dataframe(cfg)
    er_data = ERData.from_dataframe(df, "synthetic_pipeline")

    with _patched_pipeline(cfg.u_spacecraft):
        results = pipeline.process_merged_data(
            er_data, use_parallel=False, use_torch=False
        )
    return results


def _summarize_results(
    cfg: InjectionConfig, results: pipeline.PotentialResults
) -> None:
    n_total = results.projected_potential.size
    finite_mask = np.isfinite(results.projected_potential)
    n_finite = int(np.count_nonzero(finite_mask))

    if n_finite == 0:
        print("No valid U_surface results (all NaN).")
        return

    u_fit = float(np.nanmedian(results.projected_potential))
    bs_fit = float(np.nanmedian(results.bs_over_bm))
    chi2_fit = float(np.nanmedian(results.fit_chi2))

    print("Pipeline injection test summary")
    print("-" * 60)
    print(f"True U_surface:   {cfg.u_surface:8.2f} V")
    print(f"Fitted U_surface: {u_fit:8.2f} V (median, {n_finite}/{n_total} rows)")
    print(f"True bs_over_bm:  {cfg.bs_over_bm:8.3f}")
    print(f"Fitted bs_over_bm:{bs_fit:8.3f}")
    print(f"Median chi2:      {chi2_fit:8.2f}")


def _parse_args() -> InjectionConfig:
    parser = argparse.ArgumentParser(
        description="Run a synthetic end-to-end pipeline injection test."
    )
    parser.add_argument("--u-surface", type=float, default=-160.0)
    parser.add_argument("--bs-over-bm", type=float, default=0.95)
    parser.add_argument("--u-spacecraft", type=float, default=10.0)
    parser.add_argument("--beam-amp", type=float, default=0.0)
    parser.add_argument("--noise-level", type=float, default=0.1)
    parser.add_argument("--noise-mode", choices=("log", "linear"), default="log")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return InjectionConfig(
        u_surface=args.u_surface,
        bs_over_bm=args.bs_over_bm,
        u_spacecraft=args.u_spacecraft,
        beam_amp=args.beam_amp,
        noise_level=args.noise_level,
        noise_mode=args.noise_mode,
        seed=args.seed,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    cfg = _parse_args()
    results = run_pipeline_injection_test(cfg)
    _summarize_results(cfg, results)


if __name__ == "__main__":
    main()
