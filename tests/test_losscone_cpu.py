"""CPU-only tests for `src.losscone.cpu.LossConeFitter`.

Unlike `tests/test_losscone_parity.py` (which importorskip's torch to compare
the CPU and torch implementations), this module exercises the CPU fitter in
isolation so it still gets coverage on machines without torch installed.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src import config
from src.losscone import LossConeFitter
from src.losscone.model import synth_losscone
from src.utils.synthetic import prepare_synthetic_er


def test_losscone_cpu_fit_chunk_full_recovers_injected_potential(monkeypatch) -> None:
    """The CPU fitter should recover a known, injected U_surface within tolerance."""
    # Keep this test fast; the goal is a smoke/recovery check, not a
    # high-precision convergence benchmark.
    monkeypatch.setattr(config, "LOSS_CONE_LHS_SAMPLES", 128)
    monkeypatch.setattr(config, "LOSS_CONE_DE_POPSIZE", 32)
    monkeypatch.setattr(config, "LOSS_CONE_DE_MAXITER", 80)

    er = prepare_synthetic_er()

    energies = er.data[config.ENERGY_COLUMN].to_numpy(dtype=np.float64)[
        : config.SWEEP_ROWS
    ]
    pitch_1d = np.linspace(0.0, 180.0, config.CHANNELS, dtype=np.float64)
    pitches = np.broadcast_to(pitch_1d[None, :], (config.SWEEP_ROWS, config.CHANNELS))

    true_u_surface = -50.0
    true_bs_over_bm = 0.5
    true_beam_amp = 2.0

    norm2d = synth_losscone(
        energy_grid=energies,
        pitch_grid=pitches,
        U_surface=true_u_surface,
        U_spacecraft=0.0,
        bs_over_bm=true_bs_over_bm,
        beam_width_eV=config.LOSS_CONE_BEAM_WIDTH_EV,
        beam_amp=true_beam_amp,
        beam_pitch_sigma_deg=config.LOSS_CONE_BEAM_PITCH_SIGMA_DEG,
        background=config.LOSS_CONE_BACKGROUND,
    )
    base_flux = np.geomspace(1e3, 1e4, config.SWEEP_ROWS, dtype=np.float64)
    er.data.loc[: config.SWEEP_ROWS - 1, config.FLUX_COLS] = norm2d * base_flux[:, None]

    pitch_angle = SimpleNamespace(pitch_angles=pitches)

    cpu = LossConeFitter(
        er,
        pitch_angle=pitch_angle,
        normalization_mode="ratio",
        fit_method="halekas",
    )

    result = cpu.fit_chunk_full(0)

    # Finite, well-shaped result (a "does it run" invariant).
    assert np.isfinite(result.u_surface)
    assert np.isfinite(result.bs_over_bm)
    assert np.isfinite(result.beam_amp)
    assert np.isfinite(result.chi2)
    assert result.chunk_index == 0

    # Concrete accuracy check: recovers the generating parameters.
    assert abs(result.u_surface - true_u_surface) < 40.0
    assert abs(result.bs_over_bm - true_bs_over_bm) < 0.1
    assert abs(result.beam_amp - true_beam_amp) < 1.0


def test_losscone_cpu_fit_surface_potential_batch_shape(monkeypatch) -> None:
    """fit_surface_potential returns one finite row per measurement chunk."""
    monkeypatch.setattr(config, "LOSS_CONE_LHS_SAMPLES", 64)
    monkeypatch.setattr(config, "LOSS_CONE_DE_POPSIZE", 16)
    monkeypatch.setattr(config, "LOSS_CONE_DE_MAXITER", 40)

    er = prepare_synthetic_er()
    cpu = LossConeFitter(er, normalization_mode="ratio", fit_method="halekas")

    results = cpu.fit_surface_potential()

    n_chunks = len(er.data) // config.SWEEP_ROWS
    assert results.shape == (n_chunks, 5)
    assert np.all(np.isfinite(results))
