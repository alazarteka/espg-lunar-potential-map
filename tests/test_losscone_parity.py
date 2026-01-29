"""CPU/torch parity checks for loss-cone fitting.

These tests intentionally use small optimizer settings to keep runtime low while
ensuring the two implementations stay reasonably aligned.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src import config
from src.losscone import LossConeFitter
from src.losscone.model import synth_losscone
from src.utils.synthetic import prepare_synthetic_er


@pytest.mark.parametrize("fit_method", ["halekas"])
def test_losscone_cpu_torch_fit_chunk_full_parity(monkeypatch, fit_method: str) -> None:
    pytest.importorskip("torch")

    from src.losscone_torch import LossConeFitterTorch

    # Keep this test fast; the goal is parity, not absolute convergence.
    monkeypatch.setattr(config, "LOSS_CONE_LHS_SAMPLES", 128)
    monkeypatch.setattr(config, "LOSS_CONE_DE_POPSIZE", 32)
    monkeypatch.setattr(config, "LOSS_CONE_DE_MAXITER", 80)

    # Synthetic ER container; we'll overwrite flux with a known loss-cone model so
    # build_norm2d() reproduces the forward model exactly under "ratio" mode.
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
        fit_method=fit_method,
    )
    gpu = LossConeFitterTorch(
        er,
        pitch_angle=pitch_angle,
        normalization_mode="ratio",
        fit_method=fit_method,
        device="cpu",
        dtype="float64",
    )

    cpu_result = cpu.fit_chunk_full(0)
    gpu_result = gpu.fit_chunk_full(0)

    assert np.isfinite(cpu_result.u_surface)
    assert np.isfinite(cpu_result.bs_over_bm)
    assert np.isfinite(cpu_result.beam_amp)
    assert np.isfinite(cpu_result.chi2)

    assert np.isfinite(gpu_result.u_surface)
    assert np.isfinite(gpu_result.bs_over_bm)
    assert np.isfinite(gpu_result.beam_amp)
    assert np.isfinite(gpu_result.chi2)

    # Both implementations should converge to similar parameters on the same data.
    assert abs(cpu_result.u_surface - gpu_result.u_surface) < 20.0
    assert abs(cpu_result.bs_over_bm - gpu_result.bs_over_bm) < 0.05
    assert abs(cpu_result.beam_amp - gpu_result.beam_amp) < 0.5

    # Sanity check: both should land near the generating parameters.
    assert abs(cpu_result.u_surface - true_u_surface) < 40.0
    assert abs(cpu_result.bs_over_bm - true_bs_over_bm) < 0.1
    assert abs(cpu_result.beam_amp - true_beam_amp) < 1.0

    assert abs(cpu_result.chi2 - gpu_result.chi2) < 0.1
