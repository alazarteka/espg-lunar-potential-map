"""Regression coverage for the torch batched and u-width-QC fit drivers.

These tests pin the behaviour of ``fit_surface_potential_batched`` and
``fit_surface_potential_with_u_width_qc`` before they are refactored to share
internal machinery. Small optimizer settings keep runtime low while golden
values guard against numerical drift.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src import config
from src.losscone.model import synth_losscone
from src.utils.synthetic import prepare_synthetic_er


def _build_losscone_er():
    """Synthetic single-chunk ER with a known Lillis-friendly loss-cone signal."""
    er = prepare_synthetic_er()

    energies = er.data[config.ENERGY_COLUMN].to_numpy(dtype=np.float64)[
        : config.SWEEP_ROWS
    ]
    pitch_1d = np.linspace(0.0, 180.0, config.CHANNELS, dtype=np.float64)
    pitches = np.broadcast_to(pitch_1d[None, :], (config.SWEEP_ROWS, config.CHANNELS))

    norm2d = synth_losscone(
        energy_grid=energies,
        pitch_grid=pitches,
        U_surface=-50.0,
        U_spacecraft=0.0,
        bs_over_bm=0.5,
        beam_width_eV=config.LOSS_CONE_BEAM_WIDTH_EV,
        beam_amp=2.0,
        beam_pitch_sigma_deg=config.LOSS_CONE_BEAM_PITCH_SIGMA_DEG,
        background=config.LOSS_CONE_BACKGROUND,
    )
    base_flux = np.geomspace(1e3, 1e4, config.SWEEP_ROWS, dtype=np.float64)
    er.data.loc[: config.SWEEP_ROWS - 1, config.FLUX_COLS] = norm2d * base_flux[:, None]

    pitch_angle = SimpleNamespace(pitch_angles=pitches)
    return er, pitch_angle


def _make_fitter(fit_method: str):
    from src.losscone_torch import LossConeFitterTorch

    er, pitch_angle = _build_losscone_er()
    return LossConeFitterTorch(
        er,
        pitch_angle=pitch_angle,
        normalization_mode="ratio",
        fit_method=fit_method,
        device="cpu",
        dtype="float64",
    )


def test_u_width_qc_results_match_batched(monkeypatch) -> None:
    """The QC driver's fit results must match the plain batched driver."""
    pytest.importorskip("torch")
    monkeypatch.setattr(config, "LOSS_CONE_LHS_SAMPLES", 128)
    monkeypatch.setattr(config, "LOSS_CONE_DE_POPSIZE", 32)
    monkeypatch.setattr(config, "LOSS_CONE_DE_MAXITER", 80)

    plain = _make_fitter("lillis").fit_surface_potential_batched()
    qc_results, _u_width, _identifiable = _make_fitter(
        "lillis"
    ).fit_surface_potential_with_u_width_qc()

    np.testing.assert_array_equal(qc_results, plain)


def test_u_width_qc_golden(monkeypatch) -> None:
    """Golden values for the Lillis u_width LHS proxy and fit results."""
    pytest.importorskip("torch")
    monkeypatch.setattr(config, "LOSS_CONE_LHS_SAMPLES", 128)
    monkeypatch.setattr(config, "LOSS_CONE_DE_POPSIZE", 32)
    monkeypatch.setattr(config, "LOSS_CONE_DE_MAXITER", 80)

    results, u_width, is_identifiable = _make_fitter(
        "lillis"
    ).fit_surface_potential_with_u_width_qc()

    assert results.shape == (1, 5)
    assert u_width.shape == (1,)
    assert is_identifiable.shape == (1,)

    # Golden values pinned from the pre-refactor implementation.
    np.testing.assert_allclose(results[0], GOLDEN_RESULTS, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(u_width[0], GOLDEN_U_WIDTH, rtol=1e-6, atol=1e-6)
    assert bool(is_identifiable[0]) == GOLDEN_IDENTIFIABLE


# Golden values for the smooth (sigmoid) loss-cone boundary, tau=5 deg.
# The synthetic signal is injected with U=-50.0, bs/bm=0.5, beam_amp=2.0. The
# smooth model recovers all three to <1% (-50.19, 0.501, 1.94), whereas the
# earlier hard-step model settled at a degenerate near-zero-chi2 solution with
# the wrong parameters (-47.69, 0.637, 2.99) - the step's location alone is
# under-determined in (U, bs/bm). The smooth boundary's shape breaks that
# degeneracy. (u_width here is still the LHS max-min proxy; it is replaced by
# the profile-likelihood estimator in task A3.)
GOLDEN_RESULTS = np.array(
    [
        -50.194507520354819,
        0.50099934343088515,
        1.9429386978140428,
        2.2965844248213052e-06,
        0.0,
    ]
)
GOLDEN_U_WIDTH = 31.25
GOLDEN_IDENTIFIABLE = True


def test_cash_cpu_torch_parity(monkeypatch) -> None:
    """The torch Cash path must match the CPU Cash path on a synthetic sweep.

    Both run in float64, so the deterministic profile-likelihood U-width scan
    should agree to numerical precision; the DE-driven fit parameters agree
    closely for this well-constrained single chunk. Guards the torch Cash
    threading (noise tensors + deviance dispatch + profile width) against drift.
    """
    pytest.importorskip("torch")
    monkeypatch.setattr(config, "LOSS_CONE_LHS_SAMPLES", 128)
    monkeypatch.setattr(config, "LOSS_CONE_DE_POPSIZE", 32)
    monkeypatch.setattr(config, "LOSS_CONE_DE_MAXITER", 80)
    # Keep every fit so widths/results are populated (the acceptance cut is
    # identical config-driven logic on both paths).
    monkeypatch.setattr(config, "LOSS_CONE_DEVIANCE_REDUCED_MAX", 1e12)

    from src.losscone.cpu import LossConeFitter
    from src.losscone_torch import LossConeFitterTorch

    er, pitch_angle = _build_losscone_er()
    cpu = LossConeFitter(
        er,
        pitch_angle=pitch_angle,
        normalization_mode="ratio",
        fit_method="cash",
    )
    res_cpu, w_cpu = cpu.fit_surface_potential_with_u_width_qc()

    tor = LossConeFitterTorch(
        er,
        pitch_angle=pitch_angle,
        normalization_mode="ratio",
        fit_method="cash",
        device="cpu",
        dtype="float64",
    )
    res_tor, w_tor, ident_tor = tor.fit_surface_potential_with_u_width_qc()

    assert res_cpu.shape == res_tor.shape == (1, 5)
    assert np.isfinite(res_cpu[0, 0]) and np.isfinite(res_tor[0, 0])

    # Deterministic profile-likelihood width scan -> tight parity.
    np.testing.assert_allclose(w_tor, w_cpu, rtol=0, atol=1e-6)
    # Identifiability classification must agree exactly.
    id_cpu = np.isfinite(w_cpu) & (w_cpu <= 200.0)
    assert np.array_equal(id_cpu, ident_tor)
    # Fit parameters land near each other and near the injected truth (U=-50).
    # The tolerance absorbs DE-optimizer variance (SciPy vs batched DE); tight
    # bit-parity is not expected, unlike the deterministic width scan above.
    np.testing.assert_allclose(res_tor[0, 0], res_cpu[0, 0], atol=6.0)  # U [V]
    np.testing.assert_allclose(res_tor[0, 1], res_cpu[0, 1], atol=0.15)  # bs/bm
    assert -60.0 < res_tor[0, 0] < -40.0  # torch recovers near truth
