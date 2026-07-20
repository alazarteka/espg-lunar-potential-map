"""Unit tests for confidence-set geometry and the D2 profile CI path."""

from __future__ import annotations

import numpy as np
import pytest

from src.losscone.confidence_set import (
    OBSERVATION_LEVEL,
    ConfidenceSetBatch,
    ConfidenceSetComponent,
    GateReason,
    SweepConfidenceSet,
    components_from_retained,
    level_set_from_profile,
)
from src.losscone.profile_ci import (
    ProfileCIConfig,
    fit_profile_confidence_set,
)
from src.losscone.quasi_likelihood import (
    QuasiLikelihoodConfig,
    mean_model_for_params,
    quasi_negloglik,
    simulate_calibrated_flux,
)
from src.losscone.regime_gate import (
    beam_edge_consistency,
    expected_energy_leverage,
    prefit_regime_gate,
)
from src.losscone.response_folded import (
    ResponseFoldedParams,
    build_calibration_mask,
    energy_quadrature,
    response_folded_mean,
)


def test_components_empty_and_full_domain():
    u = np.linspace(-100, 0, 11)
    comps, *_rest = components_from_retained(
        u, np.zeros_like(u, dtype=bool), domain_lo=-100, domain_hi=0
    )
    assert comps == ()

    comps, t_lo, t_hi, is_full, is_one = components_from_retained(
        u, np.ones_like(u, dtype=bool), domain_lo=-100, domain_hi=0
    )
    assert is_full
    assert t_lo and t_hi
    assert not is_one
    assert len(comps) == 1
    assert comps[0].lo == pytest.approx(-100)
    assert comps[0].hi == pytest.approx(0)


def test_components_one_sided_and_disconnected():
    u = np.linspace(-200, 20, 45)
    # One-sided: retained only on the low side touching domain_lo.
    keep = u <= -150
    comps, t_lo, t_hi, is_full, is_one = components_from_retained(
        u, keep, domain_lo=-200.0, domain_hi=20.0
    )
    assert t_lo and not t_hi and not is_full and is_one
    assert len(comps) == 1

    # Disconnected: two islands, neither full-domain.
    keep2 = ((u >= -180) & (u <= -160)) | ((u >= -40) & (u <= -20))
    comps2, t_lo2, t_hi2, is_full2, is_one2 = components_from_retained(
        u, keep2, domain_lo=-200.0, domain_hi=20.0
    )
    assert len(comps2) == 2
    assert not is_full2 and not is_one2
    assert not t_lo2 and not t_hi2


def test_level_set_from_profile_threshold():
    u = np.linspace(-100, 0, 21)
    # Parabola centered at -40.
    lam = ((u + 40.0) / 20.0) ** 2
    comps, *_ = level_set_from_profile(u, lam, c_alpha=1.0, domain_lo=-100, domain_hi=0)
    assert len(comps) == 1
    assert comps[0].lo < -40 < comps[0].hi
    # Interpolated endpoints near +/-20 V from centre.
    assert comps[0].lo == pytest.approx(-60.0, abs=1.0)
    assert comps[0].hi == pytest.approx(-20.0, abs=1.0)


def test_confidence_set_batch_npz_roundtrip(tmp_path):
    sets = [
        SweepConfidenceSet(
            u_hat=-50.0,
            r_hat=0.7,
            beam_amp_hat=0.1,
            nll_min=1.2,
            c_alpha=1.0,
            components=(ConfidenceSetComponent(-70.0, -30.0),),
            domain_lo=-2000.0,
            domain_hi=20.0,
            gate_reason=GateReason.OK,
        ),
        SweepConfidenceSet(
            is_full_domain=True,
            domain_lo=-2000.0,
            domain_hi=20.0,
            gate_reason=GateReason.FULL_DOMAIN,
            components=(ConfidenceSetComponent(-2000.0, 20.0),),
            touches_bound_lo=True,
            touches_bound_hi=True,
        ),
    ]
    batch = ConfidenceSetBatch.from_sets(np.array([1, 2]), sets)
    arrays = batch.to_npz_arrays()
    assert arrays["spec_ci_observation_level"].item() == OBSERVATION_LEVEL
    assert arrays["spec_ci_n_components"].tolist() == [1, 1]
    path = tmp_path / "ci.npz"
    np.savez(path, **arrays)
    loaded = np.load(path, allow_pickle=False)
    assert loaded["spec_ci_u_hat"][0] == pytest.approx(-50.0)


def test_energy_quadrature_weights_sum():
    e = np.array([40.0, 100.0, 400.0])
    energies, weights = energy_quadrature(e, de_over_e=0.5, n_quad=5)
    assert energies.shape == (3, 5)
    assert weights.sum() == pytest.approx(1.0)
    # Centres should be interior to the band.
    assert np.all(energies[:, 0] < e)
    assert np.all(energies[:, -1] > e)


def test_response_folded_mean_shape():
    energies = np.geomspace(40, 2000, 8)
    pitches = np.linspace(0, 180, 16)
    pitch_2d = np.broadcast_to(pitches, (len(energies), len(pitches)))
    params = ResponseFoldedParams(
        u_surface=-80.0, bs_over_bm=0.6, beam_amp=0.2, n_energy_quad=3
    )
    mean = response_folded_mean(energies, pitch_2d, params, u_spacecraft=0.0)
    assert mean.shape == pitch_2d.shape
    assert np.all(np.isfinite(mean))


def test_calibration_mask_excludes_nonfinite_not_lillis_band():
    flux = np.array([[1.0, 0.5, np.nan], [0.0, 0.2, 0.9]])
    mask = build_calibration_mask(flux)
    # Zeros and NaNs out; mid and high values kept (no 0.07-0.79 cut).
    assert mask.tolist() == [[True, True, False], [False, True, True]]


def test_regime_gate_polarity_and_leverage():
    energies = np.geomspace(40, 2000, 10)
    bad = prefit_regime_gate(energies, polarity=0)
    assert not bad.ok
    assert bad.reason == GateReason.POLARITY_UNKNOWN

    flux = np.ones((10, 8))
    ok = prefit_regime_gate(energies, polarity=-1, flux=flux)
    assert ok.ok
    assert ok.energy_leverage > 0
    assert expected_energy_leverage(energies, np.ones(10, dtype=bool)) > 0


def test_beam_edge_consistency():
    ok = beam_edge_consistency(u_edge=-100.0, beam_centroid_eV=100.0, u_spacecraft=0.0)
    assert ok.ok
    bad = beam_edge_consistency(
        u_edge=-100.0, beam_centroid_eV=400.0, u_spacecraft=0.0, tolerance_v=50.0
    )
    assert not bad.ok
    assert bad.reason == GateReason.BEAM_EDGE_INCONSISTENT


def test_quasi_negloglik_minimum_at_truth():
    energies = np.geomspace(40, 2000, 10)
    pitches = np.linspace(0, 180, 20)
    pitch_2d = np.broadcast_to(pitches, (len(energies), len(pitches)))
    cfg = QuasiLikelihoodConfig(n_energy_quad=3, use_response_folding=True)
    truth = mean_model_for_params(
        energies,
        pitch_2d,
        u_surface=-60.0,
        bs_over_bm=0.7,
        beam_amp=0.0,
        u_spacecraft=0.0,
        cfg=cfg,
    )
    mask = build_calibration_mask(truth)
    nll_true = quasi_negloglik(truth, truth, mask, cfg=cfg)
    wrong = mean_model_for_params(
        energies,
        pitch_2d,
        u_surface=-300.0,
        bs_over_bm=0.7,
        beam_amp=0.0,
        u_spacecraft=0.0,
        cfg=cfg,
    )
    nll_wrong = quasi_negloglik(truth, wrong, mask, cfg=cfg)
    assert nll_true < nll_wrong


def test_profile_confidence_set_recovers_truth_ballpark():
    energies = np.geomspace(40, 3000, 12)
    pitches = np.linspace(0, 180, 24)
    pitch_2d = np.broadcast_to(pitches, (len(energies), len(pitches)))
    ql = QuasiLikelihoodConfig(
        n_energy_quad=3, sigma_rel=0.15, sigma_abs=1e-3, use_response_folding=True
    )
    true_u, true_r = -80.0, 0.65
    mean = mean_model_for_params(
        energies,
        pitch_2d,
        u_surface=true_u,
        bs_over_bm=true_r,
        beam_amp=0.0,
        u_spacecraft=0.0,
        cfg=ql,
    )
    mask = build_calibration_mask(mean)
    rng = np.random.default_rng(1)
    data = simulate_calibrated_flux(mean, mask, cfg=ql, rng=rng)

    cfg = ProfileCIConfig(
        coarse_u_count=41,
        n_r_grid=9,
        n_beam_grid=3,
        bootstrap_n=0,
        u_min=-400.0,
        u_max=20.0,
        r_min=0.1,
        r_max=1.0,
        ql=ql,
    )
    result = fit_profile_confidence_set(
        data,
        energies,
        pitch_2d,
        u_spacecraft=0.0,
        polarity=-1,
        cfg=cfg,
        skip_bootstrap=True,
        c_alpha_override=1.0,
    )
    cs = result.confidence_set
    assert cs.gate_reason in (GateReason.OK, GateReason.FULL_DOMAIN)
    assert np.isfinite(cs.u_hat)
    assert abs(cs.u_hat - true_u) < 40.0
    assert cs.components, "expected a nonempty confidence set"
    lo = min(c.lo for c in cs.components)
    hi = max(c.hi for c in cs.components)
    assert lo < hi
    assert lo <= true_u <= hi


def test_profile_rejects_missing_polarity():
    energies = np.geomspace(40, 2000, 8)
    pitches = np.linspace(0, 180, 12)
    pitch_2d = np.broadcast_to(pitches, (len(energies), len(pitches)))
    data = np.ones_like(pitch_2d)
    result = fit_profile_confidence_set(
        data, energies, pitch_2d, polarity=None, skip_bootstrap=True
    )
    assert result.confidence_set.gate_reason == GateReason.POLARITY_UNKNOWN
