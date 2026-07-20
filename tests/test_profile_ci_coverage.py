"""Synthetic coverage smoke for the profile-likelihood confidence set.

Under the stated calibrated-flux generative model:
- the MLE should recover truth within a few volts;
- a generous profile threshold should cover truth often;
- asymptotic chi2_1 (c_alpha=1) is known to be overconfident for this
  quasi-likelihood on coarse grids -- production code must use the
  parametric bootstrap (see profile_ci.bootstrap_c_alpha).
"""

from __future__ import annotations

import numpy as np

from src.losscone.confidence_set import GateReason
from src.losscone.profile_ci import ProfileCIConfig, fit_profile_confidence_set
from src.losscone.quasi_likelihood import (
    QuasiLikelihoodConfig,
    mean_model_for_params,
    simulate_calibrated_flux,
)
from src.losscone.response_folded import build_calibration_mask


def _covers(cs, true_u: float) -> bool:
    if cs.is_full_domain:
        return True
    if not cs.components:
        return False
    return any(c.lo <= true_u <= c.hi for c in cs.components)


def test_mle_recovers_truth_under_stated_model():
    energies = np.geomspace(40, 3000, 10)
    pitches = np.linspace(0, 180, 20)
    pitch_2d = np.broadcast_to(pitches, (len(energies), len(pitches)))
    ql = QuasiLikelihoodConfig(
        n_energy_quad=3, sigma_rel=0.2, sigma_abs=1e-3, use_response_folding=True
    )
    true_u, true_r = -70.0, 0.7
    mean = mean_model_for_params(
        energies,
        pitch_2d,
        u_surface=true_u,
        bs_over_bm=true_r,
        beam_amp=0.0,
        cfg=ql,
    )
    mask = build_calibration_mask(mean)
    cfg = ProfileCIConfig(
        coarse_u_count=31,
        n_r_grid=7,
        n_beam_grid=3,
        bootstrap_n=0,
        u_min=-300.0,
        u_max=20.0,
        r_min=0.2,
        r_max=1.0,
        ql=ql,
    )
    rng = np.random.default_rng(42)
    errs = []
    for _ in range(8):
        data = simulate_calibrated_flux(mean, mask, cfg=ql, rng=rng)
        result = fit_profile_confidence_set(
            data,
            energies,
            pitch_2d,
            polarity=-1,
            cfg=cfg,
            skip_bootstrap=True,
            c_alpha_override=1.0,
        )
        cs = result.confidence_set
        assert cs.gate_reason in (GateReason.OK, GateReason.FULL_DOMAIN)
        errs.append(cs.u_hat - true_u)
    assert float(np.mean(np.abs(errs))) < 5.0


def test_generous_threshold_covers_truth():
    """Set geometry smoke: a wide Lambda threshold should cover truth often."""
    energies = np.geomspace(40, 3000, 10)
    pitches = np.linspace(0, 180, 20)
    pitch_2d = np.broadcast_to(pitches, (len(energies), len(pitches)))
    ql = QuasiLikelihoodConfig(
        n_energy_quad=3, sigma_rel=0.2, sigma_abs=1e-3, use_response_folding=True
    )
    true_u, true_r = -70.0, 0.7
    mean = mean_model_for_params(
        energies,
        pitch_2d,
        u_surface=true_u,
        bs_over_bm=true_r,
        beam_amp=0.0,
        cfg=ql,
    )
    mask = build_calibration_mask(mean)

    cfg = ProfileCIConfig(
        coarse_u_count=31,
        n_r_grid=7,
        n_beam_grid=3,
        bootstrap_n=0,
        u_min=-300.0,
        u_max=20.0,
        r_min=0.2,
        r_max=1.0,
        ql=ql,
    )
    rng = np.random.default_rng(42)
    n_trials = 12
    covered = 0
    valid = 0
    widths = []
    for _ in range(n_trials):
        data = simulate_calibrated_flux(mean, mask, cfg=ql, rng=rng)
        result = fit_profile_confidence_set(
            data,
            energies,
            pitch_2d,
            polarity=-1,
            cfg=cfg,
            skip_bootstrap=True,
            # Generous vs chi2_1: documents that asymptotic c=1 is overconfident
            # for this QL; bootstrap calibration is required in production.
            c_alpha_override=25.0,
        )
        cs = result.confidence_set
        if cs.gate_reason not in (GateReason.OK, GateReason.FULL_DOMAIN):
            continue
        valid += 1
        covered += int(_covers(cs, true_u))
        if cs.components:
            lo = min(c.lo for c in cs.components)
            hi = max(c.hi for c in cs.components)
            widths.append(hi - lo)

    assert valid >= 8
    rate = covered / valid
    assert rate >= 0.5, f"coverage too low: {covered}/{valid} = {rate:.2f}"
    assert float(np.median(widths)) > 1.0
