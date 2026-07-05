"""Regression tests for temporal basis functions.

Guards the linear-basis normalization bug: the basis must scale with time at
reconstruction (where it is evaluated one time at a time), not saturate to ~1.0.
"""

import numpy as np

from src.temporal.basis import (
    _get_basis_func_by_name,
    build_temporal_design,
    parse_basis_spec,
)
from src.temporal.coefficients import DEFAULT_SYNODIC_PERIOD_DAYS


def test_linear_basis_scales_with_time_at_reconstruction() -> None:
    linear = _get_basis_func_by_name("linear")
    period_hours = 24.0 * DEFAULT_SYNODIC_PERIOD_DAYS

    # reconstruct_at_times evaluates each basis on a single-element array.
    v0 = linear(np.array([0.0]))[0]
    v1 = linear(np.array([period_hours]))[0]
    v2 = linear(np.array([2.0 * period_hours]))[0]

    assert v0 == 0.0
    assert np.isclose(v1, 1.0)
    # Must keep scaling with time; a t/t.max() normalization would give ~1.0 here.
    assert np.isclose(v2, 2.0)


def test_linear_basis_agrees_between_fit_and_reconstruction() -> None:
    """The divisor is data-independent, so the fit-time design column must equal
    the single-point (reconstruction) evaluation at each time."""
    bases = parse_basis_spec("linear")
    t_hours = np.linspace(0.0, 500.0, 50)
    design = build_temporal_design(t_hours, bases)  # (N, 1), fit-time evaluation

    linear = _get_basis_func_by_name("linear")
    single = np.array([linear(np.array([t]))[0] for t in t_hours])
    assert np.allclose(design[:, 0], single)


def test_constant_basis_recovers_known_harmonic_field() -> None:
    """Round-trip: synthesize a real field from known Y_lm coefficients, fit it
    with a constant temporal basis, and confirm the reconstruction matches.

    Exercises the spatial harmonic design, the lsqr solve, the reality condition,
    and reconstruction end to end.
    """
    from src.temporal.basis import fit_temporal_basis, reconstruct_at_times
    from src.temporal.coefficients import (
        _build_harmonic_design,
        _enforce_reality_condition,
        _harmonic_coefficient_count,
    )

    lmax = 2
    n_coeffs = _harmonic_coefficient_count(lmax)
    rng = np.random.default_rng(1)
    true_coeffs = _enforce_reality_condition(
        rng.standard_normal(n_coeffs) + 1j * rng.standard_normal(n_coeffs), lmax
    )

    n = 800
    lat = rng.uniform(-88.0, 88.0, n)
    lon = rng.uniform(-180.0, 180.0, n)
    design = _build_harmonic_design(lat, lon, lmax)
    potential = np.real(design @ true_coeffs)  # real field by construction

    utc = np.full(n, np.datetime64("1998-09-16T00:00:00"))
    result = fit_temporal_basis(
        utc, lat, lon, potential, lmax=lmax, basis_spec="constant"
    )

    # Noiseless, full-rank design -> essentially exact fit.
    assert result.rms_residual < 1e-3

    recon = reconstruct_at_times(result, utc[:1], reference_time=utc[0])[0]
    predicted = np.real(design @ recon.coeffs)
    assert np.allclose(predicted, potential, atol=1e-3)
