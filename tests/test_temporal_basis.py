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
