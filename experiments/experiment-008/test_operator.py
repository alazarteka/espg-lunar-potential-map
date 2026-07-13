"""Tests for the Experiment 008 observation operator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from src.temporal.coefficients import _build_harmonic_design

MODULE_PATH = Path(__file__).with_name("observation_operator.py")
SPEC = importlib.util.spec_from_file_location("experiment008_operator", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
operator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = operator
SPEC.loader.exec_module(operator)


def test_harmonic_indices_have_expected_order() -> None:
    degree, order = operator.harmonic_indices(2)
    np.testing.assert_array_equal(degree, [0, 1, 1, 1, 2, 2, 2, 2, 2])
    np.testing.assert_array_equal(order, [0, -1, 0, 1, -2, -1, 0, 1, 2])


def test_real_basis_is_approximately_orthonormal_on_sphere() -> None:
    rng = np.random.default_rng(8102)
    longitude = rng.uniform(-180.0, 180.0, 200_000)
    latitude = np.rad2deg(np.arcsin(rng.uniform(-1.0, 1.0, longitude.size)))
    basis = operator.real_spherical_harmonics(latitude, longitude, 2)
    gram = 4.0 * np.pi * (basis.T @ basis) / basis.shape[0]
    np.testing.assert_allclose(gram, np.eye(9), atol=1.2e-2)


def test_fast_all_conversion_matches_scalar_evaluation() -> None:
    geometry = operator.SweepGeometry(
        source_index=np.array([0, 0]),
        spec_no=np.array([10, 11]),
        row_utc_ns=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64),
        latitude=np.array([[0.0, 10.0, 20.0], [-30.0, -20.0, -10.0]]),
        longitude=np.array([[0.0, 30.0, 60.0], [90.0, 120.0, 150.0]]),
        polarity=np.ones((2, 3), dtype=np.int8),
        valid_rows=np.array([[True, False, True], [True, True, True]]),
        midpoint_utc_ns=np.array([2, 5], dtype=np.int64),
    )
    actual = operator.sweep_harmonics(geometry, 3, chunk_sweeps=1)
    expected = np.vstack(
        (
            operator.real_spherical_harmonics([0.0, 20.0], [0.0, 60.0], 3).mean(axis=0),
            operator.real_spherical_harmonics(
                [-30.0, -20.0, -10.0], [90.0, 120.0, 150.0], 3
            ).mean(axis=0),
        )
    )
    np.testing.assert_allclose(actual, expected, atol=2e-15)


def test_real_basis_matches_repository_complex_design() -> None:
    rng = np.random.default_rng(8128)
    lmax = 4
    latitude = rng.uniform(-90.0, 90.0, 200)
    longitude = rng.uniform(-180.0, 180.0, 200)
    real_coefficients = rng.standard_normal((lmax + 1) ** 2)
    complex_coefficients = operator.real_to_complex_coefficients(
        real_coefficients, lmax
    )
    real_field = (
        operator.real_spherical_harmonics(latitude, longitude, lmax) @ real_coefficients
    )
    complex_field = np.real(
        _build_harmonic_design(latitude, longitude, lmax) @ complex_coefficients
    )
    np.testing.assert_allclose(real_field, complex_field, atol=2e-14)
    np.testing.assert_allclose(
        np.linalg.norm(real_coefficients),
        np.linalg.norm(complex_coefficients),
        atol=2e-14,
    )


def test_midpoint_reducer_selects_nearest_valid_row() -> None:
    geometry = operator.SweepGeometry(
        source_index=np.array([0]),
        spec_no=np.array([10]),
        row_utc_ns=np.array([[0, 10, 20]], dtype=np.int64),
        latitude=np.array([[0.0, 15.0, 30.0]]),
        longitude=np.array([[0.0, 20.0, 40.0]]),
        polarity=np.ones((1, 3), dtype=np.int8),
        valid_rows=np.array([[True, True, True]]),
        midpoint_utc_ns=np.array([10], dtype=np.int64),
    )
    actual = operator.sweep_harmonics(geometry, 2, reducer="midpoint")
    expected = operator.real_spherical_harmonics([15.0], [20.0], 2)
    np.testing.assert_allclose(actual, expected)


def test_tensor_sweep_averages_row_level_products() -> None:
    day_ns = int(86_400 * 1e9)
    geometry = operator.SweepGeometry(
        source_index=np.array([0]),
        spec_no=np.array([10]),
        row_utc_ns=np.array([[0, day_ns]], dtype=np.int64),
        latitude=np.array([[0.0, 30.0]]),
        longitude=np.array([[0.0, 90.0]]),
        polarity=np.ones((1, 2), dtype=np.int8),
        valid_rows=np.array([[True, True]]),
        midpoint_utc_ns=np.array([day_ns // 2], dtype=np.int64),
    )
    actual = operator.sweep_tensor_harmonics(
        geometry,
        1,
        "constant_synodic",
        period_days=4.0,
        epoch_ns=0,
    )
    spatial = operator.real_spherical_harmonics(
        geometry.latitude.ravel(), geometry.longitude.ravel(), 1
    )
    temporal = operator.temporal_factors(
        geometry.row_utc_ns.ravel(),
        "constant_synodic",
        period_days=4.0,
        epoch_ns=0,
    )
    expected = operator.tensor_operator(spatial, temporal).mean(axis=0)
    np.testing.assert_allclose(actual[0], expected)


def test_direct_spectrum_recovers_rank_and_condition() -> None:
    matrix = np.diag([1.0, 1e-4, 1e-8])
    diagnostics = operator.direct_spectrum(matrix)
    assert diagnostics.exact_rank == 3
    assert diagnostics.effective_ranks[1e-3] == 3
    # Column normalization removes pure unit scaling from conditioning.
    assert diagnostics.condition_number == 1.0


def test_metric_spectrum_preserves_physical_visibility() -> None:
    matrix = np.diag([1.0, 1e-4, 1e-8])
    diagnostics = operator.metric_spectrum(matrix, np.eye(3))
    np.testing.assert_allclose(diagnostics.condition_number, 1e8, rtol=1e-12)
    np.testing.assert_allclose(
        diagnostics.singular_values,
        np.array([1.0, 1e-4, 1e-8]) / np.sqrt(3.0),
        rtol=1e-12,
    )


def test_tensor_operator_uses_temporal_major_columns() -> None:
    spatial = np.array([[1.0, 2.0], [3.0, 4.0]])
    temporal = np.array([[5.0, 6.0], [7.0, 8.0]])
    actual = operator.tensor_operator(spatial, temporal)
    np.testing.assert_array_equal(
        actual, [[5.0, 10.0, 6.0, 12.0], [21.0, 28.0, 24.0, 32.0]]
    )


def test_mode_visibility_reports_scaled_sample_rms() -> None:
    matrix = np.eye(2)
    mode = np.array([1.0, 0.0])
    sampled, ratio = operator.mode_visibility(matrix, mode, np.eye(2))
    np.testing.assert_allclose(sampled, 100.0 / np.sqrt(2.0))
    np.testing.assert_allclose(ratio, 1.0 / np.sqrt(2.0))


def test_underdetermined_spectrum_returns_exact_null_mode() -> None:
    matrix = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 1.0]])
    diagnostics = operator.direct_spectrum(matrix)
    assert diagnostics.smallest_physical_mode is not None
    np.testing.assert_allclose(
        matrix @ diagnostics.smallest_physical_mode,
        0.0,
        atol=1e-14,
    )


def test_temporal_field_gram_matches_numerical_quadrature() -> None:
    period_days = 7.0
    start_ns = int(2.0 * 86_400 * 1e9)
    stop_ns = int(19.0 * 86_400 * 1e9)
    actual = operator.temporal_field_gram(
        0,
        "constant_synodic",
        start_utc_ns=start_ns,
        stop_utc_ns=stop_ns,
        period_days=period_days,
    )
    utc = np.linspace(start_ns, stop_ns, 1_000_001, dtype=np.int64)
    temporal = operator.temporal_factors(
        utc,
        "constant_synodic",
        period_days=period_days,
        epoch_ns=0,
    )
    expected_temporal = temporal.T @ temporal / temporal.shape[0]
    expected = expected_temporal / (4.0 * np.pi)
    np.testing.assert_allclose(actual, expected, atol=2e-6)
