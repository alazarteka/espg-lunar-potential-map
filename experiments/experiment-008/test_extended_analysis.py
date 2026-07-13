"""Tests for Experiment 008 extended temporal analyses."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

EXPERIMENT_DIR = Path(__file__).parent
sys.path.insert(0, str(EXPERIMENT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "experiment008_extended", EXPERIMENT_DIR / "extended_analysis.py"
)
assert SPEC is not None
assert SPEC.loader is not None
extended = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = extended
SPEC.loader.exec_module(extended)


def test_temporal_column_indices_preserve_temporal_major_blocks() -> None:
    actual = extended.temporal_column_indices(1, source_lmax=2)
    np.testing.assert_array_equal(actual, [0, 1, 2, 3, 9, 10, 11, 12, 18, 19, 20, 21])


def test_contrast_indices_remove_each_temporal_monopole() -> None:
    actual = extended.contrast_indices(1, temporal_columns=3)
    np.testing.assert_array_equal(actual, [1, 2, 3, 5, 6, 7, 9, 10, 11])


def test_metric_inverse_square_root_whitens_metric() -> None:
    metric = np.array([[2.0, 0.3], [0.3, 0.7]])
    inverse = extended.metric_inverse_square_root(metric)
    np.testing.assert_allclose(inverse @ metric @ inverse, np.eye(2), atol=2e-15)


def test_common_cohort_aligns_source_spec_keys() -> None:
    def geometry(source: list[int], spec: list[int]) -> object:
        size = len(source)
        return extended.operator.SweepGeometry(
            source_index=np.asarray(source),
            spec_no=np.asarray(spec),
            row_utc_ns=np.zeros((size, 1), dtype=np.int64),
            latitude=np.zeros((size, 1)),
            longitude=np.zeros((size, 1)),
            polarity=np.ones((size, 1), dtype=np.int8),
            valid_rows=np.ones((size, 1), dtype=bool),
            midpoint_utc_ns=np.zeros(size, dtype=np.int64),
        )

    left = geometry([0, 0, 1], [10, 20, 10])
    right = geometry([1, 0, 2], [10, 20, 10])
    left_index, right_index = extended.common_cohort(left, right)
    np.testing.assert_array_equal(left_index, [1, 2])
    np.testing.assert_array_equal(right_index, [1, 0])


def test_broadband_truths_have_frozen_global_rms() -> None:
    truths = extended.broadband_truths(5, 20, seed=1234)
    np.testing.assert_allclose(np.linalg.norm(truths, axis=1), 100.0)
    shell = extended.broadband_truths(5, 20, seed=1234, shell=3)
    degree, _ = extended.operator.harmonic_indices(5)
    np.testing.assert_array_equal(shell[:, degree != 3], 0.0)


def test_recover_methods_recovers_noiseless_truth() -> None:
    rng = np.random.default_rng(25)
    matrix = rng.standard_normal((100, 4))
    truth = rng.standard_normal((3, 4))
    observations = truth @ matrix.T
    estimates = extended.recover_methods(
        matrix,
        observations,
        tsvd_thresholds=[1e-9],
        ridge_penalties=[],
    )
    np.testing.assert_allclose(estimates["unregularized"], truth, atol=2e-15)
    np.testing.assert_allclose(estimates["tsvd_1e-09"], truth, atol=2e-15)


def test_gaussian_cap_peaks_at_declared_center() -> None:
    values = extended.gaussian_cap(
        np.array([10.0, 10.0, -20.0]),
        np.array([30.0, 60.0, 30.0]),
        center_latitude_deg=10.0,
        center_longitude_deg=30.0,
        fwhm_deg=15.0,
    )
    np.testing.assert_allclose(values[0], 1.0)
    assert np.all(values[1:] < 1.0)


def test_cap_projection_has_declared_global_rms_decomposition() -> None:
    _, projected, bias = extended.cap_projection(0.0, 0.0, 30.0, lmax=5)
    np.testing.assert_allclose(np.linalg.norm(projected) ** 2 + bias**2, 100.0**2)
