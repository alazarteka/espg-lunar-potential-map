"""Focused tests for the Experiment 008 phased analysis runner."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

EXPERIMENT_DIR = Path(__file__).parent
sys.path.insert(0, str(EXPERIMENT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "experiment008_run_analysis", EXPERIMENT_DIR / "run_analysis.py"
)
assert SPEC is not None
assert SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


def test_epoch_anchored_windows_keep_boundaries_half_open() -> None:
    hour_ns = int(3600 * 1e9)
    utc = np.array([0, hour_ns - 1, hour_ns, 3 * hour_ns], dtype=np.int64)
    windows, empty = analysis.iter_windows(utc, 1.0)
    assert [window for window, _ in windows] == [0, 1, 3]
    np.testing.assert_array_equal(windows[0][1], [0, 1])
    np.testing.assert_array_equal(windows[1][1], [2])
    assert empty == 1


def test_covariance_recovery_matches_direct_diagonal_simulation() -> None:
    singular = np.array([2.0, 1.0])
    actual = analysis.covariance_recovery_quantiles(
        singular,
        rows=100,
        parameters=2,
        noise_sigma_v=25.0,
        truth_rms_v=100.0,
        draws=100_000,
        seed_namespace="test",
    )
    seed = analysis.seed_from_parts("test", "covariance", 2, 100_000)
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    standard = rng.standard_normal((100_000, 2))
    scale = 25.0 / (10.0 * singular)
    error = np.sqrt((standard * standard) @ (scale * scale)) / 100.0
    expected = (
        np.quantile(error, 0.5, method="linear"),
        np.quantile(error, 0.9, method="linear"),
    )
    np.testing.assert_allclose(actual, expected)


def test_spectrum_only_returns_physical_static_visibility() -> None:
    matrix = np.eye(3)
    actual = analysis.spectrum_only(matrix)
    np.testing.assert_allclose(actual, np.full(3, np.sqrt(4.0 * np.pi / 3.0)))


def test_tsqr_matches_direct_svd() -> None:
    rng = np.random.default_rng(8008)
    matrix = rng.standard_normal((103, 7))
    chunks = iter((matrix[:20], matrix[20:61], matrix[61:]))
    actual = analysis.tsqr_spectrum(chunks, rows=matrix.shape[0])
    expected = np.linalg.svd(matrix / np.sqrt(matrix.shape[0]), compute_uv=False)
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-15)
