"""Focused numerical checks for Experiment 003."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from run import _interval_log_probability
from scipy.stats import nbinom

HERE = Path(__file__).resolve().parent


def test_negative_binomial_interval_probability_matches_scipy() -> None:
    lower = np.array([[0.0, 1.0, 5.0, 30.0]])
    upper = np.array([[0.0, 2.0, 9.0, 40.0]])
    mu = np.array([[[0.2, 2.0, 7.0, 35.0]]])
    alpha = 0.13
    size = 1.0 / alpha
    probability = size / (size + mu[0, 0])
    expected = nbinom.cdf(upper[0], size, probability) - nbinom.cdf(
        lower[0] - 1.0, size, probability
    )
    actual = np.exp(_interval_log_probability(lower, upper, mu, alpha))[0, 0]
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-14)


def test_frozen_panel_is_complete_and_disjoint() -> None:
    with (HERE / "panel_manifest.csv").open(encoding="ascii", newline="") as source:
        rows = list(csv.DictReader(source))
    by_panel = {
        panel: {int(row["spec_no"]) for row in rows if row["panel"] == panel}
        for panel in ("benchmark", "discovery", "confirmation")
    }
    assert len(by_panel["benchmark"]) == 3
    assert len(by_panel["discovery"]) == 7
    assert len(by_panel["confirmation"]) == 50
    assert not (by_panel["benchmark"] & by_panel["discovery"])
    assert not (by_panel["benchmark"] & by_panel["confirmation"])
    assert not (by_panel["discovery"] & by_panel["confirmation"])
    assert by_panel["confirmation"] == set(range(669, 719))
