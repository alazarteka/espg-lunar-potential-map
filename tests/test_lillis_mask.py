"""Tests for Lillis relative-flux mask shapes."""

from __future__ import annotations

import numpy as np

from src.losscone.masks import build_lillis_mask


def test_build_lillis_mask_3d_matches_per_spectrum_2d() -> None:
    rng = np.random.default_rng(0)
    n_spec, n_e, n_pitch = 3, 8, 12
    raw = rng.uniform(1.0, 100.0, size=(n_spec, n_e, n_pitch))
    pitches = np.tile(np.linspace(0.0, 180.0, n_pitch), (n_spec, n_e, 1))

    batch = build_lillis_mask(raw, pitches)
    assert batch.shape == (n_spec, n_e, n_pitch)

    for i in range(n_spec):
        expected = build_lillis_mask(raw[i], pitches[i])
        np.testing.assert_array_equal(batch[i], expected)
