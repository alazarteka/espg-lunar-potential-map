"""Tests for the Poisson noise model and Cash-deviance objective (CPU + torch)."""

from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.losscone import noise
from src.losscone.chi2 import compute_cash_deviance_batch


def test_channel_integration_times():
    dt = noise.channel_integration_times()
    assert dt.shape == (config.CHANNELS,)
    # Δt = ACCUMULATION_TIME / N_azimuth(band); 2.5 s / {4,8,16}
    assert set(np.round(dt, 6)) <= {
        round(2.5 / 4, 6),
        round(2.5 / 8, 6),
        round(2.5 / 16, 6),
    }
    assert np.all(dt > 0)


def test_reconstruct_counts_formula_and_clamp():
    energies = np.array([100.0, 1000.0])
    dt = np.array([0.3, 0.6, 0.15])
    raw = np.array([[10.0, 20.0, -5.0], [1.0, 2.0, 3.0]])
    lam = noise.reconstruct_counts(raw, energies, dt)
    g = float(config.GEOMETRIC_FACTOR.magnitude)
    expected = np.clip(raw, 0, None) * g * energies[:, None] * dt[None, :]
    np.testing.assert_allclose(lam, expected)
    assert lam[0, 2] == 0.0  # negative flux clamped to zero count


def test_chunk_noise_kappa_identity():
    energies = np.array([100.0, 1000.0])
    dt = np.array([0.3, 0.6, 0.15])
    raw = np.array([[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]])
    f_scale = np.array([12.0, 4.0])  # per-row incident normalization
    norm2d = raw / f_scale[:, None]
    counts, kappa, bg = noise.chunk_noise(raw, norm2d, energies, dt)
    # κ should equal F_scale·G·E·Δt (independent of the per-bin flux)
    np.testing.assert_allclose(kappa, noise.count_scale(f_scale, energies, dt))
    np.testing.assert_allclose(bg, config.LOSS_CONE_BACKGROUND_RATE_CPS * dt)


def _rng_case(seed=0):
    rng = np.random.default_rng(seed)
    nE, nP, n_models = 6, 8, 5
    models = rng.uniform(0.05, 1.5, size=(n_models, nE, nP))
    kappa = rng.uniform(50.0, 500.0, size=(nE, nP))
    bg = rng.uniform(1.0, 5.0, size=(nE, nP))
    counts = rng.uniform(0.0, 2000.0, size=(nE, nP))
    mask = rng.uniform(size=(nE, nP)) > 0.2
    return models, counts, kappa, bg, mask


def test_deviance_zero_at_exact_match():
    _, counts, kappa, bg, mask = _rng_case(1)
    # Choose the model so μ = m·κ + b == counts exactly -> per-bin deviance 0.
    m = (counts - bg) / kappa
    m = np.clip(m, 0.05, None)[None, :, :]
    dev = compute_cash_deviance_batch(m, counts, kappa, bg, model_mask=mask)
    # Where the clip didn't bind, deviance is ~0; overall it is the global min.
    assert np.isfinite(dev[0])
    dev_off = compute_cash_deviance_batch(m * 1.3, counts, kappa, bg, model_mask=mask)
    assert dev_off[0] > dev[0]  # moving away from truth raises the deviance


def test_deviance_handles_zero_counts():
    models, counts, kappa, bg, mask = _rng_case(2)
    counts = counts.copy()
    counts[0, 0] = 0.0  # a zero-count bin (C ln(C/μ) -> 0, no NaN)
    dev = compute_cash_deviance_batch(models, counts, kappa, bg, model_mask=mask)
    assert np.all(np.isfinite(dev))
    assert np.all(dev >= 0.0)


def test_deviance_reduced_and_too_few_bins():
    models, counts, kappa, bg, _ = _rng_case(3)
    full = np.ones(counts.shape, dtype=bool)
    raw = compute_cash_deviance_batch(models, counts, kappa, bg, model_mask=full)
    red = compute_cash_deviance_batch(
        models, counts, kappa, bg, model_mask=full, reduced=True
    )
    dof = max(counts.size - 3, 1)
    np.testing.assert_allclose(red, raw / dof)
    tiny = np.zeros(counts.shape, dtype=bool)
    tiny[0, :2] = True  # < LILLIS_MIN_VALID_BINS
    assert np.all(
        np.isnan(
            compute_cash_deviance_batch(models, counts, kappa, bg, model_mask=tiny)
        )
    )


def test_cpu_torch_deviance_parity():
    torch = pytest.importorskip("torch")
    from src.losscone.torch.chi2 import compute_cash_deviance_batch_torch

    models, counts, kappa, bg, mask = _rng_case(4)
    cpu = compute_cash_deviance_batch(models, counts, kappa, bg, model_mask=mask)
    t = compute_cash_deviance_batch_torch(
        torch.tensor(models, dtype=torch.float64),
        torch.tensor(counts, dtype=torch.float64),
        torch.tensor(kappa, dtype=torch.float64),
        torch.tensor(bg, dtype=torch.float64),
        torch.tensor(mask, dtype=torch.bool),
    ).numpy()
    np.testing.assert_allclose(cpu, t, rtol=1e-10, atol=1e-10)


def test_cpu_torch_deviance_parity_multichunk():
    torch = pytest.importorskip("torch")
    from src.losscone.torch.chi2 import (
        compute_cash_deviance_batch_torch,
        compute_cash_deviance_multi_chunk_torch,
    )

    models, counts, kappa, bg, mask = _rng_case(5)
    # multi-chunk with N=1 must match the single-chunk batch form.
    batch = compute_cash_deviance_batch_torch(
        torch.tensor(models),
        torch.tensor(counts),
        torch.tensor(kappa),
        torch.tensor(bg),
        torch.tensor(mask),
    )
    multi = compute_cash_deviance_multi_chunk_torch(
        torch.tensor(models).unsqueeze(0),
        torch.tensor(counts).unsqueeze(0),
        torch.tensor(kappa).unsqueeze(0),
        torch.tensor(bg).unsqueeze(0),
        torch.tensor(mask).unsqueeze(0),
    )
    np.testing.assert_allclose(batch.numpy(), multi.squeeze(0).numpy(), rtol=1e-12)
