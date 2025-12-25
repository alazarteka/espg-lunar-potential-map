#!/usr/bin/env python3
"""
Benchmark batched PyTorch Kappa fitter vs sequential scipy.

Usage:
    uv run python scripts/dev/benchmark_kappa_torch.py --input data/1998/091_120APR/3D980415.TAB
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from src import config
from src.flux import ERData


def prepare_spectra_data(er_data: ERData, n_spectra: int | None = None):
    """
    Prepare data arrays for batch fitting.

    Returns:
        energy: (E,) energy grid
        flux_data: (N, E) flux measurements
        density_estimates: (N,) density estimates
        spec_nos: (N,) spectrum numbers
    """
    from src.kappa import Kappa

    spec_nos = er_data.data[config.SPEC_NO_COLUMN].unique()
    if n_spectra is not None:
        spec_nos = spec_nos[:n_spectra]

    energy = None
    flux_list = []
    density_list = []
    valid_spec_nos = []

    for spec_no in spec_nos:
        try:
            kappa = Kappa(er_data, spec_no)
            if not kappa.is_data_valid:
                continue

            if energy is None:
                energy = kappa.energy_centers_mag

            flux_list.append(kappa.omnidirectional_differential_particle_flux_mag)
            density_list.append(kappa.density_estimate_mag)
            valid_spec_nos.append(spec_no)
        except Exception as e:
            print(f"Skipping spec {spec_no}: {e}")
            continue

    flux_data = np.array(flux_list)
    density_estimates = np.array(density_list)

    return energy, flux_data, density_estimates, valid_spec_nos


def benchmark_scipy(er_data: ERData, n_spectra: int, n_starts: int = 10):
    """Benchmark sequential scipy fitting."""
    from src.kappa import Kappa

    spec_nos = er_data.data[config.SPEC_NO_COLUMN].unique()[:n_spectra]

    results = []
    start = time.time()

    for i, spec_no in enumerate(spec_nos):
        try:
            kappa = Kappa(er_data, spec_no)
            if not kappa.is_data_valid:
                continue

            # use_weights=False for fair comparison with torch (which uses uniform weights)
            fit = kappa.fit(n_starts=n_starts, use_fast=True, use_weights=False)
            if fit is not None:
                results.append({
                    'spec_no': spec_no,
                    'kappa': fit.params.kappa,
                    'theta': fit.params.theta.magnitude,
                    'chi2': fit.error,
                })
        except Exception as e:
            print(f"Error on spec {spec_no}: {e}")

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            print(f"  scipy: {i + 1}/{n_spectra} ({rate:.1f} spec/s)")

    elapsed = time.time() - start
    return results, elapsed


def benchmark_torch(er_data: ERData, n_spectra: int):
    """Benchmark batched PyTorch fitting."""
    from src.kappa_torch import KappaFitterTorch

    print("Preparing batch data...")
    energy, flux_data, density_estimates, spec_nos = prepare_spectra_data(er_data, n_spectra)

    print(f"Prepared {len(spec_nos)} valid spectra")

    fitter = KappaFitterTorch(
        device="cpu",
        popsize=30,
        maxiter=100,
    )

    print("Running batched fit...")
    start = time.time()
    kappa, theta, chi2 = fitter.fit_batch(energy, flux_data, density_estimates)
    elapsed = time.time() - start

    results = [
        {'spec_no': sn, 'kappa': k, 'theta': t, 'chi2': c}
        for sn, k, t, c in zip(spec_nos, kappa, theta, chi2)
    ]

    return results, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--n-spectra", type=int, default=50)
    parser.add_argument("--compare", action="store_true", help="Run scipy comparison")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    er_data = ERData(str(args.input))

    total_spectra = len(er_data.data[config.SPEC_NO_COLUMN].unique())
    print(f"Total spectra in file: {total_spectra}")

    # Benchmark torch
    print(f"\n=== PyTorch batched ({args.n_spectra} spectra) ===")
    torch_results, torch_time = benchmark_torch(er_data, args.n_spectra)
    print(f"Torch time: {torch_time:.2f}s ({len(torch_results) / torch_time:.1f} spec/s)")

    if args.compare:
        # Benchmark scipy
        print(f"\n=== scipy sequential ({args.n_spectra} spectra) ===")
        scipy_results, scipy_time = benchmark_scipy(er_data, args.n_spectra)
        print(f"Scipy time: {scipy_time:.2f}s ({len(scipy_results) / scipy_time:.1f} spec/s)")

        # Compare
        print(f"\n=== Speedup ===")
        print(f"Torch: {torch_time:.2f}s")
        print(f"Scipy: {scipy_time:.2f}s")
        print(f"Speedup: {scipy_time / torch_time:.1f}x")

        # Compare results
        print(f"\n=== Result comparison (first 5) ===")
        for i in range(min(5, len(torch_results))):
            tr = torch_results[i]
            sr = scipy_results[i] if i < len(scipy_results) else None
            print(f"Spec {tr['spec_no']}:")
            print(f"  Torch: κ={tr['kappa']:.3f}, θ={tr['theta']:.2e}, χ²={tr['chi2']:.2f}")
            if sr:
                print(f"  Scipy: κ={sr['kappa']:.3f}, θ={sr['theta']:.2e}, χ²={sr['chi2']:.2f}")
    else:
        # Just show torch results
        print(f"\n=== Results (first 5) ===")
        for tr in torch_results[:5]:
            print(f"Spec {tr['spec_no']}: κ={tr['kappa']:.3f}, θ={tr['theta']:.2e}, χ²={tr['chi2']:.2f}")


if __name__ == "__main__":
    main()
