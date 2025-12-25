#!/usr/bin/env python3
"""
Test Kappa fitter variations: weights and dayside refit.

Compares:
1. Kappa fits with vs without log-flux uncertainty weights
2. Dayside SC-potential with vs without energy-shift refit

Usage:
    uv run python scripts/dev/test_kappa_variations.py --input data/1998/091_120APR/3D980415.TAB
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from src import config
from src.flux import ERData
from src.kappa import Kappa


def compute_sigma_log_flux(kappa_obj: Kappa) -> np.ndarray:
    """Extract log-flux uncertainty from Kappa object."""
    return kappa_obj.sigma_log_flux


def test_weights_impact(er_data: ERData, n_spectra: int = 50):
    """
    Test Kappa fitting with and without log-flux uncertainty weights.

    Compares:
    - No weights (uniform = 1)
    - Log-flux uncertainty weights (1 / sigma_log_flux)
    - Clipped weights (1 / max(sigma_log_flux, min_sigma))
    """
    print("\n" + "=" * 60)
    print("TEST: Kappa fits with vs without log-flux weights")
    print("=" * 60)

    # First, examine typical sigma_log_flux values
    spec_nos_sample = er_data.data[config.SPEC_NO_COLUMN].unique()[:5]
    print("\n--- Typical sigma_log_flux values ---")
    for spec_no in spec_nos_sample:
        kappa = Kappa(er_data, spec_no)
        if kappa.is_data_valid:
            sigma = kappa.sigma_log_flux
            weights = 1.0 / (sigma + config.EPS)
            # Also show clipped version
            sigma_clipped = np.maximum(sigma, 0.05)  # 5% min uncertainty
            weights_clipped = 1.0 / sigma_clipped
            print(f"Spec {spec_no}:")
            print(f"  Raw:     sigma=[{sigma.min():.4f}, {sigma.max():.4f}], weights=[{weights.min():.1f}, {weights.max():.0f}]")
            print(f"  Clipped: sigma=[{sigma_clipped.min():.4f}, {sigma_clipped.max():.4f}], weights=[{weights_clipped.min():.1f}, {weights_clipped.max():.1f}]")
    print()

    spec_nos = er_data.data[config.SPEC_NO_COLUMN].unique()[:n_spectra]

    results_no_weights = []
    results_with_weights = []
    results_clipped_weights = []

    start = time.time()
    for spec_no in spec_nos:
        try:
            kappa_obj = Kappa(er_data, spec_no)
            if not kappa_obj.is_data_valid:
                continue

            # Fit without weights (use_weights=False)
            fit_no_w = kappa_obj.fit(n_starts=5, use_fast=True, use_weights=False)
            # Fit with weights (use_weights=True, the default)
            fit_with_w = kappa_obj.fit(n_starts=5, use_fast=True, use_weights=True)

            # Fit with clipped weights (manually)
            # This uses the internal objective with modified sigma_log_flux
            original_sigma = kappa_obj.sigma_log_flux.copy()
            kappa_obj.sigma_log_flux = np.maximum(original_sigma, 0.05)
            fit_clipped = kappa_obj.fit(n_starts=5, use_fast=True, use_weights=True)
            kappa_obj.sigma_log_flux = original_sigma  # restore

            if fit_no_w and fit_with_w and fit_clipped:
                results_no_weights.append({
                    'spec_no': spec_no,
                    'kappa': fit_no_w.params.kappa,
                    'theta': fit_no_w.params.theta.magnitude,
                    'chi2': fit_no_w.error,
                })
                results_with_weights.append({
                    'spec_no': spec_no,
                    'kappa': fit_with_w.params.kappa,
                    'theta': fit_with_w.params.theta.magnitude,
                    'chi2': fit_with_w.error,
                })
                results_clipped_weights.append({
                    'spec_no': spec_no,
                    'kappa': fit_clipped.params.kappa,
                    'theta': fit_clipped.params.theta.magnitude,
                    'chi2': fit_clipped.error,
                })
        except Exception as e:
            print(f"  Error on spec {spec_no}: {e}")

    elapsed = time.time() - start
    print(f"\nProcessed {len(results_no_weights)} spectra in {elapsed:.1f}s")

    # Compare results
    if results_no_weights:
        kappa_no_w = np.array([r['kappa'] for r in results_no_weights])
        kappa_with_w = np.array([r['kappa'] for r in results_with_weights])
        kappa_clipped = np.array([r['kappa'] for r in results_clipped_weights])

        theta_no_w = np.array([r['theta'] for r in results_no_weights])
        theta_with_w = np.array([r['theta'] for r in results_with_weights])
        theta_clipped = np.array([r['theta'] for r in results_clipped_weights])

        chi2_no_w = np.array([r['chi2'] for r in results_no_weights])
        chi2_with_w = np.array([r['chi2'] for r in results_with_weights])
        chi2_clipped = np.array([r['chi2'] for r in results_clipped_weights])

        print(f"\n--- Summary statistics ---")
        print(f"{'Method':<20} {'κ mean':>10} {'κ std':>10} {'θ mean':>12} {'χ² median':>12}")
        print("-" * 66)
        print(f"{'No weights':<20} {kappa_no_w.mean():>10.3f} {kappa_no_w.std():>10.3f} {theta_no_w.mean():>12.2e} {np.median(chi2_no_w):>12.1f}")
        print(f"{'Raw weights':<20} {kappa_with_w.mean():>10.3f} {kappa_with_w.std():>10.3f} {theta_with_w.mean():>12.2e} {np.median(chi2_with_w):>12.1f}")
        print(f"{'Clipped (5%) weights':<20} {kappa_clipped.mean():>10.3f} {kappa_clipped.std():>10.3f} {theta_clipped.mean():>12.2e} {np.median(chi2_clipped):>12.1f}")

        # Compare clipped vs no weights
        kappa_diff = kappa_clipped - kappa_no_w
        theta_ratio = theta_clipped / theta_no_w
        print(f"\n--- Clipped weights vs No weights ---")
        print(f"κ difference: mean={kappa_diff.mean():.3f}, std={kappa_diff.std():.3f}")
        print(f"θ ratio: mean={theta_ratio.mean():.3f}, std={theta_ratio.std():.3f}")

        # Sample comparison
        print(f"\n--- First 5 spectra ---")
        for i in range(min(5, len(results_no_weights))):
            nw = results_no_weights[i]
            ww = results_with_weights[i]
            cw = results_clipped_weights[i]
            print(f"Spec {nw['spec_no']}:")
            print(f"  No weights:      κ={nw['kappa']:.3f}, θ={nw['theta']:.2e}, χ²={nw['chi2']:.1f}")
            print(f"  Raw weights:     κ={ww['kappa']:.3f}, θ={ww['theta']:.2e}, χ²={ww['chi2']:.1f}")
            print(f"  Clipped weights: κ={cw['kappa']:.3f}, θ={cw['theta']:.2e}, χ²={cw['chi2']:.1f}")


def test_dayside_refit(er_data: ERData, n_spectra: int = 30):
    """
    Test dayside SC-potential with and without energy-shift refit.

    The full algorithm:
    1. Fit kappa distribution
    2. Compute U from JU curve
    3. Shift energies by -U
    4. Refit kappa
    5. Recompute U from new JU curve

    The fast version skips steps 3-5.
    """
    from src.physics.charging import electron_current_density_magnitude
    from src.physics.jucurve import U_from_J
    from src.spacecraft_potential import calculate_potential
    from src.potential_mapper.spice import load_spice_files

    print("\n" + "=" * 60)
    print("TEST: Dayside SC-potential with vs without refit")
    print("=" * 60)

    load_spice_files()

    spec_nos = er_data.data[config.SPEC_NO_COLUMN].unique()[:n_spectra * 3]  # Extra to find dayside

    results_no_refit = []
    results_with_refit = []
    timing_no_refit = []
    timing_with_refit = []

    for spec_no in spec_nos:
        if len(results_no_refit) >= n_spectra:
            break

        try:
            # Fast path (no refit)
            kappa = Kappa(er_data, spec_no)
            if not kappa.is_data_valid:
                continue

            start = time.time()
            fit = kappa.fit(n_starts=5, use_fast=True)
            if fit is None:
                continue

            density, kappa_val, theta = fit.params.to_tuple()
            current = electron_current_density_magnitude(
                density, kappa_val, theta, E_min=1e1, E_max=2e4, n_steps=10
            )
            U_fast = U_from_J(J_target=current, U_min=0.0, U_max=150.0)
            time_fast = time.time() - start

            # Full path (with refit) - use the original calculate_potential
            start = time.time()
            result = calculate_potential(er_data, spec_no)
            time_full = time.time() - start

            if result is None:
                continue

            params_full, U_full = result
            U_full_val = U_full.magnitude

            # Only compare dayside (positive U)
            if U_full_val > 0:
                results_no_refit.append({
                    'spec_no': spec_no,
                    'U': float(U_fast),
                    'kappa': kappa_val,
                    'theta': theta,
                })
                results_with_refit.append({
                    'spec_no': spec_no,
                    'U': U_full_val,
                    'kappa': params_full.kappa,
                    'theta': params_full.theta.magnitude,
                })
                timing_no_refit.append(time_fast)
                timing_with_refit.append(time_full)

        except Exception as e:
            continue

    print(f"\nFound {len(results_no_refit)} dayside spectra")

    if results_no_refit:
        U_fast = np.array([r['U'] for r in results_no_refit])
        U_full = np.array([r['U'] for r in results_with_refit])
        U_diff = U_fast - U_full
        U_rel_err = np.abs(U_diff) / (np.abs(U_full) + 1e-6)

        time_fast = np.array(timing_no_refit)
        time_full = np.array(timing_with_refit)

        print(f"\n--- Potential difference (fast - full) ---")
        print(f"U difference [V]: mean={U_diff.mean():.3f}, std={U_diff.std():.3f}, "
              f"range=[{U_diff.min():.3f}, {U_diff.max():.3f}]")
        print(f"Relative error: mean={U_rel_err.mean()*100:.2f}%, max={U_rel_err.max()*100:.2f}%")

        print(f"\n--- Timing ---")
        print(f"Fast (no refit): mean={time_fast.mean()*1000:.1f}ms, total={time_fast.sum():.2f}s")
        print(f"Full (with refit): mean={time_full.mean()*1000:.1f}ms, total={time_full.sum():.2f}s")
        print(f"Speedup: {time_full.sum() / time_fast.sum():.1f}x")

        # Sample comparison
        print(f"\n--- First 5 dayside spectra ---")
        for i in range(min(5, len(results_no_refit))):
            nf = results_no_refit[i]
            wf = results_with_refit[i]
            print(f"Spec {nf['spec_no']}:")
            print(f"  Fast:  U={nf['U']:.2f}V, κ={nf['kappa']:.3f}")
            print(f"  Full:  U={wf['U']:.2f}V, κ={wf['kappa']:.3f}")
            print(f"  Diff:  ΔU={nf['U']-wf['U']:.3f}V ({100*(nf['U']-wf['U'])/wf['U']:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--n-spectra", type=int, default=50,
                        help="Number of spectra to test")
    parser.add_argument("--test", choices=["weights", "refit", "both"], default="both",
                        help="Which test to run")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    er_data = ERData(str(args.input))

    total_spectra = len(er_data.data[config.SPEC_NO_COLUMN].unique())
    print(f"Total spectra in file: {total_spectra}")

    if args.test in ("weights", "both"):
        test_weights_impact(er_data, args.n_spectra)

    if args.test in ("refit", "both"):
        test_dayside_refit(er_data, args.n_spectra)


if __name__ == "__main__":
    main()
