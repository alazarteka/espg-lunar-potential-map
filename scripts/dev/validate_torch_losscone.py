#!/usr/bin/env python3
"""
Validate GPU loss-cone fitter against CPU baseline.

Compares results for the Halekas Figure 5 spectrum (data/1999/.../3D990429.TAB, spec 653).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from src import config
from src.flux import ERData, LossConeFitter, PitchAngle


def run_cpu_fit(
    er_data: ERData,
    pitch_angle: PitchAngle,
    spacecraft_potential: np.ndarray,
    spec_no: int,
) -> tuple[tuple[float, float, float, float], float]:
    """Run CPU fitter and return (results, elapsed_time)."""
    fitter = LossConeFitter(
        er_data,
        str(config.DATA_DIR / config.THETA_FILE),
        pitch_angle,
        spacecraft_potential,
        normalization_mode="ratio",
        incident_flux_stat="mean",
    )

    start = time.perf_counter()
    result = fitter._fit_surface_potential(spec_no)
    elapsed = time.perf_counter() - start

    return result, elapsed


def run_gpu_fit(
    er_data: ERData,
    pitch_angle: PitchAngle,
    spacecraft_potential: np.ndarray,
    spec_no: int,
    device: str | None = None,
) -> tuple[tuple[float, float, float, float], float]:
    """Run GPU fitter and return (results, elapsed_time)."""
    from src.model_torch import LossConeFitterTorch

    fitter = LossConeFitterTorch(
        er_data,
        str(config.DATA_DIR / config.THETA_FILE),
        pitch_angle,
        spacecraft_potential,
        normalization_mode="ratio",
        incident_flux_stat="mean",
        device=device,
    )

    # Warm-up run (JIT compilation, CUDA initialization)
    _ = fitter._fit_surface_potential_torch(spec_no)

    start = time.perf_counter()
    result = fitter._fit_surface_potential_torch(spec_no)
    elapsed = time.perf_counter() - start

    return result, elapsed


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Validate GPU loss-cone fitter")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/1999/091_120APR/3D990429.TAB"),
        help="ER data file",
    )
    parser.add_argument(
        "--spec-no",
        type=int,
        default=653,
        help="Spectrum number to fit",
    )
    parser.add_argument(
        "--usc",
        type=float,
        default=11.0,
        help="Spacecraft potential [V]",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (cuda, cpu, or auto)",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Only run CPU baseline",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("GPU Loss-Cone Fitter Validation")
    print("=" * 60)
    print(f"Data file: {args.input}")
    print(f"Spectrum: {args.spec_no}")
    print(f"USC: {args.usc} V")
    print()

    # Load data
    print("Loading data...")
    er_data = ERData(str(args.input))
    theta_file = str(config.DATA_DIR / config.THETA_FILE)
    pitch_angle = PitchAngle(er_data, theta_file)
    spacecraft_potential = np.full(len(er_data.data), args.usc)

    # Expected baseline from previous run
    expected = {
        "U_surface": -213.7,
        "bs_over_bm": 0.876,
        "beam_amp": 4.704,
        "chi2": 483.53,
    }

    # Run CPU fit
    print("\n" + "-" * 40)
    print("Running CPU fitter...")
    print("-" * 40)
    cpu_result, cpu_time = run_cpu_fit(
        er_data, pitch_angle, spacecraft_potential, args.spec_no
    )
    cpu_U, cpu_bs, cpu_beam, cpu_chi2 = cpu_result

    print(f"  U_surface = {cpu_U:.1f} V")
    print(f"  Bs/Bm     = {cpu_bs:.3f}")
    print(f"  beam_amp  = {cpu_beam:.3f}")
    print(f"  chi2      = {cpu_chi2:.2f}")
    print(f"  Time      = {cpu_time:.3f} s")

    # Check CPU matches expected
    cpu_match = (
        abs(cpu_U - expected["U_surface"]) < 1.0
        and abs(cpu_bs - expected["bs_over_bm"]) < 0.01
        and abs(cpu_beam - expected["beam_amp"]) < 0.1
    )
    print(f"\n  Matches expected baseline: {'YES' if cpu_match else 'NO'}")

    if args.cpu_only:
        return 0

    # Run GPU fit
    print("\n" + "-" * 40)
    print("Running GPU fitter...")
    print("-" * 40)

    try:
        import torch

        if args.device:
            device = args.device
        elif torch.cuda.is_available():
            device = "cuda"
            print(f"  Using CUDA: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            print("  Using Apple MPS")
        else:
            device = "cpu"
            print("  Using CPU (no GPU available)")

        gpu_result, gpu_time = run_gpu_fit(
            er_data, pitch_angle, spacecraft_potential, args.spec_no, device
        )
        gpu_U, gpu_bs, gpu_beam, gpu_chi2 = gpu_result

        print(f"  U_surface = {gpu_U:.1f} V")
        print(f"  Bs/Bm     = {gpu_bs:.3f}")
        print(f"  beam_amp  = {gpu_beam:.3f}")
        print(f"  chi2      = {gpu_chi2:.2f}")
        print(f"  Time      = {gpu_time:.3f} s")

    except ImportError as e:
        print(f"  ERROR: PyTorch not available: {e}")
        return 1
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Compare results
    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)

    # Tolerance for agreement (DE is stochastic, so allow some variance)
    tol_U = 50.0  # V - potential can vary more
    tol_bs = 0.1  # ratio
    tol_beam = 1.0  # amplitude
    tol_chi2_ratio = 2.0  # chi2 within 2x

    diff_U = abs(gpu_U - cpu_U)
    diff_bs = abs(gpu_bs - cpu_bs)
    diff_beam = abs(gpu_beam - cpu_beam)
    chi2_ratio = max(gpu_chi2, cpu_chi2) / max(min(gpu_chi2, cpu_chi2), 1e-6)

    print(f"  ΔU_surface = {diff_U:.1f} V (tol={tol_U})")
    print(f"  ΔBs/Bm     = {diff_bs:.3f} (tol={tol_bs})")
    print(f"  Δbeam_amp  = {diff_beam:.3f} (tol={tol_beam})")
    print(f"  χ² ratio   = {chi2_ratio:.2f} (tol={tol_chi2_ratio})")

    # Speedup
    speedup = cpu_time / max(gpu_time, 1e-6)
    print(f"\n  Speedup    = {speedup:.2f}x")

    # Overall assessment
    match = (
        diff_U < tol_U
        and diff_bs < tol_bs
        and diff_beam < tol_beam
        and chi2_ratio < tol_chi2_ratio
    )

    print("\n" + "=" * 60)
    if match:
        print("VALIDATION PASSED - GPU results within tolerance of CPU")
    else:
        print("VALIDATION FAILED - Results differ beyond tolerance")
    print("=" * 60)

    return 0 if match else 1


if __name__ == "__main__":
    raise SystemExit(main())
