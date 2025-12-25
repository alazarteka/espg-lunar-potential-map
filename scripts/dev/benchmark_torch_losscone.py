#!/usr/bin/env python3
"""
Benchmark GPU vs CPU loss-cone fitter on multiple chunks.

Tests real-world performance by fitting a range of measurement chunks.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from src import config
from src.flux import ERData, LossConeFitter, PitchAngle


def benchmark_cpu(
    er_data: ERData,
    pitch_angle: PitchAngle,
    spacecraft_potential: np.ndarray,
    n_chunks: int,
) -> tuple[np.ndarray, float]:
    """Benchmark CPU fitter on n_chunks."""
    fitter = LossConeFitter(
        er_data,
        str(config.DATA_DIR / config.THETA_FILE),
        pitch_angle,
        spacecraft_potential,
        normalization_mode="ratio",
        incident_flux_stat="mean",
    )

    results = []
    start = time.perf_counter()

    for chunk_idx in range(n_chunks):
        result = fitter._fit_surface_potential(chunk_idx)
        results.append(result)

    elapsed = time.perf_counter() - start
    return np.array(results), elapsed


def benchmark_gpu(
    er_data: ERData,
    pitch_angle: PitchAngle,
    spacecraft_potential: np.ndarray,
    n_chunks: int,
    device: str,
) -> tuple[np.ndarray, float]:
    """Benchmark GPU fitter on n_chunks."""
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

    # Warm-up
    _ = fitter._fit_surface_potential_torch(0)

    results = []
    start = time.perf_counter()

    for chunk_idx in range(n_chunks):
        result = fitter._fit_surface_potential_torch(chunk_idx)
        results.append(result)

    elapsed = time.perf_counter() - start
    return np.array(results), elapsed


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark GPU vs CPU loss-cone fitter")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/1999/091_120APR/3D990429.TAB"),
        help="ER data file",
    )
    parser.add_argument(
        "--n-chunks",
        type=int,
        default=20,
        help="Number of chunks to benchmark",
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
        default="cpu",
        help="Torch device (cuda, cpu)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("GPU Loss-Cone Fitter Benchmark")
    print("=" * 60)
    print(f"Data file: {args.input}")
    print(f"Chunks: {args.n_chunks}")
    print(f"USC: {args.usc} V")
    print(f"Device: {args.device}")
    print()

    # Load data
    print("Loading data...")
    er_data = ERData(str(args.input))
    theta_file = str(config.DATA_DIR / config.THETA_FILE)
    pitch_angle = PitchAngle(er_data, theta_file)
    spacecraft_potential = np.full(len(er_data.data), args.usc)

    max_chunks = len(er_data.data) // config.SWEEP_ROWS
    n_chunks = min(args.n_chunks, max_chunks)
    print(f"Available chunks: {max_chunks}, benchmarking: {n_chunks}")

    # CPU benchmark
    print("\n" + "-" * 40)
    print("CPU Benchmark (scipy differential_evolution)")
    print("-" * 40)
    cpu_results, cpu_time = benchmark_cpu(
        er_data, pitch_angle, spacecraft_potential, n_chunks
    )
    cpu_rate = n_chunks / cpu_time
    print(f"  Total time: {cpu_time:.2f} s")
    print(f"  Rate: {cpu_rate:.2f} chunks/s")
    print(f"  Per chunk: {cpu_time / n_chunks * 1000:.1f} ms")

    # GPU benchmark
    print("\n" + "-" * 40)
    print(f"GPU Benchmark (torch on {args.device})")
    print("-" * 40)
    try:
        gpu_results, gpu_time = benchmark_gpu(
            er_data, pitch_angle, spacecraft_potential, n_chunks, args.device
        )
        gpu_rate = n_chunks / gpu_time
        print(f"  Total time: {gpu_time:.2f} s")
        print(f"  Rate: {gpu_rate:.2f} chunks/s")
        print(f"  Per chunk: {gpu_time / n_chunks * 1000:.1f} ms")
    except Exception as e:
        print(f"  ERROR: {e}")
        return 1

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    speedup = cpu_time / gpu_time
    print(f"  CPU: {cpu_time:.2f} s ({cpu_rate:.2f} chunks/s)")
    print(f"  GPU: {gpu_time:.2f} s ({gpu_rate:.2f} chunks/s)")
    print(f"  Speedup: {speedup:.2f}x")

    # Compare chi2 values
    cpu_chi2 = cpu_results[:, 3]
    gpu_chi2 = gpu_results[:, 3]
    # Filter out NaN, Inf, and extreme values (penalty values)
    valid_mask = (
        np.isfinite(cpu_chi2) & np.isfinite(gpu_chi2) &
        (cpu_chi2 < 1e10) & (gpu_chi2 < 1e10)
    )

    if valid_mask.sum() >= 2:
        chi2_corr = np.corrcoef(cpu_chi2[valid_mask], gpu_chi2[valid_mask])[0, 1]
        mean_chi2_diff = np.mean(np.abs(cpu_chi2[valid_mask] - gpu_chi2[valid_mask]))
        mean_cpu_chi2 = np.mean(cpu_chi2[valid_mask])
        mean_gpu_chi2 = np.mean(gpu_chi2[valid_mask])
        print(f"\n  Valid fits: {valid_mask.sum()}/{n_chunks}")
        print(f"  Mean CPU χ²: {mean_cpu_chi2:.2f}")
        print(f"  Mean GPU χ²: {mean_gpu_chi2:.2f}")
        print(f"  Mean |Δχ²|: {mean_chi2_diff:.2f}")
        print(f"  χ² correlation: {chi2_corr:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
