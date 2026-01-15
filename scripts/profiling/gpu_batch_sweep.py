#!/usr/bin/env python3
"""
GPU batch size sweep for optimal VRAM utilization.

Tests different batch sizes for GPU-accelerated components and finds
the optimal configuration for the current hardware.

Usage:
    uv run python scripts/profiling/gpu_batch_sweep.py
    uv run python scripts/profiling/gpu_batch_sweep.py --component losscone
    uv run python scripts/profiling/gpu_batch_sweep.py --component kappa
    uv run python scripts/profiling/gpu_batch_sweep.py --batch-sizes 50,100,150,200
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="Mean of empty slice")

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class SweepResult:
    """Result from a single batch size test."""

    batch_size: int
    throughput: float  # chunks or spectra per second
    time_total: float  # seconds
    vram_peak_mb: float
    vram_allocated_mb: float
    success: bool
    error: str | None = None


def get_gpu_info() -> dict:
    """Get GPU information."""
    if not HAS_TORCH or not torch.cuda.is_available():
        return {"available": False}

    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": props.name,
        "vram_total_mb": props.total_memory / 1024 / 1024,
        "compute_capability": f"{props.major}.{props.minor}",
        "multiprocessors": props.multi_processor_count,
    }


def reset_gpu_memory():
    """Reset GPU memory state."""
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()


def get_vram_stats() -> tuple[float, float]:
    """Get current and peak VRAM usage in MB."""
    if not HAS_TORCH or not torch.cuda.is_available():
        return 0.0, 0.0
    allocated = torch.cuda.memory_allocated() / 1024 / 1024
    peak = torch.cuda.max_memory_allocated() / 1024 / 1024
    return allocated, peak


def sweep_losscone(
    batch_sizes: list[int],
    data_file: Path,
    warmup: bool = True,
) -> list[SweepResult]:
    """
    Sweep batch sizes for LossConeFitterTorch.

    Args:
        batch_sizes: List of batch sizes to test
        data_file: Path to flux data file
        warmup: Run a warmup iteration first

    Returns:
        List of SweepResult for each batch size
    """
    from src import config
    from src.flux import ERData, PitchAngle
    from src.model_torch import LossConeFitterTorch

    print(f"\n{'='*60}")
    print("LossConeFitter Batch Size Sweep")
    print(f"{'='*60}")
    print(f"Data file: {data_file.name}")

    # Load data
    er_data = ERData(str(data_file))
    n_rows = len(er_data.data)
    n_chunks = n_rows // config.SWEEP_ROWS
    print(f"Total chunks: {n_chunks}")

    theta_path = str(config.DATA_DIR / config.THETA_FILE)
    pitch_angle = PitchAngle(er_data)

    results = []

    for batch_size in batch_sizes:
        print(f"\n--- Testing batch_size={batch_size} ---")
        reset_gpu_memory()

        try:
            fitter = LossConeFitterTorch(
                er_data=er_data,
                thetas=theta_path,
                pitch_angle=pitch_angle,
                device="cuda",
            )

            # Warmup run (first run has JIT overhead)
            if warmup and batch_size == batch_sizes[0]:
                print("  Warmup run...")
                _ = fitter.fit_surface_potential_batched(batch_size=min(batch_size, 25))
                reset_gpu_memory()

            # Timed run
            t0 = time.perf_counter()
            _ = fitter.fit_surface_potential_batched(batch_size=batch_size)
            elapsed = time.perf_counter() - t0

            allocated, peak = get_vram_stats()
            throughput = n_chunks / elapsed

            result = SweepResult(
                batch_size=batch_size,
                throughput=throughput,
                time_total=elapsed,
                vram_peak_mb=peak,
                vram_allocated_mb=allocated,
                success=True,
            )
            print(
                f"  Throughput: {throughput:.1f} chunks/s | "
                f"Time: {elapsed:.2f}s | "
                f"VRAM peak: {peak:.0f} MB"
            )

        except torch.cuda.OutOfMemoryError as e:
            result = SweepResult(
                batch_size=batch_size,
                throughput=0,
                time_total=0,
                vram_peak_mb=0,
                vram_allocated_mb=0,
                success=False,
                error=f"OOM: {e}",
            )
            print(f"  OOM at batch_size={batch_size}")
            reset_gpu_memory()

        except Exception as e:
            result = SweepResult(
                batch_size=batch_size,
                throughput=0,
                time_total=0,
                vram_peak_mb=0,
                vram_allocated_mb=0,
                success=False,
                error=str(e),
            )
            print(f"  Error: {e}")

        results.append(result)

        # Clean up
        del fitter
        reset_gpu_memory()

    return results


def sweep_kappa(
    batch_sizes: list[int],
    n_energies: int = 32,
    warmup: bool = True,
) -> list[SweepResult]:
    """
    Sweep batch sizes for KappaFitterTorch.

    For Kappa fitter, batch_size = number of spectra processed at once.

    Args:
        batch_sizes: List of spectra counts to test
        n_energies: Number of energy bins per spectrum
        warmup: Run a warmup iteration first

    Returns:
        List of SweepResult for each batch size
    """
    from src.kappa_torch import KappaFitterTorch

    print(f"\n{'='*60}")
    print("KappaFitter Batch Size Sweep (spectra count)")
    print(f"{'='*60}")
    print(f"Energy bins per spectrum: {n_energies}")

    # Standard LP energy grid
    energies = np.logspace(np.log10(10), np.log10(20000), n_energies)

    results = []

    for n_spectra in batch_sizes:
        print(f"\n--- Testing n_spectra={n_spectra} ---")
        reset_gpu_memory()

        try:
            # Generate synthetic data
            rng = np.random.default_rng(42)
            flux_data = rng.lognormal(10, 2, (n_spectra, n_energies))
            density_estimates = rng.uniform(1e5, 1e7, n_spectra)

            fitter = KappaFitterTorch(device="cuda", dtype="float32")

            # Warmup
            if warmup and n_spectra == batch_sizes[0]:
                print("  Warmup run...")
                small_flux = flux_data[:min(25, n_spectra)]
                small_dens = density_estimates[:min(25, n_spectra)]
                _ = fitter.fit_batch(energies, small_flux, small_dens)
                reset_gpu_memory()

            # Timed run
            t0 = time.perf_counter()
            _ = fitter.fit_batch(energies, flux_data, density_estimates)
            elapsed = time.perf_counter() - t0

            allocated, peak = get_vram_stats()
            throughput = n_spectra / elapsed

            result = SweepResult(
                batch_size=n_spectra,
                throughput=throughput,
                time_total=elapsed,
                vram_peak_mb=peak,
                vram_allocated_mb=allocated,
                success=True,
            )
            print(
                f"  Throughput: {throughput:.1f} spectra/s | "
                f"Time: {elapsed:.2f}s | "
                f"VRAM peak: {peak:.0f} MB"
            )

        except torch.cuda.OutOfMemoryError as e:
            result = SweepResult(
                batch_size=n_spectra,
                throughput=0,
                time_total=0,
                vram_peak_mb=0,
                vram_allocated_mb=0,
                success=False,
                error=f"OOM: {e}",
            )
            print(f"  OOM at n_spectra={n_spectra}")
            reset_gpu_memory()

        except Exception as e:
            result = SweepResult(
                batch_size=n_spectra,
                throughput=0,
                time_total=0,
                vram_peak_mb=0,
                vram_allocated_mb=0,
                success=False,
                error=str(e),
            )
            print(f"  Error: {e}")

        results.append(result)
        reset_gpu_memory()

    return results


def print_summary(
    component: str,
    results: list[SweepResult],
    gpu_info: dict,
):
    """Print sweep summary and recommendations."""
    print(f"\n{'='*60}")
    print(f"SUMMARY: {component}")
    print(f"{'='*60}")

    if gpu_info.get("available"):
        print(f"GPU: {gpu_info['name']}")
        print(f"VRAM Total: {gpu_info['vram_total_mb']:.0f} MB")

    successful = [r for r in results if r.success]
    if not successful:
        print("\nNo successful runs!")
        return

    # Find optimal (highest throughput)
    optimal = max(successful, key=lambda r: r.throughput)

    # Find max batch size before OOM
    max_safe = max(successful, key=lambda r: r.batch_size)

    print(f"\n{'Batch Size':>12} | {'Throughput':>14} | {'Time':>8} | {'VRAM Peak':>10} | Status")
    print("-" * 70)

    for r in results:
        marker = ""
        if r.success:
            if r.batch_size == optimal.batch_size:
                marker = " <-- OPTIMAL"
            elif r.batch_size == max_safe.batch_size and r.batch_size != optimal.batch_size:
                marker = " <-- MAX SAFE"
        status = "OK" if r.success else f"FAIL: {r.error[:20]}..."

        if r.success:
            print(
                f"{r.batch_size:>12} | {r.throughput:>11.1f}/s | {r.time_total:>7.2f}s | "
                f"{r.vram_peak_mb:>8.0f} MB | {status}{marker}"
            )
        else:
            print(f"{r.batch_size:>12} | {'---':>14} | {'---':>8} | {'---':>10} | {status}")

    # Compute formula coefficients
    if len(successful) >= 2 and gpu_info.get("available"):
        vram_mb = gpu_info["vram_total_mb"]
        # Estimate: batch_size = (VRAM_MB * efficiency) / per_chunk_mb
        # We solve for per_chunk_mb using optimal result
        per_chunk_mb = (optimal.vram_peak_mb * 0.8) / optimal.batch_size  # 80% safety margin
        efficiency = 0.66  # Leave 34% headroom

        print(f"\n--- Recommendations ---")
        print(f"Optimal batch_size: {optimal.batch_size}")
        print(f"Peak VRAM at optimal: {optimal.vram_peak_mb:.0f} MB ({optimal.vram_peak_mb/vram_mb*100:.0f}% of total)")
        print(f"Throughput at optimal: {optimal.throughput:.1f}/s")

        if per_chunk_mb > 0:
            print(f"\nEstimated formula for this GPU:")
            print(f"  batch_size = int(VRAM_MB * {efficiency:.2f} / {per_chunk_mb:.1f})")
            print(f"  For {vram_mb:.0f} MB: batch_size = {int(vram_mb * efficiency / per_chunk_mb)}")


def main():
    parser = argparse.ArgumentParser(description="GPU batch size sweep")
    parser.add_argument(
        "--component",
        choices=["losscone", "kappa", "all"],
        default="all",
        help="Which component to sweep",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default=None,
        help="Comma-separated batch sizes to test (e.g., 25,50,100,200)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip warmup iteration",
    )
    args = parser.parse_args()

    if not HAS_TORCH:
        print("ERROR: PyTorch not available")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    gpu_info = get_gpu_info()
    print(f"GPU: {gpu_info['name']}")
    print(f"VRAM: {gpu_info['vram_total_mb']:.0f} MB")
    print(f"Compute Capability: {gpu_info['compute_capability']}")

    # Determine batch sizes to test
    if args.batch_sizes:
        batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
    else:
        # Auto-scale based on VRAM
        vram_mb = gpu_info["vram_total_mb"]
        if vram_mb >= 8000:
            batch_sizes = [50, 100, 200, 300, 400, 500]
        elif vram_mb >= 4000:
            batch_sizes = [25, 50, 100, 150, 200, 250]
        else:
            batch_sizes = [25, 50, 75, 100, 125, 150]

    print(f"Testing batch sizes: {batch_sizes}")

    # Find data file
    from src import config

    data_dir = config.DATA_DIR
    flux_files = sorted(data_dir.glob("**/3D*.TAB"))

    if not flux_files and args.component in ("losscone", "all"):
        print("WARNING: No flux files found for losscone sweep. Run data acquisition first.")
        if args.component == "losscone":
            sys.exit(1)

    warmup = not args.no_warmup

    # Run sweeps
    if args.component in ("losscone", "all") and flux_files:
        losscone_results = sweep_losscone(batch_sizes, flux_files[0], warmup=warmup)
        print_summary("LossConeFitter", losscone_results, gpu_info)

    if args.component in ("kappa", "all"):
        # For kappa, batch_size = number of spectra
        kappa_batch_sizes = [s * 10 for s in batch_sizes]  # Scale up for kappa
        kappa_results = sweep_kappa(kappa_batch_sizes, warmup=warmup)
        print_summary("KappaFitter", kappa_results, gpu_info)


if __name__ == "__main__":
    main()
