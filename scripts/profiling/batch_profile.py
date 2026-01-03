#!/usr/bin/env python3
"""
Profile a single batch of GPU loss-cone fitting.

Measures time breakdown across phases and functions to identify bottlenecks.

Usage:
    uv run python scripts/profiling/batch_profile.py
    uv run python scripts/profiling/batch_profile.py --batch-size 400
"""

from __future__ import annotations

import argparse
import gc
import sys
import warnings
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

warnings.filterwarnings("ignore", message="Mean of empty slice")

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class TimingStats:
    """Accumulated timing statistics."""

    total_ms: float = 0.0
    call_count: int = 0
    children: dict[str, "TimingStats"] = field(default_factory=dict)

    def add(self, ms: float):
        self.total_ms += ms
        self.call_count += 1


class GPUProfiler:
    """GPU-aware profiler using CUDA events."""

    def __init__(self):
        self.stats: dict[str, TimingStats] = defaultdict(TimingStats)
        self.stack: list[str] = []
        self._de_iterations = 0

    def reset(self):
        self.stats = defaultdict(TimingStats)
        self.stack = []
        self._de_iterations = 0

    @contextmanager
    def region(self, name: str):
        """Time a code region using CUDA events."""
        if not HAS_TORCH or not torch.cuda.is_available():
            yield
            return

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start.record()
        self.stack.append(name)

        try:
            yield
        finally:
            end.record()
            torch.cuda.synchronize()

            elapsed_ms = start.elapsed_time(end)
            self.stats[name].add(elapsed_ms)
            self.stack.pop()

    def record_de_iterations(self, n_iter: int):
        """Record DE iteration count."""
        self._de_iterations = n_iter

    def get_de_iterations(self) -> int:
        return self._de_iterations


# Global profiler instance
_profiler = GPUProfiler()


def wrap_function(module, func_name: str, profile_name: str):
    """Wrap a function with profiling."""
    original = getattr(module, func_name)

    def wrapped(*args, **kwargs):
        with _profiler.region(profile_name):
            return original(*args, **kwargs)

    setattr(module, func_name, wrapped)
    return original  # Return original for restoration


def profile_single_batch(batch_size: int, data_file) -> dict:
    """
    Profile a single batch of loss-cone fitting.

    Returns timing breakdown dictionary.
    """
    from src import config
    from src.flux import ERData, PitchAngle
    from src import model_torch
    from src.model_torch import (
        LossConeFitterTorch,
        synth_losscone_multi_chunk_torch,
        compute_chi2_multi_chunk_torch,
    )

    # Reset profiler and GPU memory
    _profiler.reset()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    # Load data
    er_data = ERData(str(data_file))
    n_rows = len(er_data.data)
    n_chunks = n_rows // config.SWEEP_ROWS
    theta_path = str(config.DATA_DIR / config.THETA_FILE)
    pitch_angle = PitchAngle(er_data, theta_path)

    # Create fitter
    fitter = LossConeFitterTorch(
        er_data=er_data,
        thetas=theta_path,
        pitch_angle=pitch_angle,
        device="cuda",
    )

    # Wrap hot functions for profiling
    originals = {}
    originals["synth"] = wrap_function(
        model_torch, "synth_losscone_multi_chunk_torch", "physics_model"
    )
    originals["chi2"] = wrap_function(
        model_torch, "compute_chi2_multi_chunk_torch", "chi2_compute"
    )

    # Also wrap the phase methods on the fitter instance
    original_lhs = fitter._fit_batch_lhs
    original_de = fitter._fit_batch_de
    original_precompute = fitter._precompute_chunk_data

    def wrapped_precompute(*args, **kwargs):
        with _profiler.region("data_staging"):
            return original_precompute(*args, **kwargs)

    def wrapped_lhs(*args, **kwargs):
        with _profiler.region("lhs_phase"):
            return original_lhs(*args, **kwargs)

    def wrapped_de(*args, **kwargs):
        with _profiler.region("de_phase"):
            result = original_de(*args, **kwargs)
            # result is (best_params, best_chi2, n_iter)
            if len(result) == 3:
                _profiler.record_de_iterations(result[2])
            return result

    fitter._precompute_chunk_data = wrapped_precompute
    fitter._fit_batch_lhs = wrapped_lhs
    fitter._fit_batch_de = wrapped_de

    # Warmup run (JIT compilation)
    print("Warmup run...")
    _ = fitter.fit_surface_potential_batched(batch_size=min(25, batch_size))
    _profiler.reset()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Profile a single batch
    print(f"Profiling batch_size={batch_size}...")
    chunk_indices = list(range(min(batch_size, n_chunks)))

    with _profiler.region("total"):
        # Data staging
        energies, pitches, norm2d, mask, sc_pot, valid_indices = (
            fitter._precompute_chunk_data(chunk_indices)
        )

        if len(valid_indices) > 0:
            # LHS phase
            lhs_params, lhs_chi2 = fitter._fit_batch_lhs(
                energies, pitches, norm2d, mask, sc_pot, n_lhs=400
            )

            # DE phase
            de_params, de_chi2, n_iter = fitter._fit_batch_de(
                energies, pitches, norm2d, mask, sc_pot, x0=lhs_params
            )

    # Get VRAM stats
    vram_peak = torch.cuda.max_memory_allocated() / 1024 / 1024

    # Restore original functions
    model_torch.synth_losscone_multi_chunk_torch = originals["synth"]
    model_torch.compute_chi2_multi_chunk_torch = originals["chi2"]

    return {
        "batch_size": batch_size,
        "n_chunks_processed": len(valid_indices),
        "stats": dict(_profiler.stats),
        "de_iterations": _profiler.get_de_iterations(),
        "vram_peak_mb": vram_peak,
    }


def print_results(results: dict):
    """Print profiling results as a formatted table."""
    stats = results["stats"]
    total_ms = stats.get("total", TimingStats()).total_ms

    if total_ms == 0:
        print("No timing data collected!")
        return

    print(f"\n{'='*60}")
    print(f"Single Batch Profile (batch_size={results['batch_size']})")
    print(f"{'='*60}")
    print(f"Chunks processed: {results['n_chunks_processed']}")
    print(f"DE iterations: {results['de_iterations']}/500", end="")
    if results["de_iterations"] < 500:
        print(" (converged early)")
    else:
        print(" (hit max)")
    print(f"VRAM peak: {results['vram_peak_mb']:.0f} MB")

    print(f"\n{'Phase':<25} {'Time (ms)':>12} {'%Total':>10} {'Calls':>8}")
    print("-" * 60)

    # Helper to print a row
    def print_row(name: str, key: str, indent: int = 0):
        s = stats.get(key, TimingStats())
        prefix = "  " * indent + ("└─ " if indent > 0 else "")
        pct = (s.total_ms / total_ms * 100) if total_ms > 0 else 0
        calls = s.call_count if s.call_count > 0 else ""
        print(f"{prefix}{name:<{25-len(prefix)}} {s.total_ms:>12.1f} {pct:>9.1f}% {calls:>8}")

    # Top-level phases
    print_row("Data staging", "data_staging")
    print_row("LHS phase", "lhs_phase")

    # LHS breakdown - estimate from physics/chi2 calls during LHS
    # We need to calculate LHS-specific physics time
    lhs_time = stats.get("lhs_phase", TimingStats()).total_ms
    de_time = stats.get("de_phase", TimingStats()).total_ms
    physics_time = stats.get("physics_model", TimingStats()).total_ms
    chi2_time = stats.get("chi2_compute", TimingStats()).total_ms
    physics_calls = stats.get("physics_model", TimingStats()).call_count
    chi2_calls = stats.get("chi2_compute", TimingStats()).call_count

    # Estimate LHS physics time (1 call during LHS, rest during DE)
    if physics_calls > 1:
        lhs_physics_time = physics_time / physics_calls  # ~1 call
        de_physics_time = physics_time - lhs_physics_time
    else:
        lhs_physics_time = physics_time
        de_physics_time = 0

    if chi2_calls > 1:
        lhs_chi2_time = chi2_time / chi2_calls
        de_chi2_time = chi2_time - lhs_chi2_time
    else:
        lhs_chi2_time = chi2_time
        de_chi2_time = 0

    # Print LHS breakdown
    lhs_physics_pct = (lhs_physics_time / total_ms * 100) if total_ms > 0 else 0
    lhs_chi2_pct = (lhs_chi2_time / total_ms * 100) if total_ms > 0 else 0
    print(f"  └─ {'Physics':<21} {lhs_physics_time:>12.1f} {lhs_physics_pct:>9.1f}%        1")
    print(f"  └─ {'Chi2':<21} {lhs_chi2_time:>12.1f} {lhs_chi2_pct:>9.1f}%        1")

    print_row("DE phase", "de_phase")

    # DE breakdown
    de_physics_pct = (de_physics_time / total_ms * 100) if total_ms > 0 else 0
    de_chi2_pct = (de_chi2_time / total_ms * 100) if total_ms > 0 else 0
    de_physics_calls = max(0, physics_calls - 1)
    de_chi2_calls = max(0, chi2_calls - 1)
    de_overhead_time = de_time - de_physics_time - de_chi2_time
    de_overhead_pct = (de_overhead_time / total_ms * 100) if total_ms > 0 else 0

    print(f"  └─ {'Physics':<21} {de_physics_time:>12.1f} {de_physics_pct:>9.1f}% {de_physics_calls:>8}")
    print(f"  └─ {'Chi2':<21} {de_chi2_time:>12.1f} {de_chi2_pct:>9.1f}% {de_chi2_calls:>8}")
    print(f"  └─ {'DE overhead':<21} {de_overhead_time:>12.1f} {de_overhead_pct:>9.1f}%")

    print("-" * 60)
    print_row("TOTAL", "total")

    # Summary
    print(f"\n--- Summary ---")
    print(f"Physics model: {physics_time:.1f} ms ({physics_time/total_ms*100:.1f}%) - {physics_calls} calls")
    print(f"Chi2 compute:  {chi2_time:.1f} ms ({chi2_time/total_ms*100:.1f}%) - {chi2_calls} calls")
    print(f"Other:         {total_ms - physics_time - chi2_time:.1f} ms ({(total_ms - physics_time - chi2_time)/total_ms*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Profile single batch GPU fitting")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=400,
        help="Batch size to profile (default: 400)",
    )
    args = parser.parse_args()

    if not HAS_TORCH:
        print("ERROR: PyTorch not available")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    # Find data file
    from src import config

    data_dir = config.DATA_DIR
    flux_files = sorted(data_dir.glob("**/3D*.TAB"))

    if not flux_files:
        print("ERROR: No flux files found. Run data acquisition first.")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Data file: {flux_files[0].name}")

    results = profile_single_batch(args.batch_size, flux_files[0])
    print_results(results)


if __name__ == "__main__":
    main()
