#!/usr/bin/env python3
"""
Quick test to compare batched vs sequential loss-cone fitting.

Usage:
    uv run python scripts/dev/test_batched_losscone.py
"""

import time
import warnings
from pathlib import Path

import numpy as np

# Suppress nanmean warnings for empty slices
warnings.filterwarnings("ignore", message="Mean of empty slice")

from src import config
from src.flux import ERData, PitchAngle
from src.model_torch import LossConeFitterTorch


def main():
    # Find a test data file
    data_dir = config.DATA_DIR
    flux_files = sorted(data_dir.glob("**/3D*.TAB"))

    if not flux_files:
        print("No flux files found. Run data acquisition first.")
        return

    # Use first file for quick test
    test_file = flux_files[0]
    print(f"Using test file: {test_file.name}")

    # Load data
    er_data = ERData(str(test_file))
    n_rows = len(er_data.data)
    n_chunks = n_rows // config.SWEEP_ROWS
    print(f"Loaded {n_rows} rows, {n_chunks} chunks")

    # Test with all chunks
    test_chunks = n_chunks
    print(f"Testing with {test_chunks} chunks")

    # Create fitter
    theta_path = str(config.DATA_DIR / config.THETA_FILE)
    pitch_angle = PitchAngle(er_data)

    # Use CUDA if available
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    fitter = LossConeFitterTorch(
        er_data=er_data,
        thetas=theta_path,
        pitch_angle=pitch_angle,
        device=device,
    )

    # Test sequential (original method) - just first few chunks for comparison
    seq_test_chunks = min(20, test_chunks)
    print(f"\n--- Sequential fitting (original, {seq_test_chunks} chunks) ---")
    t0 = time.perf_counter()
    results_seq = np.zeros((test_chunks, 5))
    for i in range(seq_test_chunks):
        U, bs, amp, chi2 = fitter._fit_surface_potential_torch(i)
        results_seq[i] = [U, bs, amp, chi2, i]
    t_seq = time.perf_counter() - t0
    print(f"Sequential: {t_seq:.2f}s ({t_seq/seq_test_chunks*1000:.1f}ms/chunk)")
    print(f"Estimated for {test_chunks} chunks: {t_seq/seq_test_chunks*test_chunks:.2f}s")

    # Test batched method with VRAM-aware mega-batching
    print("\n--- Batched fitting (with mega-batching for VRAM) ---")
    t0 = time.perf_counter()

    # Use the batched method (default batch_size=100 is optimal for ~3GB VRAM)
    results_batched = fitter.fit_surface_potential_batched()

    t_batched = time.perf_counter() - t0
    valid_count = int(np.sum(~np.isnan(results_batched[:, 0])))
    print(f"Batched total: {t_batched:.2f}s ({t_batched/valid_count*1000:.1f}ms/chunk, {valid_count} valid)")

    # Compare results
    print("\n--- Comparison ---")

    # Compare valid chunks (only the ones we tested sequentially)
    compare_range = slice(0, seq_test_chunks)
    valid_seq = ~np.isnan(results_seq[compare_range, 0])
    valid_batch = ~np.isnan(results_batched[compare_range, 0])
    both_valid = valid_seq & valid_batch

    if both_valid.any():
        seq_subset = results_seq[compare_range][both_valid]
        batch_subset = results_batched[compare_range][both_valid]
        u_diff = np.abs(seq_subset[:, 0] - batch_subset[:, 0])
        chi2_seq = seq_subset[:, 3]
        chi2_batch = batch_subset[:, 3]

        print(f"Chunks compared: {both_valid.sum()}")
        print(f"U_surface diff: mean={u_diff.mean():.2f}V, max={u_diff.max():.2f}V")
        print(f"Chi2 seq:    mean={chi2_seq.mean():.2f}, min={chi2_seq.min():.2f}")
        print(f"Chi2 batch:  mean={chi2_batch.mean():.2f}, min={chi2_batch.min():.2f}")

    # Compute speedup based on estimated sequential time
    t_seq_estimated = t_seq / seq_test_chunks * test_chunks
    print(f"\nEstimated speedup: {t_seq_estimated/t_batched:.1f}x")


if __name__ == "__main__":
    main()
