#!/usr/bin/env python3
"""Benchmark the frozen Experiment 006 batched LM candidate optimizer.

Examples:

    uv run python experiments/experiment-006/benchmark_optimizer.py --device cuda:0
    uv run python experiments/experiment-006/benchmark_optimizer.py --device cpu \
        --case-counts 1 --starts 2 --maximum-iterations 3 --output /tmp/lm-smoke.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import beam_model
import beam_torch
import numpy as np
import optimizer_torch
import torch

ENERGY_DOMAIN = (20.0, 2000.0)
PHYSICAL_LOWER = (0.01, 0.01, 90.0, 0.5)
PHYSICAL_UPPER = (2.0, 3.0, 180.0, 45.0)


def build_problem(
    cases: int, starts: int, device: torch.device
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Create deterministic implementation-only cases and multistarts."""
    energies_numpy = np.array(
        [37.673, 65.278, 113.111, 195.994, 339.609, 588.458, 1019.652],
        dtype=np.float64,
    )
    low_numpy, high_numpy = beam_model.channel_edges_ev(energies_numpy)
    pitch_rows = np.linspace(92.0, 179.0, 24, dtype=np.float64)
    row_ids_numpy = np.repeat(np.arange(len(energies_numpy)), len(pitch_rows))
    pitches_numpy = np.tile(pitch_rows, len(energies_numpy))

    generator = np.random.default_rng(6106)
    truth = np.empty((cases, 15), dtype=np.float64)
    truth[:, 0] = generator.uniform(5.0, 14.0, cases)
    truth[:, 1] = np.exp(generator.uniform(np.log(110.0), np.log(480.0), cases))
    truth[:, 2] = generator.uniform(14.0, 38.0, cases)
    truth[:, 3:] = generator.uniform(-0.8, 0.8, (cases, 12))

    start_values = np.repeat(truth[:, None, :], starts, axis=1)
    center_factors = np.linspace(0.88, 1.12, starts)
    sigma_offsets = np.linspace(-2.0, 2.0, starts)
    amplitude_factors = np.linspace(0.9, 1.1, starts)[::-1]
    start_values[:, :, 0] *= amplitude_factors[None, :]
    start_values[:, :, 1] *= center_factors[None, :]
    start_values[:, :, 2] += sigma_offsets[None, :]
    coefficient_offsets = generator.normal(0.0, 0.04, (cases, starts, 12))
    start_values[:, :, 3:] += coefficient_offsets

    def repeated(values: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float64, device=device)
        return tensor.unsqueeze(0).expand(cases, *tensor.shape).contiguous()

    energies = repeated(energies_numpy)
    low = repeated(low_numpy)
    high = repeated(high_numpy)
    row_ids = torch.as_tensor(row_ids_numpy, dtype=torch.long, device=device)
    row_ids = row_ids.unsqueeze(0).expand(cases, -1).contiguous()
    pitches = repeated(pitches_numpy)
    truth_tensor = torch.as_tensor(truth, dtype=torch.float64, device=device)
    row_mask = torch.ones_like(energies, dtype=torch.bool)
    sample_mask = torch.ones_like(row_ids, dtype=torch.bool)
    with torch.inference_mode():
        observed = beam_torch.model_values(
            energies,
            low,
            high,
            row_ids,
            pitches,
            truth_tensor,
            energy_domain_ev=ENERGY_DOMAIN,
            fwhm_fraction=0.48,
            physical_lower=PHYSICAL_LOWER,
            physical_upper=PHYSICAL_UPPER,
            row_valid_mask=row_mask,
            sample_valid_mask=sample_mask,
        )
    initial = torch.as_tensor(start_values, dtype=torch.float64, device=device)
    problem = (
        energies,
        low,
        high,
        row_ids,
        pitches,
        observed,
        initial,
        row_mask,
        sample_mask,
    )
    return problem, truth_tensor


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_one(
    cases: int,
    starts: int,
    device: torch.device,
    maximum_iterations: int,
) -> dict[str, Any]:
    """Time one complete deterministic optimizer batch."""
    problem, truth = build_problem(cases, starts, device)
    energies, low, high, row_ids, pitches, observed, initial, rows, samples = problem
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    result = optimizer_torch.projected_levenberg_marquardt(
        energies,
        low,
        high,
        row_ids,
        pitches,
        observed,
        initial,
        row_valid_mask=rows,
        sample_valid_mask=samples,
        energy_domain_ev=ENERGY_DOMAIN,
        fwhm_fraction=0.48,
        physical_lower=PHYSICAL_LOWER,
        physical_upper=PHYSICAL_UPPER,
        max_iterations=maximum_iterations,
    )
    synchronize(device)
    elapsed = time.perf_counter() - started
    response_fwhm = 0.48 * truth[:, 1, None]
    center_error = (result.parameters[:, :, 1] - truth[:, 1, None]).abs()
    record: dict[str, Any] = {
        "device": str(device),
        "cases": cases,
        "starts_per_case": starts,
        "trajectories": cases * starts,
        "elapsed_seconds": elapsed,
        "trajectories_per_second": cases * starts / elapsed,
        "converged_fraction": float(result.converged.double().mean().cpu()),
        "median_iterations": float(
            result.diagnostics.iterations.double().median().cpu()
        ),
        "maximum_iterations_used": int(result.diagnostics.iterations.max().cpu()),
        "median_center_error_response_fwhm": float(
            (center_error / response_fwhm).median().cpu()
        ),
        "maximum_center_error_response_fwhm": float(
            (center_error / response_fwhm).max().cpu()
        ),
        "median_objective": float(result.objectives.median().cpu()),
        "maximum_objective": float(result.objectives.max().cpu()),
    }
    if device.type == "cuda":
        record["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
    return record


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    results = [
        run_one(cases, args.starts, device, args.maximum_iterations)
        for cases in args.case_counts
    ]
    return {
        "protocol": "experiment-006-d1-optimizer-benchmark-1",
        "implementation_only": True,
        "confirmation_data_opened": False,
        "seed": 6106,
        "maximum_iterations": args.maximum_iterations,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
        },
        "hardware": {
            "requested_device": str(device),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--case-counts", nargs="+", type=int, default=[1, 4, 8, 16, 21])
    parser.add_argument("--starts", type=int, default=12)
    parser.add_argument("--maximum-iterations", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("d1-optimizer-benchmark.json"),
    )
    args = parser.parse_args()
    result = run_benchmark(args)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
