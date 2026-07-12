#!/usr/bin/env python3
"""Benchmark Experiment 006 canonical CPU and batched tensor forward models.

Examples:

    uv run python experiments/experiment-006/benchmark_forward.py --device cuda:0
    uv run python experiments/experiment-006/benchmark_forward.py --device cpu \
        --batch-sizes 4 8 --iterations 2 --output /tmp/forward-smoke.json
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
import torch


def make_case_batch(
    batch_size: int, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, ...]:
    """Create deterministic implementation-only cases on the native LP grid."""
    energies_numpy = np.array(
        [
            15927.024,
            9191.748,
            5304.710,
            3061.436,
            1766.805,
            1019.652,
            588.458,
            339.609,
            195.994,
            113.111,
            65.278,
            37.673,
            21.742,
            12.548,
            7.241,
        ],
        dtype=np.float64,
    )
    low_numpy, high_numpy = beam_model.channel_edges_ev(energies_numpy)
    pitch_rows = np.linspace(91.0, 179.0, 44, dtype=np.float64)
    row_ids_numpy = np.repeat(np.arange(len(energies_numpy)), len(pitch_rows))
    pitches_numpy = np.tile(pitch_rows, len(energies_numpy))

    generator = np.random.default_rng(6006)
    parameters_numpy = np.empty((batch_size, 15), dtype=np.float64)
    parameters_numpy[:, 0] = generator.uniform(4.0, 16.0, batch_size)
    parameters_numpy[:, 1] = np.exp(
        generator.uniform(np.log(100.0), np.log(500.0), batch_size)
    )
    parameters_numpy[:, 2] = generator.uniform(12.0, 42.0, batch_size)
    parameters_numpy[:, 3:] = generator.uniform(-2.0, 2.0, (batch_size, 12))

    def repeated(values: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=dtype, device=device)
        return tensor.unsqueeze(0).expand(batch_size, *tensor.shape).contiguous()

    energies = repeated(energies_numpy)
    low = repeated(low_numpy)
    high = repeated(high_numpy)
    row_ids = torch.as_tensor(row_ids_numpy, dtype=torch.long, device=device)
    row_ids = row_ids.unsqueeze(0).expand(batch_size, -1).contiguous()
    pitches = repeated(pitches_numpy)
    parameters = torch.as_tensor(parameters_numpy, dtype=dtype, device=device)
    row_mask = torch.ones_like(energies, dtype=torch.bool)
    sample_mask = torch.ones_like(row_ids, dtype=torch.bool)
    return energies, low, high, row_ids, pitches, parameters, row_mask, sample_mask


def evaluate(batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Evaluate the frozen batched model."""
    energies, low, high, row_ids, pitches, parameters, row_mask, sample_mask = batch
    return beam_torch.model_values(
        energies,
        low,
        high,
        row_ids,
        pitches,
        parameters,
        energy_domain_ev=(20.0, 2000.0),
        fwhm_fraction=0.48,
        physical_lower=(0.01, 0.01, 90.0, 0.5),
        physical_upper=(2.0, 3.0, 180.0, 45.0),
        row_valid_mask=row_mask,
        sample_valid_mask=sample_mask,
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_tensor(
    batch: tuple[torch.Tensor, ...], iterations: int, device: torch.device
) -> dict[str, float]:
    """Time repeated batched forward evaluations after warmup."""
    with torch.inference_mode():
        for _ in range(3):
            evaluate(batch)
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for _ in range(iterations):
            evaluate(batch)
        synchronize(device)
        elapsed = time.perf_counter() - started
    batch_size = batch[0].shape[0]
    result = {
        "elapsed_seconds": elapsed,
        "case_evaluations": batch_size * iterations,
        "case_evaluations_per_second": batch_size * iterations / elapsed,
    }
    if device.type == "cuda":
        result["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
    return result


def reference_case(batch: tuple[torch.Tensor, ...]) -> np.ndarray:
    """Evaluate the first case with the canonical NumPy model."""
    energies, low, high, row_ids, pitches, parameters, *_masks = batch
    return beam_model.model_values(
        energies[0].double().cpu().numpy(),
        low[0].double().cpu().numpy(),
        high[0].double().cpu().numpy(),
        row_ids[0].cpu().numpy(),
        pitches[0].double().cpu().numpy(),
        parameters[0].double().cpu().numpy(),
        energy_domain_ev=(20.0, 2000.0),
        fwhm_fraction=0.48,
        physical_lower=np.array([0.01, 0.01, 90.0, 0.5]),
        physical_upper=np.array([2.0, 3.0, 180.0, 45.0]),
    )


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run the requested deterministic benchmark matrix."""
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    results: list[dict[str, Any]] = []
    for dtype_name in args.dtypes:
        dtype = getattr(torch, dtype_name)
        for batch_size in args.batch_sizes:
            batch = make_case_batch(batch_size, dtype, device)
            modeled = evaluate(batch)
            reference = reference_case(batch)
            represented = modeled[0].double().cpu().detach().numpy()
            relative = np.abs(represented - reference) / np.maximum(
                np.abs(reference), 1e-12
            )
            timing = benchmark_tensor(batch, args.iterations, device)
            results.append(
                {
                    "device": str(device),
                    "dtype": dtype_name,
                    "batch_size": batch_size,
                    "maximum_relative_forward_error": float(np.max(relative)),
                    **timing,
                }
            )
    return {
        "protocol": "experiment-006-d1-forward-benchmark-1",
        "implementation_only": True,
        "confirmation_data_opened": False,
        "seed": 6006,
        "iterations": args.iterations,
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
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=[16, 32, 64, 128, 256]
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=["float32", "float64"],
        default=["float32", "float64"],
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("d1-forward-benchmark.json"),
    )
    args = parser.parse_args()
    result = run_benchmark(args)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
