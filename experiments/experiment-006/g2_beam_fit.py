"""Batched G2 differential-evolution beam fitting with CPU-polished winners."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import beam_torch
import fit_reference
import g2_de_torch
import numpy as np
import start_parameters
import torch
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class PackedObservations:
    """Padded observations and the corresponding frozen starts."""

    energies_ev: Tensor
    channel_low_ev: Tensor
    channel_high_ev: Tensor
    row_ids: Tensor
    pitches_deg: Tensor
    observed: Tensor
    row_valid_mask: Tensor
    sample_valid_mask: Tensor
    starts: Tensor


@dataclass(frozen=True)
class G2BatchResult:
    """GPU island endpoints and final CPU-polished results."""

    results: tuple[fit_reference.FitResult | None, ...]
    polish_trajectories: tuple[tuple[fit_reference.StartFit, ...], ...]
    island_parameters: np.ndarray
    island_objectives: np.ndarray
    gpu_seconds: float
    cpu_polish_seconds: float
    evaluations_per_problem: int


def load_configuration(path: Path | None = None) -> dict[str, Any]:
    """Load the prospectively frozen G2 configuration."""
    source = path or Path(__file__).with_name("configuration-g2.json")
    return json.loads(source.read_text(encoding="ascii"))


def root_seed(config: dict[str, Any]) -> int:
    """Map the frozen corpus seed namespace to a deterministic unsigned seed."""
    value = str(config["corpus"]["seed_root"]).encode("ascii")
    return int.from_bytes(hashlib.blake2b(value, digest_size=4).digest(), "little")


def pack_observations(
    observations: Sequence[fit_reference.Observation],
    *,
    device: torch.device | str,
) -> PackedObservations:
    """Pad independent observations and construct the same twelve CPU starts."""
    if not observations:
        raise ValueError("At least one observation is required")
    rows = max(len(observation.energies_ev) for observation in observations)
    samples = max(len(observation.observed) for observation in observations)
    count = len(observations)
    floating = torch.float64
    energies = torch.ones((count, rows), dtype=floating, device=device)
    low = torch.ones_like(energies)
    high = torch.full_like(energies, 1.01)
    row_ids = torch.zeros((count, samples), dtype=torch.long, device=device)
    pitches = torch.full((count, samples), 180.0, dtype=floating, device=device)
    observed = torch.ones_like(pitches)
    row_valid = torch.zeros_like(energies, dtype=torch.bool)
    sample_valid = torch.zeros_like(pitches, dtype=torch.bool)
    starts = []
    lower, upper = fit_reference.parameter_bounds()

    for index, observation in enumerate(observations):
        row_count = len(observation.energies_ev)
        sample_count = len(observation.observed)
        energies[index, :row_count] = torch.tensor(
            observation.energies_ev, dtype=floating, device=device
        )
        low[index, :row_count] = torch.tensor(
            observation.channel_low_ev, dtype=floating, device=device
        )
        high[index, :row_count] = torch.tensor(
            observation.channel_high_ev, dtype=floating, device=device
        )
        row_ids[index, :sample_count] = torch.tensor(
            observation.row_ids, dtype=torch.long, device=device
        )
        pitches[index, :sample_count] = torch.tensor(
            observation.pitches_deg, dtype=floating, device=device
        )
        observed[index, :sample_count] = torch.tensor(
            observation.observed, dtype=floating, device=device
        )
        row_valid[index, :row_count] = True
        sample_valid[index, :sample_count] = True
        starts.append(start_parameters.construct_starts(observation, lower, upper))
    return PackedObservations(
        energies,
        low,
        high,
        row_ids,
        pitches,
        observed,
        row_valid,
        sample_valid,
        torch.as_tensor(np.asarray(starts), dtype=floating, device=device),
    )


def beam_objective(packed: PackedObservations):
    """Return a callable evaluating all problems, islands, and members."""

    def objective(parameters: Tensor) -> Tensor:
        if parameters.ndim != 4 or parameters.shape[0] != packed.energies_ev.shape[0]:
            raise ValueError("Parameters must have shape [problems, islands, pop, 15]")
        problems, islands, population, _ = parameters.shape

        def expand_rows(value: Tensor) -> Tensor:
            return value[:, None, None, :].expand(
                problems, islands, population, value.shape[-1]
            )

        modeled = beam_torch.model_values(
            expand_rows(packed.energies_ev),
            expand_rows(packed.channel_low_ev),
            expand_rows(packed.channel_high_ev),
            expand_rows(packed.row_ids),
            expand_rows(packed.pitches_deg),
            parameters,
            energy_domain_ev=(20.0, 2000.0),
            fwhm_fraction=0.48,
            physical_lower=(0.01, 0.01, 90.0, 0.5),
            physical_upper=(2.0, 3.0, 180.0, 45.0),
            row_valid_mask=expand_rows(packed.row_valid_mask),
            sample_valid_mask=expand_rows(packed.sample_valid_mask),
        )
        return beam_torch.masked_log_objective(
            expand_rows(packed.observed),
            modeled,
            expand_rows(packed.sample_valid_mask),
        )

    return objective


def fit_observations(
    observations: Sequence[fit_reference.Observation],
    problem_keys: Sequence[int],
    *,
    device: torch.device | str = "cuda",
    configuration: dict[str, Any] | None = None,
    cpu_max_nfev: int = 2000,
) -> G2BatchResult:
    """Run global GPU islands and CPU-polish every island winner."""
    config = configuration or load_configuration()
    settings = config["estimator"]
    packed = pack_observations(observations, device=device)
    lower, upper = fit_reference.parameter_bounds()
    if str(settings["gpu_dtype"]) != "float64":
        raise ValueError("G2 scientific fitting requires float64")
    started = time.perf_counter()
    de_result = g2_de_torch.differential_evolution(
        beam_objective(packed),
        packed.starts,
        problem_keys,
        lower,
        upper,
        islands=int(settings["islands"]),
        population=int(settings["population_per_island"]),
        generations=int(settings["maximum_generations"]),
        mutation=tuple(float(value) for value in settings["mutation_range"]),
        crossover=float(settings["crossover_probability"]),
        migration_interval=int(settings["migration_interval_generations"]),
        migration_elites=int(settings["migration_elites_per_island"]),
        root_seed=root_seed(config),
    )
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))
    gpu_seconds = time.perf_counter() - started
    island_parameters = de_result.parameters.detach().cpu().numpy()
    island_objectives = de_result.objectives.detach().cpu().numpy()
    polish_started = time.perf_counter()
    polished = tuple(
        fit_reference._fit_starts(
            observation,
            fit_reference._validate_starts(island_parameters[index]),
            max_nfev=cpu_max_nfev,
        )
        for index, observation in enumerate(observations)
    )
    results = tuple(
        fit_reference._summarize(fits)
        if any(fit.success and np.isfinite(fit.objective) for fit in fits)
        else None
        for fits in polished
    )
    return G2BatchResult(
        results,
        polished,
        island_parameters,
        island_objectives,
        gpu_seconds,
        time.perf_counter() - polish_started,
        de_result.evaluations_per_problem,
    )
