#!/usr/bin/env python3
"""Run the frozen Experiment 006 512-case basin-certification corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import beam_model
import fit_reference
import numpy as np
import optimizer_torch
import start_parameters
import support_data
import torch

N_STARTS = 12
N_PARAMETERS = 15


@dataclass(frozen=True)
class CertificationCase:
    """One deterministic implementation-only certification case."""

    case_id: int
    arm: str
    challenge: str
    support_spec_no: int
    true_center_ev: float
    true_amplitude: float
    true_pitch_sigma_deg: float
    center_identifiable: bool
    observation: fit_reference.Observation


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_configuration(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="ascii"))


def observation_support(
    support: support_data.GeometrySupport,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return reflected geometry samples over the frozen energy domain."""
    valid_energy = (support.energies_ev >= 20.0) & (support.energies_ev <= 2000.0)
    indices = np.flatnonzero(valid_energy)
    row_ids: list[np.ndarray] = []
    pitches: list[np.ndarray] = []
    for row_id, source_index in enumerate(indices):
        row_pitch = support.pitches_deg[source_index]
        valid = np.isfinite(row_pitch) & (row_pitch >= 90.0) & (row_pitch <= 180.0)
        row_ids.append(np.full(np.count_nonzero(valid), row_id, dtype=np.int64))
        pitches.append(row_pitch[valid])
    return (
        support.energies_ev[indices],
        support.channel_low_ev[indices],
        support.channel_high_ev[indices],
        np.concatenate(row_ids),
        np.concatenate(pitches),
    )


def _modeled(
    observation: fit_reference.Observation, parameters: np.ndarray
) -> np.ndarray:
    return beam_model.model_values(
        observation.energies_ev,
        observation.channel_low_ev,
        observation.channel_high_ev,
        observation.row_ids,
        observation.pitches_deg,
        parameters,
        energy_domain_ev=observation.energy_domain_ev,
        fwhm_fraction=observation.fwhm_fraction,
        physical_lower=np.asarray(observation.physical_lower),
        physical_upper=np.asarray(observation.physical_upper),
    )


def _base_case(
    support: support_data.GeometrySupport,
    rng: np.random.Generator,
    center: float,
    amplitude: float,
    sigma: float,
) -> tuple[fit_reference.Observation, np.ndarray]:
    energies, low, high, row_ids, pitches = observation_support(support)
    truth = np.r_[amplitude, center, sigma, rng.uniform(-0.8, 0.8, 12)]
    blank = fit_reference.Observation(
        energies, low, high, row_ids, pitches, np.ones(pitches.size)
    )
    return replace(blank, observed=_modeled(blank, truth)), truth


def _apply_challenge(
    observation: fit_reference.Observation,
    truth: np.ndarray,
    challenge: str,
    configuration: dict[str, Any],
) -> tuple[fit_reference.Observation, bool]:
    """Apply one frozen discrepancy/adversarial transformation."""
    corpus = configuration["corpus"]
    values = observation.observed.copy()
    row_ids = observation.row_ids
    energies = observation.energies_ev[row_ids]
    pitches = observation.pitches_deg
    identifiable = challenge not in {
        "null",
        "flat_pitch",
        "single_energy",
        "duplicate_samples",
        "sparse_pitch",
        "zero_dynamic_range",
    }
    if challenge == "correlated_log_residual":
        phase = 1.3 * np.log(energies / 100.0) + np.deg2rad(pitches)
        values *= np.exp(
            float(corpus["correlated_log_residual_amplitude"]) * np.sin(phase)
        )
    elif challenge in {"asymmetric_energy_tail", "broad_secondary", "narrow_secondary"}:
        secondary = truth.copy()
        secondary[0] *= float(corpus["asymmetric_tail_fraction"])
        secondary[1] *= float(corpus["asymmetric_tail_center_factor"])
        if challenge == "broad_secondary":
            secondary[2] = min(50.0, truth[2] * 1.7)
        elif challenge == "narrow_secondary":
            secondary[2] = max(3.0, truth[2] * 0.55)
        baseline = truth.copy()
        baseline[0] = 0.0
        values += _modeled(observation, secondary) - _modeled(observation, baseline)
    elif challenge == "energy_tilt":
        values *= np.exp(0.025 * np.log(energies / np.median(energies)))
    elif challenge == "null":
        no_beam = truth.copy()
        no_beam[0] = 0.0
        values = _modeled(observation, no_beam)
    elif challenge == "near_null":
        weak = truth.copy()
        weak[0] = 1e-5
        values = _modeled(observation, weak)
    elif challenge == "flat_pitch":
        values = np.repeat(np.median(values), values.size)
    elif challenge == "zero_dynamic_range":
        values = np.ones_like(values)
    elif challenge == "bound_center":
        edge = truth.copy()
        edge[1] = 20.0
        values = _modeled(observation, edge)
    elif challenge in {"single_energy", "duplicate_samples", "sparse_pitch"}:
        if challenge == "single_energy":
            keep = row_ids == row_ids[len(row_ids) // 2]
        elif challenge == "sparse_pitch":
            keep = np.zeros(values.size, dtype=bool)
            for row in np.unique(row_ids):
                keep[np.flatnonzero(row_ids == row)[:2]] = True
        else:
            keep = np.arange(values.size) < max(16, values.size // 5)
        selected_rows = row_ids[keep]
        unique_rows, remapped = np.unique(selected_rows, return_inverse=True)
        observation = replace(
            observation,
            energies_ev=observation.energies_ev[unique_rows],
            channel_low_ev=observation.channel_low_ev[unique_rows],
            channel_high_ev=observation.channel_high_ev[unique_rows],
            row_ids=remapped,
            pitches_deg=pitches[keep],
        )
        values = values[keep]
        if challenge == "duplicate_samples":
            observation = replace(
                observation,
                row_ids=np.tile(observation.row_ids, 2),
                pitches_deg=np.tile(observation.pitches_deg, 2),
            )
            values = np.tile(values, 2)
    return replace(
        observation, observed=np.maximum(values, np.finfo(float).tiny)
    ), identifiable


def make_cases(
    supports: list[support_data.GeometrySupport], configuration: dict[str, Any]
) -> list[CertificationCase]:
    """Enumerate the independently reviewed frozen 512-case corpus."""
    corpus = configuration["corpus"]
    if len(supports) != int(corpus["support_count"]):
        raise ValueError("The frozen corpus requires exactly 12 calibration supports")
    rng = np.random.default_rng(int(corpus["seed"]))
    anchors = np.asarray(corpus["beam_energy_anchors_ev"], dtype=float)
    amplitudes = np.asarray(corpus["amplitudes"], dtype=float)
    sigmas = np.asarray(corpus["pitch_sigma_deg"], dtype=float)
    cases: list[CertificationCase] = []

    def append(arm: str, challenge: str, ordinal: int, support_index: int) -> None:
        support = supports[support_index % len(supports)]
        center = float(anchors[ordinal % anchors.size])
        amplitude = float(amplitudes[(ordinal // anchors.size) % amplitudes.size])
        sigma = float(sigmas[(ordinal + 2 * support_index) % sigmas.size])
        observation, truth = _base_case(support, rng, center, amplitude, sigma)
        observation, identifiable = _apply_challenge(
            observation, truth, challenge, configuration
        )
        cases.append(
            CertificationCase(
                len(cases),
                arm,
                challenge,
                support.spec_no,
                center,
                amplitude,
                sigma,
                identifiable,
                observation,
            )
        )

    for support_index in range(12):
        for ordinal in range(8):
            append("analytic_matched", "matched", ordinal, support_index)
    for challenge in corpus["generic_challenges"]:
        for ordinal in range(32):
            append("generic_discrepancy", str(challenge), ordinal, ordinal)
    for challenge in corpus["null_adversarial_challenges"]:
        for ordinal in range(8):
            append("null_adversarial", str(challenge), ordinal, ordinal)
    calibration_challenges = ("matched", *corpus["generic_challenges"])
    for support_index in range(12):
        for ordinal in range(16):
            challenge = str(
                calibration_challenges[ordinal % len(calibration_challenges)]
            )
            append("calibration_support", challenge, ordinal, support_index)

    expected = {str(key): int(value) for key, value in corpus["arms"].items()}
    actual = {arm: sum(case.arm == arm for case in cases) for arm in expected}
    if len(cases) != int(corpus["total_cases"]) or actual != expected:
        raise RuntimeError(f"Frozen corpus quota error: {actual}")
    return cases


def canonical_objectives(
    observation: fit_reference.Observation, candidates: np.ndarray
) -> np.ndarray:
    """Evaluate candidate objectives with the canonical NumPy model."""
    objectives = np.full(len(candidates), np.inf)
    for index, parameters in enumerate(np.asarray(candidates, dtype=float)):
        if np.isfinite(parameters).all():
            residual = beam_model.log_residuals(
                observation.observed, _modeled(observation, parameters)
            )
            objectives[index] = float(residual @ residual)
    return objectives


def canonical_ranking(objectives: np.ndarray) -> np.ndarray:
    """Return stable finite-first candidate ordering."""
    objectives = np.asarray(objectives, dtype=float)
    values = np.where(np.isfinite(objectives), objectives, np.inf)
    return np.argsort(values, kind="stable")


def eligible_ranking(objectives: np.ndarray, converged: np.ndarray) -> np.ndarray:
    """Rank converged finite candidates before all rejected candidates."""
    objectives = np.asarray(objectives, dtype=float)
    eligible = np.isfinite(objectives) & np.asarray(converged, dtype=bool)
    values = np.where(eligible, objectives, np.inf)
    return np.argsort(values, kind="stable")


def log_model_rms(
    observation: fit_reference.Observation,
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    first_model = np.maximum(_modeled(observation, first), np.finfo(float).tiny)
    second_model = np.maximum(_modeled(observation, second), np.finfo(float).tiny)
    return float(
        np.sqrt(np.mean(np.square(np.log(first_model) - np.log(second_model))))
    )


def basin_matches(
    observation: fit_reference.Observation,
    reference: fit_reference.FitResult,
    polished: fit_reference.FitResult,
    center_identifiable: bool,
    configuration: dict[str, Any],
) -> np.ndarray:
    """Identify polished candidates in the reference objective/model basin."""
    gate = configuration["basin_match"]
    identifiable = (
        center_identifiable
        and reference.jacobian_rank >= int(gate["identifiable_rank"])
        and reference.parameters[0] > float(gate["identifiable_amplitude_min"])
    )
    scale = max(abs(reference.objective), 1e-12)
    matches = []
    for fitted in polished.fits:
        objective_match = abs(fitted.objective - reference.objective) / scale <= float(
            gate["cpu_polished_objective_relative_difference_max"]
        )
        model_match = log_model_rms(
            observation, reference.parameters, fitted.parameters
        ) <= float(gate["log_model_rms_max"])
        center_match = not identifiable or abs(
            fitted.parameters[1] - reference.parameters[1]
        ) / (0.48 * max(abs(reference.parameters[1]), 1.0)) <= float(
            gate["cpu_polished_center_difference_response_fwhm_max"]
        )
        matches.append(objective_match and model_match and center_match)
    return np.asarray(matches, dtype=bool)


def gpu_candidates(
    case: CertificationCase, starts: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run all 12 practical starts as one float64 GPU batch."""
    observation = case.observation

    def floating(value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float64, device=device).unsqueeze(0)

    result = optimizer_torch.projected_levenberg_marquardt(
        floating(observation.energies_ev),
        floating(observation.channel_low_ev),
        floating(observation.channel_high_ev),
        torch.as_tensor(observation.row_ids, dtype=torch.long, device=device).unsqueeze(
            0
        ),
        floating(observation.pitches_deg),
        floating(observation.observed),
        torch.as_tensor(starts, dtype=torch.float64, device=device).unsqueeze(0),
        max_iterations=100,
    )
    return (
        result.parameters[0].detach().cpu().numpy(),
        result.objectives[0].detach().cpu().numpy(),
        result.converged[0].detach().cpu().numpy(),
    )


def _json(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return json.dumps(value, separators=(",", ":"), allow_nan=True)


def run_case(
    case: CertificationCase, configuration: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    """Retain the complete CPU, GPU, ranking, polish, and hybrid record."""
    timers: dict[str, float] = {}
    started = time.perf_counter()
    starts = start_parameters.construct_starts(
        case.observation, *fit_reference.parameter_bounds()
    )
    if starts.shape != (N_STARTS, N_PARAMETERS):
        raise RuntimeError("Committed practical constructor did not return 12 starts")
    timers["starts_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    reference = fit_reference.fit_reference(case.observation, starts)
    timers["reference_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    candidates, gpu_objectives, converged = gpu_candidates(case, starts, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timers["gpu_seconds"] = time.perf_counter() - started
    numpy_objectives = canonical_objectives(case.observation, candidates)
    ranking = eligible_ranking(numpy_objectives, converged)
    started = time.perf_counter()
    polished = fit_reference.fit_reference(case.observation, candidates)
    timers["all_gpu_polish_seconds"] = time.perf_counter() - started
    matches = basin_matches(
        case.observation, reference, polished, case.center_identifiable, configuration
    )
    matches &= converged & np.isfinite(numpy_objectives)
    matching = np.flatnonzero(matches[ranking])
    first_rank = int(matching[0] + 1) if matching.size else 0
    top_k = int(configuration["top_k_policy"]["candidate"])
    certified = fit_reference.certify_candidates(
        case.observation,
        candidates,
        gpu_objectives,
        starts,
        candidate_converged=converged,
        top_k=top_k,
    )
    final_model_rms = log_model_rms(
        case.observation, reference.parameters, certified.result.parameters
    )
    identifiable = case.center_identifiable and reference.jacobian_rank >= int(
        configuration["basin_match"]["identifiable_rank"]
    )
    final_center = float("nan")
    if identifiable:
        final_center = abs(certified.result.parameters[1] - reference.parameters[1]) / (
            0.48 * max(abs(reference.parameters[1]), 1.0)
        )
    scale = max(abs(reference.objective), 1e-12)
    row = {
        "case_id": case.case_id,
        "arm": case.arm,
        "challenge": case.challenge,
        "support_spec_no": case.support_spec_no,
        "true_center_ev": case.true_center_ev,
        "true_amplitude": case.true_amplitude,
        "true_pitch_sigma_deg": case.true_pitch_sigma_deg,
        "reference_winner_start": reference.winner_index,
        "reference_objective": reference.objective,
        "reference_rank": reference.jacobian_rank,
        "reference_success": bool(
            reference.fits[reference.winner_index].success
            and np.isfinite(reference.objective)
        ),
        "reference_fits": _json(
            [
                {
                    "start": fit.start_index,
                    "objective": fit.objective,
                    "success": fit.success,
                    "parameters": fit.parameters.tolist(),
                }
                for fit in reference.fits
            ]
        ),
        "gpu_candidates": _json(candidates),
        "gpu_objectives": _json(gpu_objectives),
        "gpu_converged": _json(converged.astype(bool)),
        "canonical_objectives": _json(numpy_objectives),
        "canonical_ranking": _json(ranking),
        "cpu_polished_gpu_fits": _json(
            [
                {
                    "candidate": index,
                    "objective": fit.objective,
                    "success": fit.success,
                    "parameters": fit.parameters.tolist(),
                }
                for index, fit in enumerate(polished.fits)
            ]
        ),
        "basin_matches": _json(matches),
        "minimum_matching_gpu_rank": first_rank,
        "gpu_finite_candidates": int(np.count_nonzero(np.isfinite(numpy_objectives))),
        "gpu_converged_candidates": int(np.count_nonzero(converged)),
        "used_fallback": certified.used_fallback,
        "fallback_reason": certified.fallback_reason or "",
        "final_objective_relative_difference": abs(
            certified.result.objective - reference.objective
        )
        / scale,
        "final_log_model_rms": final_model_rms,
        "final_center_difference_response_fwhm": final_center,
        "final_rank": certified.result.jacobian_rank,
        "final_contacts_bound": _contacts_bound(certified.result.parameters),
        **timers,
    }
    for k in range(1, N_STARTS + 1):
        row[f"recall_k{k}"] = bool(first_rank and first_rank <= k)
    return row


def _contacts_bound(parameters: np.ndarray, tolerance: float = 1e-8) -> bool:
    lower, upper = fit_reference.parameter_bounds()
    distance = np.minimum(parameters - lower, upper - parameters)
    return bool(np.any(distance <= tolerance * np.maximum(upper - lower, 1.0)))


def _group_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for value in sorted({str(row[field]) for row in rows}):
        selected = [row for row in rows if str(row[field]) == value]
        groups[value] = {
            "cases": len(selected),
            "recall_by_k": {
                str(k): float(np.mean([row[f"recall_k{k}"] for row in selected]))
                for k in range(1, N_STARTS + 1)
            },
            "fallback_fraction": float(
                np.mean([row["used_fallback"] for row in selected])
            ),
        }
    return groups


def summarize(
    rows: list[dict[str, Any]], configuration: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate K=1..12 recall and every frozen final-certification gate."""
    if not rows:
        raise ValueError("At least one result row is required")
    gate = configuration["basin_match"]
    recall = {
        str(k): float(np.mean([row[f"recall_k{k}"] for row in rows]))
        for k in range(1, N_STARTS + 1)
    }
    ceiling = int(configuration["gates"]["top_k_ceiling"])
    passing = [k for k in range(1, ceiling + 1) if recall[str(k)] == 1.0]
    objective_pass = all(
        float(row["final_objective_relative_difference"])
        <= float(gate["cpu_polished_objective_relative_difference_max"])
        for row in rows
    )
    model_pass = all(
        float(row["final_log_model_rms"]) <= float(gate["log_model_rms_max"])
        for row in rows
    )
    center_pass = all(
        not np.isfinite(float(row["final_center_difference_response_fwhm"]))
        or float(row["final_center_difference_response_fwhm"])
        <= float(gate["cpu_polished_center_difference_response_fwhm_max"])
        for row in rows
    )
    finite_pass = all(int(row["gpu_finite_candidates"]) == N_STARTS for row in rows)
    reference_pass = all(bool(row["reference_success"]) for row in rows)
    eligible_pass = all(
        int(row["gpu_converged_candidates"])
        >= int(configuration["gates"]["gpu_eligible_candidates_min"])
        for row in rows
    )
    return {
        "cases": len(rows),
        "recall_by_k": recall,
        "minimum_full_recall_k_with_ceiling_6": min(passing) if passing else None,
        "top_k_ceiling_6_pass": bool(passing),
        "finite_candidates_pass": finite_pass,
        "reference_success_pass": reference_pass,
        "gpu_eligible_count_pass": eligible_pass,
        "certified_objective_parity_pass": objective_pass,
        "certified_model_parity_pass": model_pass,
        "certified_center_parity_pass": center_pass,
        "bound_contact_cases": sum(bool(row["final_contacts_bound"]) for row in rows),
        "rank_deficient_cases": sum(
            int(row["final_rank"]) < N_PARAMETERS for row in rows
        ),
        "fallback_cases": sum(bool(row["used_fallback"]) for row in rows),
        "fallback_reasons": {
            reason: sum(row["fallback_reason"] == reason for row in rows)
            for reason in sorted(
                {row["fallback_reason"] for row in rows if row["fallback_reason"]}
            )
        },
        "arms": _group_summary(rows, "arm"),
        "challenges": _group_summary(rows, "challenge"),
        "full_gate_pass": bool(
            passing
            and reference_pass
            and finite_pass
            and eligible_pass
            and objective_pass
            and model_pass
            and center_pass
        ),
        "elapsed_components_seconds": {
            key: float(sum(float(row[key]) for row in rows))
            for key in (
                "starts_seconds",
                "reference_seconds",
                "gpu_seconds",
                "all_gpu_polish_seconds",
            )
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    experiment_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--configuration",
        type=Path,
        default=experiment_dir / "configuration-d1-certification.json",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=experiment_dir,
        help=(
            "Artifact directory; required outside the experiment directory for "
            "partial runs"
        ),
    )
    args = parser.parse_args()
    if (args.shard_index is None) != (args.shard_count is None):
        parser.error("--shard-index and --shard-count must be supplied together")
    if args.shard_count is not None and (
        args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count
    ):
        parser.error("shard index must satisfy 0 <= index < count")
    if args.limit is not None and args.output_dir.resolve() == experiment_dir:
        parser.error(
            "partial runs require --output-dir outside the experiment directory"
        )
    configuration = load_configuration(args.configuration)
    supports = support_data.load_geometry_supports(
        Path("data/1999/091_120APR/3D990429.TAB"),
        experiment_dir / "support_manifest.csv",
    )
    cases = make_cases(supports, configuration)
    if args.shard_count is not None:
        cases = [
            case
            for case in cases
            if case.case_id % args.shard_count == args.shard_index
        ]
    if args.limit is not None:
        cases = cases[: args.limit]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    started = time.perf_counter()
    rows = []
    for index, case in enumerate(cases, start=1):
        rows.append(run_case(case, configuration, device))
        print(
            f"[{index}/{len(cases)}] case {case.case_id} {case.arm}/{case.challenge}",
            flush=True,
        )
    summary = summarize(rows, configuration)
    summary["elapsed_seconds"] = time.perf_counter() - started
    summary["repository_commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "d1-certification-results.csv"
    summary_path = args.output_dir / "d1-certification-summary.json"
    write_csv(results_path, rows)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    provenance = {
        "repository_commit": summary["repository_commit"],
        "configuration_sha256": sha256_file(args.configuration),
        "results_sha256": sha256_file(results_path),
        "summary_sha256": sha256_file(summary_path),
        "confirmation_flux_opened": False,
        "device": str(device),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
    }
    (args.output_dir / "d1-certification-provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
