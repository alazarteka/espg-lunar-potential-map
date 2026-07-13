#!/usr/bin/env python3
"""Run one shard of the frozen Experiment 006 G2 certification corpus."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import beam_model
import c1_estimator
import fit_reference
import g2_beam_fit
import numpy as np
import simulator
import start_parameters
import support_data

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "g2-certification-corpus.csv"
C1_CONFIG = ROOT / "configuration-c1.json"
G2_CONFIG = ROOT / "configuration-g2.json"


def load_rows(path: Path = CORPUS) -> list[dict[str, str]]:
    """Load and validate the frozen ordered corpus."""
    with path.open(encoding="ascii", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [int(row["case_id"]) for row in rows] != list(range(192)):
        raise ValueError("G2 corpus must contain ordered case IDs 0 through 191")
    return rows


def matched_observation(
    support: support_data.GeometrySupport, row: dict[str, str]
) -> fit_reference.Observation:
    """Construct one canonical matched-model observation."""
    pitches = np.asarray(support.pitches_deg, dtype=np.float64)
    selected = np.isfinite(pitches) & (pitches >= 90.0)
    row_ids, _ = np.nonzero(selected)
    parameters = np.array(
        [
            float(row["amplitude"]),
            float(row["true_center_ev"]),
            float(row["pitch_sigma_deg"]),
            *[float(row[f"coefficient_{index}"]) for index in range(12)],
        ]
    )
    observation = fit_reference.Observation(
        support.energies_ev,
        support.channel_low_ev,
        support.channel_high_ev,
        row_ids.astype(np.int64),
        pitches[selected],
        np.ones(row_ids.size),
    )
    values = beam_model.model_values(
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
    return fit_reference.Observation(
        observation.energies_ev,
        observation.channel_low_ev,
        observation.channel_high_ev,
        observation.row_ids,
        observation.pitches_deg,
        values,
    )


def make_observation(
    support: support_data.GeometrySupport,
    row: dict[str, str],
    c1_config: dict[str, Any],
) -> fit_reference.Observation:
    """Construct the frozen matched or independent-sector observation."""
    if row["generator"] == "canonical_matched":
        return matched_observation(support, row)
    case = simulator.simulate_case(support, row, c1_config)
    branches = c1_estimator.construct_observations(
        support.energies_ev,
        support.channel_low_ev,
        support.channel_high_ev,
        support.pitches_deg,
        case.native_flux,
        case.native_valid,
        case.incident_mean,
        case.conjugate,
        configuration=c1_config,
    )
    return branches[row["normalization"]].observation


def cpu_reference(
    observation: fit_reference.Observation,
) -> tuple[fit_reference.FitResult | None, tuple[fit_reference.StartFit, ...]]:
    """Run and retain all old complete twelve-start CPU trajectories."""
    lower, upper = fit_reference.parameter_bounds()
    starts = start_parameters.construct_starts(observation, lower, upper)
    fits = fit_reference._fit_starts(observation, starts, max_nfev=2000)
    result = (
        fit_reference._summarize(fits)
        if any(fit.success and np.isfinite(fit.objective) for fit in fits)
        else None
    )
    return result, fits


def _status_summary(fits: tuple[fit_reference.StartFit, ...]) -> str:
    """Encode SciPy status counts without discarding failed trajectories."""
    statuses, counts = np.unique([fit.status for fit in fits], return_counts=True)
    pairs = zip(statuses, counts, strict=True)
    return ";".join(f"{int(status)}:{int(count)}" for status, count in pairs)


def run_case(
    row: dict[str, str],
    support: support_data.GeometrySupport,
    c1_config: dict[str, Any],
    g2_config: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    """Evaluate all frozen G2 gates for one certification case."""
    observation = make_observation(support, row, c1_config)
    old, old_fits = cpu_reference(observation)
    case_id = int(row["case_id"])
    new = g2_beam_fit.fit_observations(
        [observation], [case_id], device=device, configuration=g2_config
    )
    replay = g2_beam_fit.fit_observations(
        [observation], [case_id], device=device, configuration=g2_config
    )
    result = new.results[0]
    replay_result = replay.results[0]
    fwhm = 0.48 * float(row["true_center_ev"])
    new_fits = new.polish_trajectories[0]
    successful_centers = np.array(
        [
            fit.parameters[1]
            for fit in new_fits
            if fit.success and np.isfinite(fit.objective)
        ]
    )
    matched_error = (
        np.nan
        if result is None
        else abs(result.parameters[1] - float(row["true_center_ev"])) / fwhm
    )
    if old is None or result is None:
        objective_delta = np.nan
    else:
        objective_scale = max(abs(old.objective), 1e-12)
        objective_delta = (result.objective - old.objective) / objective_scale
    replay_equal = (result is None) == (replay_result is None)
    if result is not None and replay_result is not None:
        replay_equal &= np.array_equal(result.parameters, replay_result.parameters)
    return {
        "case_id": case_id,
        "arm": row["arm"],
        "support_id": int(row["support_id"]),
        "normalization": row["normalization"],
        "true_center_ev": float(row["true_center_ev"]),
        "old_available": old is not None,
        "new_available": result is not None,
        "old_status_counts": _status_summary(old_fits),
        "new_status_counts": _status_summary(new_fits),
        "old_center_ev": np.nan if old is None else float(old.parameters[1]),
        "new_center_ev": np.nan if result is None else float(result.parameters[1]),
        "old_objective": np.nan if old is None else float(old.objective),
        "new_objective": np.nan if result is None else float(result.objective),
        "objective_relative_delta": float(objective_delta),
        "matched_error_fwhm": float(matched_error),
        "island_center_spread_fwhm": (
            np.nan
            if successful_centers.size == 0
            else float(np.ptp(successful_centers) / fwhm)
        ),
        "jacobian_rank": -1 if result is None else int(result.jacobian_rank),
        "covariance_finite": (
            False if result is None else bool(np.isfinite(result.covariance).all())
        ),
        "deterministic_replay": bool(
            replay_equal
            and np.array_equal(new.island_parameters, replay.island_parameters)
            and np.array_equal(new.island_objectives, replay.island_objectives)
        ),
        "gpu_seconds": new.gpu_seconds,
        "cpu_polish_seconds": new.cpu_polish_seconds,
        "confirmation_flux_opened": False,
        "partial_c1_scientific_outcomes_opened": False,
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one deterministic LF-only certification shard."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows()
    if not 0 <= args.start < args.stop <= len(rows):
        raise ValueError("Invalid certification shard range")
    c1_config = json.loads(C1_CONFIG.read_text(encoding="ascii"))
    g2_config = json.loads(G2_CONFIG.read_text(encoding="ascii"))
    supports = support_data.load_geometry_supports(
        Path("data/1999/091_120APR/3D990429.TAB"), ROOT / "support_manifest.csv"
    )
    geometry = {support.spec_no: support for support in supports}
    output: list[dict[str, Any]] = []
    for row in rows[args.start : args.stop]:
        print(f"case_id={row['case_id']}", flush=True)
        output.append(
            run_case(
                row,
                geometry[int(row["support_id"])],
                c1_config,
                g2_config,
                args.device,
            )
        )
        write_rows(args.output, output)


if __name__ == "__main__":
    main()
