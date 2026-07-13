#!/usr/bin/env python3
"""Benchmark one geometry-realistic, matched 17-fit G2 case without C1 data."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import certify_c1_estimator
import g2_beam_fit
import numpy as np
import support_data


def omit_row(observation, row: int):
    """Return the matched observation without samples from one energy row."""
    keep = observation.row_ids != row
    return replace(
        observation,
        row_ids=observation.row_ids[keep],
        pitches_deg=observation.pitches_deg[keep],
        observed=observation.observed[keep],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--generations", type=int)
    parser.add_argument("--population", type=int)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    supports = support_data.load_geometry_supports(
        Path("data/1999/091_120APR/3D990429.TAB"),
        directory / "support_manifest.csv",
    )
    fixture = certify_c1_estimator.make_fixture(supports[0], 220.0, "incident_mean")
    primary = fixture.observation
    observations = [primary, replace(primary)]
    observations.extend(
        omit_row(primary, int(row)) for row in np.unique(primary.row_ids)
    )
    config = g2_beam_fit.load_configuration()
    if args.generations is not None:
        config["estimator"]["maximum_generations"] = args.generations
    if args.population is not None:
        config["estimator"]["population_per_island"] = args.population
    result = g2_beam_fit.fit_observations(
        observations,
        list(range(len(observations))),
        device=args.device,
        configuration=config,
    )
    centers = np.array([fit.parameters[1] for fit in result.results])
    print(
        json.dumps(
            {
                "problems": len(observations),
                "gpu_seconds": result.gpu_seconds,
                "cpu_polish_seconds": result.cpu_polish_seconds,
                "evaluations_per_problem": result.evaluations_per_problem,
                "maximum_center_error_fwhm": float(
                    np.max(np.abs(centers - 220.0)) / (0.48 * 220.0)
                ),
                "device": args.device,
                "confirmation_flux_opened": False,
                "c1_outcomes_opened": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
