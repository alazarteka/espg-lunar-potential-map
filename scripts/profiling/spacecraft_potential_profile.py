"""Profiler for spacecraft potential estimation using synthetic data.

Usage:
    uv run python scripts/profiling/spacecraft_potential_profile.py
    uv run python scripts/profiling/spacecraft_potential_profile.py --dayside
    uv run python scripts/profiling/spacecraft_potential_profile.py --iterations 10
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
from functools import partial
from unittest.mock import patch

import numpy as np

from src.spacecraft_potential import calculate_potential
from src.utils.synthetic import prepare_synthetic_er


def run_once(spec_no: int, dayside: bool = False) -> None:
    er_data = prepare_synthetic_er()
    fake_et = 0.0
    fake_pos = np.array([1700.0, 0.0, 0.0])
    fake_sun_vec = np.array([1.0, 0.0, 0.0])
    # Dayside: intersection is None (line of sight to sun doesn't hit moon)
    # Nightside: intersection is a point on the moon surface
    fake_intersection = None if dayside else fake_pos

    with (
        patch("src.spacecraft_potential.spice.str2et", return_value=fake_et),
        patch(
            "src.spacecraft_potential.get_lp_position_wrt_moon", return_value=fake_pos
        ),
        patch(
            "src.spacecraft_potential.get_lp_vector_to_sun_in_lunar_frame",
            return_value=fake_sun_vec,
        ),
        patch(
            "src.spacecraft_potential.get_intersection_or_none",
            return_value=fake_intersection,
        ),
    ):
        calculate_potential(er_data, spec_no)


def main(iterations: int = 3, spec_no: int = 1, dayside: bool = False) -> None:
    mode = "dayside" if dayside else "nightside"
    print(f"Profiling {mode} spacecraft potential ({iterations} iterations)\n")

    profile = cProfile.Profile()
    run = partial(run_once, spec_no, dayside)
    for _ in range(iterations):
        profile.runcall(run)

    stats = pstats.Stats(profile)
    stats.strip_dirs().sort_stats("cumtime").print_stats(25)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=3, help="Number of runs")
    parser.add_argument("--spec-no", type=int, default=1, help="Spectrum number")
    parser.add_argument("--dayside", action="store_true", help="Profile dayside path")
    args = parser.parse_args()
    main(iterations=args.iterations, spec_no=args.spec_no, dayside=args.dayside)
