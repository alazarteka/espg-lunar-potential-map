"""Quick profiler for nightside spacecraft potential estimation using synthetic data."""

from __future__ import annotations

import cProfile
import pstats
from functools import partial
from unittest.mock import patch

import numpy as np

from src.spacecraft_potential import calculate_potential
from src.utils.synthetic import prepare_synthetic_er


def run_once(spec_no: int) -> None:
    er_data = prepare_synthetic_er()
    fake_et = 0.0
    fake_pos = np.array([1700.0, 0.0, 0.0])
    fake_sun_vec = np.array([-1.0, 0.0, 0.0])

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
            "src.spacecraft_potential.get_intersection_or_none", return_value=fake_pos
        ),
    ):
        calculate_potential(er_data, spec_no)


def main(iterations: int = 3, spec_no: int = 1) -> None:
    profile = cProfile.Profile()
    run = partial(run_once, spec_no)
    for _ in range(iterations):
        profile.runcall(run)

    stats = pstats.Stats(profile)
    stats.strip_dirs().sort_stats("cumtime").print_stats(25)


if __name__ == "__main__":
    main()
