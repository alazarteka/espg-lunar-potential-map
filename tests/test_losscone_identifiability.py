from __future__ import annotations

from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest

import src.losscone.identifiability as identification
from src.losscone.identifiability import (
    AffineNuisanceScenario,
    BoundaryBrackets,
    HardBinDesign,
    IdentificationStatus,
    boundary_brackets_from_pattern,
    generate_hard_pattern,
    ideal_boundary_q,
    identify_delta_v_interval,
    solve_ideal_known_spacecraft,
    solve_ideal_unknown_spacecraft,
    standard_nuisance_scenarios,
)
from src.losscone.model import synth_losscone


def _design(pitches: np.ndarray | None = None) -> HardBinDesign:
    energies = np.array([50.0, 100.0, 200.0, 400.0])
    if pitches is None:
        pitch_row = np.array([30.0, 75.0, 100.0, 120.0, 140.0, 160.0, 175.0])
        pitches = np.tile(pitch_row, (energies.size, 1))
    return HardBinDesign(
        energies_ev=energies,
        pitches_deg=pitches,
        row_times_s=np.array([0.0, 20.0, 50.0, 80.0]),
    )


def _truth_brackets(design: HardBinDesign) -> BoundaryBrackets:
    pattern = generate_hard_pattern(
        design,
        delta_v_midpoint=-60.0,
        spacecraft_midpoint_v=10.0,
        field_ratio=0.5,
    )
    return boundary_brackets_from_pattern(design, pattern)


def test_known_spacecraft_two_boundaries_recover_parameters() -> None:
    energies = np.array([100.0, 400.0])
    q = ideal_boundary_q(
        energies,
        surface_potential_v=-50.0,
        spacecraft_potential_v=10.0,
        field_ratio=0.5,
    )
    result = solve_ideal_known_spacecraft(energies, q, spacecraft_potential_v=10.0)
    assert result.field_ratio == pytest.approx(0.5)
    assert result.surface_potential_v == pytest.approx(-50.0)
    assert result.delta_v == pytest.approx(-60.0)


def test_unknown_spacecraft_three_boundaries_recover_parameters() -> None:
    energies = np.array([100.0, 200.0, 400.0])
    q = ideal_boundary_q(
        energies,
        surface_potential_v=-50.0,
        spacecraft_potential_v=10.0,
        field_ratio=0.5,
    )
    result = solve_ideal_unknown_spacecraft(energies, q)
    assert result.field_ratio == pytest.approx(0.5)
    assert result.surface_potential_v == pytest.approx(-50.0)
    assert result.spacecraft_potential_v == pytest.approx(10.0)
    assert result.delta_v == pytest.approx(-60.0)


def test_unknown_spacecraft_rejects_degenerate_constant_boundary() -> None:
    with pytest.raises(ValueError, match="boundary differences"):
        solve_ideal_unknown_spacecraft(
            np.array([100.0, 200.0, 400.0]), np.array([0.5, 0.5, 0.5])
        )


def test_threshold_equality_is_inside_and_upper_closed() -> None:
    energy = np.array([100.0])
    pitches = np.array([[60.0, 120.0, 150.0, 170.0]])
    design = HardBinDesign(energy, pitches, np.array([0.0]), valid_rows=[True])
    # q = 0.25 exactly equals the pitch=150° threshold.
    pattern = np.array([[True, True, True, False]])
    brackets = boundary_brackets_from_pattern(design, pattern)
    assert brackets.upper[0] == pytest.approx(0.25)
    assert brackets.lower[0] == pytest.approx(np.sin(np.deg2rad(170.0)) ** 2)
    assert bool(brackets.lower_is_open[0])


def test_truth_invalid_energy_row_is_fixed_out_before_bracketing() -> None:
    design = HardBinDesign(
        energies_ev=np.array([5.0, 50.0, 100.0]),
        pitches_deg=np.tile(np.array([45.0, 120.0, 160.0]), (3, 1)),
        row_times_s=np.array([0.0, 40.0, 80.0]),
    )
    pattern = generate_hard_pattern(
        design,
        delta_v_midpoint=-50.0,
        spacecraft_midpoint_v=10.0,
        field_ratio=0.5,
    )
    truth_valid = design.energies_ev > 10.0
    brackets = boundary_brackets_from_pattern(
        design,
        pattern,
        valid_rows=truth_valid,
    )
    assert not bool(brackets.valid_rows[0])
    assert np.all(brackets.valid_rows[1:])


def test_pattern_matches_existing_zero_beam_numpy_model() -> None:
    design = _design()
    pattern = generate_hard_pattern(
        design,
        delta_v_midpoint=-60.0,
        spacecraft_midpoint_v=10.0,
        field_ratio=0.5,
    )
    model = synth_losscone(
        design.energies_ev,
        design.pitches_deg,
        U_surface=-50.0,
        U_spacecraft=10.0,
        bs_over_bm=0.5,
        beam_amp=0.0,
    )
    assert np.array_equal(pattern, model == 1.0)


def test_oracle_interval_contains_generating_truth() -> None:
    design = _design()
    brackets = _truth_brackets(design)
    scenario = standard_nuisance_scenarios(
        truth_field_ratio=0.5, truth_spacecraft_potential_v=10.0
    )[0]
    result = identify_delta_v_interval(design, brackets, scenario)
    assert result.status == IdentificationStatus.SHARP_INTERVAL
    assert result.delta_v_lower <= -60.0 <= result.delta_v_upper
    assert result.beam_energy_lower == pytest.approx(-result.delta_v_upper)
    assert result.beam_energy_upper == pytest.approx(-result.delta_v_lower)
    assert result.surface_interval_is_sharp
    assert result.surface_v_lower == pytest.approx(result.delta_v_lower + 10.0)


def test_fixed_spacecraft_value_overrides_bounding_envelope() -> None:
    design = _design()
    brackets = _truth_brackets(design)
    scenario = AffineNuisanceScenario(
        "fixed_sc",
        (0.5, 0.5),
        (-100.0, 100.0),
        fixed_field_ratio=0.5,
        fixed_spacecraft_midpoint_v=10.0,
    )
    result = identify_delta_v_interval(design, brackets, scenario)
    assert result.surface_interval_is_sharp
    assert result.surface_v_lower == pytest.approx(result.delta_v_lower + 10.0)
    assert result.surface_v_upper == pytest.approx(result.delta_v_upper + 10.0)


def test_widening_nuisance_does_not_shrink_interval() -> None:
    design = _design()
    brackets = _truth_brackets(design)
    scenarios = standard_nuisance_scenarios(
        truth_field_ratio=0.5, truth_spacecraft_potential_v=10.0
    )
    results = [
        identify_delta_v_interval(design, brackets, scenario) for scenario in scenarios
    ]
    for narrower, wider in pairwise(results[:5]):
        assert wider.delta_v_lower <= narrower.delta_v_lower
        assert wider.delta_v_upper >= narrower.delta_v_upper
    # The broad-S stress case removes affine drift, so it is a superset of
    # scenario 4 rather than a strict continuation of scenario 5.
    assert results[5].delta_v_lower <= results[3].delta_v_lower
    assert results[5].delta_v_upper >= results[3].delta_v_upper


def test_zero_affine_drift_matches_constant_scenario() -> None:
    design = _design()
    brackets = _truth_brackets(design)
    constant = AffineNuisanceScenario("constant", (0.3, 1.1), (5.0, 15.0))
    zero_drift = AffineNuisanceScenario(
        "zero_drift",
        (0.3, 1.1),
        (5.0, 15.0),
        delta_drift_total_bounds_v=(0.0, 0.0),
        spacecraft_drift_total_bounds_v=(0.0, 0.0),
    )
    a = identify_delta_v_interval(design, brackets, constant)
    b = identify_delta_v_interval(design, brackets, zero_drift)
    assert a.delta_v_lower == pytest.approx(b.delta_v_lower)
    assert a.delta_v_upper == pytest.approx(b.delta_v_upper)


def test_no_reflected_cells_returns_full_domain() -> None:
    pitches = np.tile(np.array([10.0, 45.0, 89.0]), (4, 1))
    design = _design(pitches)
    pattern = np.ones_like(pitches, dtype=bool)
    brackets = boundary_brackets_from_pattern(design, pattern)
    scenario = standard_nuisance_scenarios(
        truth_field_ratio=0.5, truth_spacecraft_potential_v=10.0
    )[0]
    result = identify_delta_v_interval(design, brackets, scenario)
    assert result.status == IdentificationStatus.FULL_DOMAIN
    assert result.delta_v_lower == -2500.0
    assert result.delta_v_upper == 2000.0


@pytest.mark.parametrize(
    ("truth_delta", "expected_reflected_inside", "domain_flag"),
    [(-500.0, True, "hits_lower_domain"), (500.0, False, "hits_upper_domain")],
)
def test_clipped_rows_supply_only_one_sided_information(
    truth_delta: float,
    expected_reflected_inside: bool,
    domain_flag: str,
) -> None:
    design = _design()
    pattern = generate_hard_pattern(
        design,
        delta_v_midpoint=truth_delta,
        spacecraft_midpoint_v=0.0,
        field_ratio=1.0,
    )
    reflected = design.pitches_deg > 90.0
    assert np.all(pattern[reflected] == expected_reflected_inside)
    brackets = boundary_brackets_from_pattern(design, pattern)
    scenario = AffineNuisanceScenario(
        "fixed", (1.0, 1.0), (0.0, 0.0), fixed_field_ratio=1.0
    )
    result = identify_delta_v_interval(design, brackets, scenario)
    assert result.delta_v_lower <= truth_delta <= result.delta_v_upper
    assert getattr(result, domain_flag)


def test_incompatible_boundary_values_are_infeasible() -> None:
    design = _design()
    brackets = BoundaryBrackets(
        lower=np.array([0.9, 0.1, 0.9, 0.1]),
        upper=np.array([0.9, 0.1, 0.9, 0.1]),
        lower_is_open=np.zeros(4, dtype=bool),
        informative_rows=np.ones(4, dtype=bool),
        valid_rows=np.ones(4, dtype=bool),
    )
    scenario = AffineNuisanceScenario(
        "fixed", (0.5, 0.5), (10.0, 10.0), fixed_field_ratio=0.5
    )
    result = identify_delta_v_interval(design, brackets, scenario)
    assert result.status == IdentificationStatus.INFEASIBLE_PATTERN
    assert np.isnan(result.delta_v_lower)


def test_numerical_failure_conservatively_returns_full_domain(monkeypatch) -> None:
    design = _design()
    brackets = _truth_brackets(design)
    scenario = standard_nuisance_scenarios(
        truth_field_ratio=0.5, truth_spacecraft_potential_v=10.0
    )[0]

    failure = SimpleNamespace(success=False, status=4, message="forced failure")
    monkeypatch.setattr(identification, "linprog", lambda *_args, **_kwargs: failure)
    result = identify_delta_v_interval(design, brackets, scenario)
    assert result.status == IdentificationStatus.UNRESOLVED_NUMERICAL
    assert result.delta_v_lower == -2500.0
    assert result.delta_v_upper == 2000.0


def test_oracle_lp_matches_exhaustive_delta_grid() -> None:
    design = _design()
    truth_pattern = generate_hard_pattern(
        design,
        delta_v_midpoint=-60.0,
        spacecraft_midpoint_v=10.0,
        field_ratio=0.5,
    )
    brackets = boundary_brackets_from_pattern(design, truth_pattern)
    scenario = AffineNuisanceScenario(
        "oracle",
        (0.5, 0.5),
        (10.0, 10.0),
        fixed_field_ratio=0.5,
        fixed_spacecraft_midpoint_v=10.0,
    )
    result = identify_delta_v_interval(
        design, brackets, scenario, delta_v_domain=(-200.0, 50.0)
    )

    grid = np.linspace(-200.0, 50.0, 25001)
    matches = np.array(
        [
            np.array_equal(
                generate_hard_pattern(
                    design,
                    delta_v_midpoint=value,
                    spacecraft_midpoint_v=10.0,
                    field_ratio=0.5,
                ),
                truth_pattern,
            )
            for value in grid
        ]
    )
    assert matches.any()
    step = float(grid[1] - grid[0])
    assert result.delta_v_lower == pytest.approx(float(grid[matches][0]), abs=step)
    assert result.delta_v_upper == pytest.approx(float(grid[matches][-1]), abs=step)
