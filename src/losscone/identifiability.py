"""Structural identification for the finite hard-bin loss-cone model.

This module deliberately does not define a likelihood or confidence interval.
It computes the closure of the parameter set that produces a declared binary
loss-cone pattern under explicit nuisance bounds.

The canonical estimand is ``delta_v = V_surface - V_spacecraft``. The secondary
electron beam energy is therefore ``-delta_v``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from scipy.optimize import OptimizeWarning, linprog

DELTA_V_DOMAIN = (-2500.0, 2000.0)
NUMERICAL_PAD_V = 1.0e-6
DENOMINATOR_MARGIN_EV = 1.0e-9

_R = 0
_B0 = 1
_B1 = 2
_S0 = 3
_S1 = 4
_N_ORIGINAL_VARIABLES = 5


class IdentificationStatus(StrEnum):
    """Outcome of one hard-bin identified-set calculation."""

    SHARP_INTERVAL = "sharp_interval"
    FULL_DOMAIN = "full_domain"
    INFEASIBLE_PATTERN = "infeasible_pattern"
    INVALID_DESIGN = "invalid_design"
    UNRESOLVED_NUMERICAL = "unresolved_numerical"


@dataclass(frozen=True)
class HardBinDesign:
    """Fixed energy/pitch/time design for one 3-D ER sweep."""

    energies_ev: np.ndarray
    pitches_deg: np.ndarray
    row_times_s: np.ndarray
    valid_rows: np.ndarray | None = None

    def __post_init__(self) -> None:
        energies = np.asarray(self.energies_ev, dtype=np.float64).copy()
        pitches = np.asarray(self.pitches_deg, dtype=np.float64).copy()
        times = np.asarray(self.row_times_s, dtype=np.float64).copy()

        if energies.ndim != 1:
            raise ValueError("energies_ev must be one-dimensional")
        if pitches.ndim != 2 or pitches.shape[0] != energies.size:
            raise ValueError("pitches_deg must have shape (n_energy, n_pitch)")
        if times.shape != energies.shape:
            raise ValueError("row_times_s must match energies_ev")

        if self.valid_rows is None:
            valid = np.ones(energies.size, dtype=bool)
        else:
            valid = np.asarray(self.valid_rows, dtype=bool).copy()
            if valid.shape != energies.shape:
                raise ValueError("valid_rows must match energies_ev")

        valid &= (
            np.isfinite(energies)
            & (energies > 0.0)
            & np.isfinite(times)
            & np.isfinite(pitches).any(axis=1)
        )
        finite_pitches = pitches[np.isfinite(pitches)]
        if np.any((finite_pitches < 0.0) | (finite_pitches > 180.0)):
            raise ValueError("finite pitches_deg values must lie in [0, 180]")
        object.__setattr__(self, "energies_ev", energies)
        object.__setattr__(self, "pitches_deg", pitches)
        object.__setattr__(self, "row_times_s", times)
        object.__setattr__(self, "valid_rows", valid)

    @property
    def tau(self) -> np.ndarray:
        """Row times normalized so the sweep endpoints are -1/2 and +1/2."""
        finite = np.isfinite(self.row_times_s)
        if np.count_nonzero(finite) < 2:
            return np.full_like(self.row_times_s, np.nan)
        t_min = float(np.min(self.row_times_s[finite]))
        t_max = float(np.max(self.row_times_s[finite]))
        span = t_max - t_min
        if span <= 0:
            return np.full_like(self.row_times_s, np.nan)
        midpoint = 0.5 * (t_min + t_max)
        return (self.row_times_s - midpoint) / span


@dataclass(frozen=True)
class BoundaryBrackets:
    """Per-energy closure brackets for ``q = sin²(alpha_losscone)``."""

    lower: np.ndarray
    upper: np.ndarray
    lower_is_open: np.ndarray
    informative_rows: np.ndarray
    valid_rows: np.ndarray

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=np.float64).copy()
        upper = np.asarray(self.upper, dtype=np.float64).copy()
        lower_open = np.asarray(self.lower_is_open, dtype=bool).copy()
        informative = np.asarray(self.informative_rows, dtype=bool).copy()
        valid = np.asarray(self.valid_rows, dtype=bool).copy()
        shape = lower.shape
        if lower.ndim != 1:
            raise ValueError("BoundaryBrackets arrays must be one-dimensional")
        if any(arr.shape != shape for arr in (upper, lower_open, informative, valid)):
            raise ValueError("all BoundaryBrackets arrays must have matching shapes")
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError("boundary endpoints must be finite")
        if np.any(lower < 0.0) or np.any(upper > 1.0):
            raise ValueError("boundary endpoints must lie in [0, 1]")
        if np.any(lower > upper + 1.0e-12):
            raise ValueError("boundary lower endpoint exceeds upper endpoint")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "lower_is_open", lower_open)
        object.__setattr__(self, "informative_rows", informative)
        object.__setattr__(self, "valid_rows", valid)


@dataclass(frozen=True)
class AffineNuisanceScenario:
    """Nuisance bounds for a constant-field, affine-with-time sweep model."""

    scenario_id: str
    field_ratio_bounds: tuple[float, float]
    spacecraft_midpoint_bounds_v: tuple[float, float]
    delta_drift_total_bounds_v: tuple[float, float] = (0.0, 0.0)
    spacecraft_drift_total_bounds_v: tuple[float, float] = (0.0, 0.0)
    fixed_field_ratio: float | None = None
    fixed_spacecraft_midpoint_v: float | None = None
    description: str = ""

    def __post_init__(self) -> None:
        r_lo, r_hi = self.field_ratio_bounds
        s_lo, s_hi = self.spacecraft_midpoint_bounds_v
        d_lo, d_hi = self.delta_drift_total_bounds_v
        sd_lo, sd_hi = self.spacecraft_drift_total_bounds_v
        if not (0 < r_lo <= r_hi):
            raise ValueError("field-ratio bounds must be positive and ordered")
        if s_lo > s_hi or d_lo > d_hi or sd_lo > sd_hi:
            raise ValueError("nuisance bounds must be ordered")
        if self.fixed_field_ratio is not None and not (
            r_lo <= self.fixed_field_ratio <= r_hi
        ):
            raise ValueError("fixed_field_ratio lies outside its bounds")
        if self.fixed_spacecraft_midpoint_v is not None and not (
            s_lo <= self.fixed_spacecraft_midpoint_v <= s_hi
        ):
            raise ValueError("fixed spacecraft potential lies outside its bounds")

    @property
    def maximum_spacecraft_potential_v(self) -> float:
        """Largest spacecraft potential anywhere in the affine sweep envelope."""
        midpoint_max = (
            float(self.fixed_spacecraft_midpoint_v)
            if self.fixed_spacecraft_midpoint_v is not None
            else float(self.spacecraft_midpoint_bounds_v[1])
        )
        return midpoint_max + 0.5 * max(
            abs(float(self.spacecraft_drift_total_bounds_v[0])),
            abs(float(self.spacecraft_drift_total_bounds_v[1])),
        )

    @property
    def effective_spacecraft_midpoint_bounds_v(self) -> tuple[float, float]:
        """Spacecraft bounds after applying an optional exact midpoint."""
        if self.fixed_spacecraft_midpoint_v is not None:
            fixed = float(self.fixed_spacecraft_midpoint_v)
            return fixed, fixed
        return self.spacecraft_midpoint_bounds_v


@dataclass(frozen=True)
class IdentificationResult:
    """Conservative closure of a hard-bin identified interval."""

    status: IdentificationStatus
    delta_v_lower: float
    delta_v_upper: float
    beam_energy_lower: float
    beam_energy_upper: float
    surface_v_lower: float
    surface_v_upper: float
    surface_interval_is_sharp: bool
    informative_rows: int
    used_rows: int
    hits_lower_domain: bool
    hits_upper_domain: bool
    solver: str
    message: str = ""


@dataclass(frozen=True)
class IdealBoundarySolution:
    """Parameters recovered from exact, unclipped loss-cone boundaries."""

    field_ratio: float
    surface_potential_v: float
    spacecraft_potential_v: float
    delta_v: float


def ideal_boundary_q(
    energies_ev: np.ndarray,
    *,
    surface_potential_v: float,
    spacecraft_potential_v: float,
    field_ratio: float,
) -> np.ndarray:
    """Return the unclipped ideal boundary ``q(E)``."""
    energies = np.asarray(energies_ev, dtype=np.float64)
    return float(field_ratio) * (
        1.0 + float(surface_potential_v) / (energies - spacecraft_potential_v)
    )


def solve_ideal_known_spacecraft(
    energies_ev: np.ndarray,
    q: np.ndarray,
    *,
    spacecraft_potential_v: float,
) -> IdealBoundarySolution:
    """Recover field ratio and potentials from two exact boundary values."""
    energies = np.asarray(energies_ev, dtype=np.float64)
    boundary = np.asarray(q, dtype=np.float64)
    if energies.size != 2 or boundary.shape != energies.shape:
        raise ValueError("known-spacecraft recovery requires exactly two boundaries")
    z = 1.0 / (energies - float(spacecraft_potential_v))
    if not np.isfinite(z).all() or np.isclose(z[0], z[1]):
        raise ValueError("energies do not provide distinct finite leverage")
    slope = float((boundary[1] - boundary[0]) / (z[1] - z[0]))
    field_ratio = float(boundary[0] - slope * z[0])
    if not np.isfinite(field_ratio) or field_ratio <= 0:
        raise ValueError("recovered field ratio is not positive")
    surface = slope / field_ratio
    spacecraft = float(spacecraft_potential_v)
    return IdealBoundarySolution(
        field_ratio=field_ratio,
        surface_potential_v=surface,
        spacecraft_potential_v=spacecraft,
        delta_v=surface - spacecraft,
    )


def solve_ideal_unknown_spacecraft(
    energies_ev: np.ndarray,
    q: np.ndarray,
) -> IdealBoundarySolution:
    """Recover common spacecraft potential and other parameters from three values."""
    energies = np.asarray(energies_ev, dtype=np.float64)
    boundary = np.asarray(q, dtype=np.float64)
    if energies.size != 3 or boundary.shape != energies.shape:
        raise ValueError(
            "unknown-spacecraft recovery requires exactly three boundaries"
        )

    q12 = float(boundary[0] - boundary[1])
    q23 = float(boundary[1] - boundary[2])
    e21 = float(energies[1] - energies[0])
    e32 = float(energies[2] - energies[1])
    if np.isclose(q23, 0.0) or np.isclose(e21, 0.0) or np.isclose(e32, 0.0):
        raise ValueError("boundary differences do not identify spacecraft potential")
    ratio = (q12 / q23) * (e32 / e21)
    if np.isclose(1.0 - ratio, 0.0):
        raise ValueError("degenerate boundary ratio")
    spacecraft = float((energies[2] - ratio * energies[0]) / (1.0 - ratio))
    recovered = solve_ideal_known_spacecraft(
        energies[:2], boundary[:2], spacecraft_potential_v=spacecraft
    )
    predicted = ideal_boundary_q(
        energies,
        surface_potential_v=recovered.surface_potential_v,
        spacecraft_potential_v=spacecraft,
        field_ratio=recovered.field_ratio,
    )
    if not np.allclose(predicted, boundary, rtol=1.0e-9, atol=1.0e-10):
        raise ValueError("three boundaries are inconsistent with the ideal model")
    return recovered


def generate_hard_pattern(
    design: HardBinDesign,
    *,
    delta_v_midpoint: float,
    spacecraft_midpoint_v: float,
    field_ratio: float,
    delta_drift_total_v: float = 0.0,
    spacecraft_drift_total_v: float = 0.0,
) -> np.ndarray:
    """Generate the zero-beam binary pattern for one parameter point."""
    tau = design.tau
    delta = float(delta_v_midpoint) + float(delta_drift_total_v) * tau
    spacecraft = float(spacecraft_midpoint_v) + float(spacecraft_drift_total_v) * tau
    denominator = design.energies_ev - spacecraft
    raw_q = np.full_like(design.energies_ev, np.nan, dtype=np.float64)
    valid = design.valid_rows & np.isfinite(tau) & (denominator > 0)
    raw_q[valid] = (
        float(field_ratio)
        * (design.energies_ev[valid] + delta[valid])
        / denominator[valid]
    )
    q = np.clip(raw_q, 0.0, 1.0)
    alpha = np.rad2deg(np.arcsin(np.sqrt(q)))
    pattern = design.pitches_deg <= (180.0 - alpha[:, None])
    pattern[~valid] = False
    return pattern


def boundary_brackets_from_pattern(
    design: HardBinDesign,
    pattern: np.ndarray,
    *,
    valid_rows: np.ndarray | None = None,
) -> BoundaryBrackets:
    """Compress a finite pitch pattern into exact per-energy boundary brackets."""
    inside = np.asarray(pattern, dtype=bool)
    if inside.shape != design.pitches_deg.shape:
        raise ValueError("pattern must match the design pitch grid")

    n_energy = design.energies_ev.size
    lower = np.zeros(n_energy, dtype=np.float64)
    upper = np.ones(n_energy, dtype=np.float64)
    lower_open = np.zeros(n_energy, dtype=bool)
    informative = np.zeros(n_energy, dtype=bool)
    valid = np.asarray(design.valid_rows, dtype=bool).copy()
    if valid_rows is not None:
        declared_valid = np.asarray(valid_rows, dtype=bool)
        if declared_valid.shape != valid.shape:
            raise ValueError("valid_rows must match the design energy dimension")
        valid &= declared_valid

    for i in range(n_energy):
        if not valid[i]:
            continue
        pitches = design.pitches_deg[i]
        finite = np.isfinite(pitches)
        incident = finite & (pitches <= 90.0)
        if np.any(incident & ~inside[i]):
            raise ValueError("pattern excludes an incident cell and is not a hard cone")
        reflected = finite & (pitches > 90.0)
        if not np.any(reflected):
            continue
        thresholds = np.sin(np.deg2rad(pitches[reflected])) ** 2
        reflected_inside = inside[i, reflected]
        if np.any(~reflected_inside):
            lower[i] = float(np.max(thresholds[~reflected_inside]))
            lower_open[i] = True
        if np.any(reflected_inside):
            upper[i] = float(np.min(thresholds[reflected_inside]))
        if lower[i] > upper[i] + 1.0e-12:
            raise ValueError(
                "reflected-cell pattern is not monotone in pitch threshold"
            )
        informative[i] = lower[i] > 0.0 or upper[i] < 1.0

    return BoundaryBrackets(
        lower=lower,
        upper=upper,
        lower_is_open=lower_open,
        informative_rows=informative,
        valid_rows=valid,
    )


def standard_nuisance_scenarios(
    *,
    truth_field_ratio: float,
    truth_spacecraft_potential_v: float,
) -> tuple[AffineNuisanceScenario, ...]:
    """Return the version-1 nuisance sensitivity ladder."""
    truth_r = float(truth_field_ratio)
    truth_s = float(truth_spacecraft_potential_v)
    return (
        AffineNuisanceScenario(
            scenario_id="oracle",
            field_ratio_bounds=(truth_r, truth_r),
            spacecraft_midpoint_bounds_v=(truth_s, truth_s),
            fixed_field_ratio=truth_r,
            fixed_spacecraft_midpoint_v=truth_s,
            description="Known field ratio and spacecraft potential; no drift.",
        ),
        AffineNuisanceScenario(
            scenario_id="common_r",
            field_ratio_bounds=(0.3, 1.1),
            spacecraft_midpoint_bounds_v=(truth_s, truth_s),
            fixed_spacecraft_midpoint_v=truth_s,
            description="Common unknown field ratio; known spacecraft potential.",
        ),
        AffineNuisanceScenario(
            scenario_id="sc_pm5",
            field_ratio_bounds=(0.3, 1.1),
            spacecraft_midpoint_bounds_v=(truth_s - 5.0, truth_s + 5.0),
            description="Common field ratio and ±5 V spacecraft-potential envelope.",
        ),
        AffineNuisanceScenario(
            scenario_id="sc_pm20",
            field_ratio_bounds=(0.3, 1.1),
            spacecraft_midpoint_bounds_v=(truth_s - 20.0, truth_s + 20.0),
            description="Common field ratio and ±20 V spacecraft-potential envelope.",
        ),
        AffineNuisanceScenario(
            scenario_id="affine_drift_20",
            field_ratio_bounds=(0.3, 1.1),
            spacecraft_midpoint_bounds_v=(truth_s - 20.0, truth_s + 20.0),
            delta_drift_total_bounds_v=(-20.0, 20.0),
            spacecraft_drift_total_bounds_v=(-20.0, 20.0),
            description="±20 V midpoint envelope plus bounded affine sweep drift.",
        ),
        AffineNuisanceScenario(
            scenario_id="broad_sc",
            field_ratio_bounds=(0.3, 1.1),
            spacecraft_midpoint_bounds_v=(-1500.0, 150.0),
            description="Broad current implementation spacecraft-potential envelope.",
        ),
    )


def _append_bound(
    rows: list[np.ndarray], rhs: list[float], coefficients: list[float], bound: float
) -> None:
    rows.append(np.asarray(coefficients, dtype=np.float64))
    rhs.append(float(bound))


def _original_lp_constraints(
    design: HardBinDesign,
    brackets: BoundaryBrackets,
    scenario: AffineNuisanceScenario,
    delta_v_domain: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``A x <= c`` in ``x=(r,b0,b1,s0,s1)`` plus fixed safe rows."""
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    delta_lo, delta_hi = delta_v_domain

    # Keep the midpoint and both affine sweep endpoints inside the declared
    # computational domain. Because tau spans [-1/2, 1/2], these endpoint
    # constraints also bound every intervening row.
    _append_bound(rows, rhs, [delta_lo, -1.0, 0.0, 0.0, 0.0], 0.0)
    _append_bound(rows, rhs, [-delta_hi, 1.0, 0.0, 0.0, 0.0], 0.0)
    for endpoint_sign in (-0.5, 0.5):
        _append_bound(
            rows,
            rhs,
            [delta_lo, -1.0, -endpoint_sign, 0.0, 0.0],
            0.0,
        )
        _append_bound(
            rows,
            rhs,
            [-delta_hi, 1.0, endpoint_sign, 0.0, 0.0],
            0.0,
        )

    drift_lo, drift_hi = scenario.delta_drift_total_bounds_v
    _append_bound(rows, rhs, [drift_lo, 0.0, -1.0, 0.0, 0.0], 0.0)
    _append_bound(rows, rhs, [-drift_hi, 0.0, 1.0, 0.0, 0.0], 0.0)

    s_lo, s_hi = scenario.effective_spacecraft_midpoint_bounds_v
    _append_bound(rows, rhs, [0.0, 0.0, 0.0, 1.0, 0.0], s_hi)
    _append_bound(rows, rhs, [0.0, 0.0, 0.0, -1.0, 0.0], -s_lo)
    sd_lo, sd_hi = scenario.spacecraft_drift_total_bounds_v
    _append_bound(rows, rhs, [0.0, 0.0, 0.0, 0.0, 1.0], sd_hi)
    _append_bound(rows, rhs, [0.0, 0.0, 0.0, 0.0, -1.0], -sd_lo)

    tau = design.tau
    maximum_s = scenario.maximum_spacecraft_potential_v
    safe = (
        design.valid_rows
        & brackets.valid_rows
        & np.isfinite(tau)
        & (design.energies_ev > maximum_s + DENOMINATOR_MARGIN_EV)
    )

    for i in np.flatnonzero(safe & brackets.informative_rows):
        energy = float(design.energies_ev[i])
        time = float(tau[i])
        lower = float(brackets.lower[i])
        upper = float(brackets.upper[i])
        if lower > 0.0:
            _append_bound(
                rows,
                rhs,
                [-energy, -1.0, -time, -lower, -lower * time],
                -lower * energy,
            )
        if upper < 1.0:
            _append_bound(
                rows,
                rhs,
                [energy, 1.0, time, upper, upper * time],
                upper * energy,
            )

    for i in np.flatnonzero(safe):
        _append_bound(
            rows,
            rhs,
            [0.0, 0.0, 0.0, 1.0, float(tau[i])],
            float(design.energies_ev[i]) - DENOMINATOR_MARGIN_EV,
        )

    return np.vstack(rows), np.asarray(rhs, dtype=np.float64), safe


def _transformed_lp(
    original_a: np.ndarray,
    original_c: np.ndarray,
    scenario: AffineNuisanceScenario,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[tuple[float | None, float | None]],
]:
    """Apply Charnes-Cooper with ``t=1/r`` and ``y=x/r``."""
    a_ub = np.column_stack((original_a, -original_c))
    b_ub = np.zeros(original_a.shape[0], dtype=np.float64)
    a_eq = np.zeros((1, _N_ORIGINAL_VARIABLES + 1), dtype=np.float64)
    a_eq[0, _R] = 1.0
    b_eq = np.array([1.0], dtype=np.float64)

    r_lo, r_hi = scenario.field_ratio_bounds
    t_bounds = (1.0 / float(r_hi), 1.0 / float(r_lo))
    if scenario.fixed_field_ratio is not None:
        fixed_t = 1.0 / float(scenario.fixed_field_ratio)
        t_bounds = (fixed_t, fixed_t)
    bounds: list[tuple[float | None, float | None]] = [
        (None, None),
        (None, None),
        (None, None),
        (None, None),
        (None, None),
        t_bounds,
    ]
    return a_ub, b_ub, a_eq, b_eq, bounds


def _full_domain_result(
    *,
    status: IdentificationStatus,
    scenario: AffineNuisanceScenario,
    domain: tuple[float, float],
    informative_rows: int,
    used_rows: int,
    message: str,
) -> IdentificationResult:
    lower, upper = domain
    s_lo, s_hi = scenario.effective_spacecraft_midpoint_bounds_v
    surface_sharp = bool(np.isclose(s_lo, s_hi))
    return IdentificationResult(
        status=status,
        delta_v_lower=lower,
        delta_v_upper=upper,
        beam_energy_lower=-upper,
        beam_energy_upper=-lower,
        surface_v_lower=lower + s_lo,
        surface_v_upper=upper + s_hi,
        surface_interval_is_sharp=surface_sharp,
        informative_rows=informative_rows,
        used_rows=used_rows,
        hits_lower_domain=True,
        hits_upper_domain=True,
        solver="scipy.optimize.linprog(method=highs)",
        message=message,
    )


def identify_delta_v_interval(
    design: HardBinDesign,
    brackets: BoundaryBrackets,
    scenario: AffineNuisanceScenario,
    *,
    delta_v_domain: tuple[float, float] = DELTA_V_DOMAIN,
) -> IdentificationResult:
    """Compute the sharp hard-bin identified-set closure for ``delta_v``.

    The returned set is conditional on the declared deterministic hard model and
    nuisance scenario. It is not a statistical confidence interval.
    """
    delta_lo, delta_hi = map(float, delta_v_domain)
    if delta_lo >= delta_hi:
        raise ValueError("delta_v_domain must be strictly ordered")
    if brackets.lower.shape != design.energies_ev.shape:
        raise ValueError("brackets must match the design energy dimension")

    tau = design.tau
    if not np.isfinite(tau[design.valid_rows]).all():
        return _full_domain_result(
            status=IdentificationStatus.INVALID_DESIGN,
            scenario=scenario,
            domain=(delta_lo, delta_hi),
            informative_rows=0,
            used_rows=0,
            message="valid rows do not have a finite, nonzero time span",
        )

    original_a, original_c, safe = _original_lp_constraints(
        design, brackets, scenario, (delta_lo, delta_hi)
    )
    informative_rows = int(np.count_nonzero(safe & brackets.informative_rows))
    used_rows = int(np.count_nonzero(safe))
    if informative_rows == 0:
        return _full_domain_result(
            status=IdentificationStatus.FULL_DOMAIN,
            scenario=scenario,
            domain=(delta_lo, delta_hi),
            informative_rows=0,
            used_rows=used_rows,
            message="no fixed-safe energy row brackets the loss-cone boundary",
        )

    a_ub, b_ub, a_eq, b_eq, bounds = _transformed_lp(original_a, original_c, scenario)
    objective = np.zeros(_N_ORIGINAL_VARIABLES + 1, dtype=np.float64)
    objective[_B0] = 1.0
    options = {
        "primal_feasibility_tolerance": 1.0e-9,
        "dual_feasibility_tolerance": 1.0e-9,
        # SciPy forwards unrecognized options to HiGHS. Pinning this native
        # option prevents each worker from recruiting additional solver cores.
        "threads": 1,
    }
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Unrecognized options detected: .*'threads': 1.*",
            category=OptimizeWarning,
        )
        lower_result = linprog(
            objective,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
            options=options,
        )
        upper_result = linprog(
            -objective,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
            options=options,
        )

    if lower_result.status == 2 and upper_result.status == 2:
        nan = float("nan")
        return IdentificationResult(
            status=IdentificationStatus.INFEASIBLE_PATTERN,
            delta_v_lower=nan,
            delta_v_upper=nan,
            beam_energy_lower=nan,
            beam_energy_upper=nan,
            surface_v_lower=nan,
            surface_v_upper=nan,
            surface_interval_is_sharp=False,
            informative_rows=informative_rows,
            used_rows=used_rows,
            hits_lower_domain=False,
            hits_upper_domain=False,
            solver="scipy.optimize.linprog(method=highs)",
            message="pattern is incompatible with the declared nuisance scenario",
        )
    if not lower_result.success or not upper_result.success:
        return _full_domain_result(
            status=IdentificationStatus.UNRESOLVED_NUMERICAL,
            scenario=scenario,
            domain=(delta_lo, delta_hi),
            informative_rows=informative_rows,
            used_rows=used_rows,
            message=(
                f"lower status={lower_result.status}: {lower_result.message}; "
                f"upper status={upper_result.status}: {upper_result.message}"
            ),
        )

    raw_lower = float(lower_result.fun)
    raw_upper = float(-upper_result.fun)
    if (
        not np.isfinite(raw_lower)
        or not np.isfinite(raw_upper)
        or raw_lower > raw_upper + NUMERICAL_PAD_V
    ):
        return _full_domain_result(
            status=IdentificationStatus.UNRESOLVED_NUMERICAL,
            scenario=scenario,
            domain=(delta_lo, delta_hi),
            informative_rows=informative_rows,
            used_rows=used_rows,
            message=(
                "HiGHS returned non-finite or inconsistent endpoint objectives: "
                f"lower={raw_lower}, upper={raw_upper}"
            ),
        )
    lower = max(delta_lo, raw_lower - NUMERICAL_PAD_V)
    upper = min(delta_hi, raw_upper + NUMERICAL_PAD_V)
    hits_lower = lower <= delta_lo + NUMERICAL_PAD_V
    hits_upper = upper >= delta_hi - NUMERICAL_PAD_V
    status = (
        IdentificationStatus.FULL_DOMAIN
        if hits_lower and hits_upper
        else IdentificationStatus.SHARP_INTERVAL
    )
    s_lo, s_hi = scenario.effective_spacecraft_midpoint_bounds_v
    surface_sharp = bool(np.isclose(s_lo, s_hi))
    return IdentificationResult(
        status=status,
        delta_v_lower=lower,
        delta_v_upper=upper,
        beam_energy_lower=-upper,
        beam_energy_upper=-lower,
        surface_v_lower=lower + s_lo,
        surface_v_upper=upper + s_hi,
        surface_interval_is_sharp=surface_sharp,
        informative_rows=informative_rows,
        used_rows=used_rows,
        hits_lower_domain=hits_lower,
        hits_upper_domain=hits_upper,
        solver="scipy.optimize.linprog(method=highs)",
    )


__all__ = [
    "DELTA_V_DOMAIN",
    "AffineNuisanceScenario",
    "BoundaryBrackets",
    "HardBinDesign",
    "IdealBoundarySolution",
    "IdentificationResult",
    "IdentificationStatus",
    "boundary_brackets_from_pattern",
    "generate_hard_pattern",
    "ideal_boundary_q",
    "identify_delta_v_interval",
    "solve_ideal_known_spacecraft",
    "solve_ideal_unknown_spacecraft",
    "standard_nuisance_scenarios",
]
