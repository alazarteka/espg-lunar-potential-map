"""Per-sweep DeltaU confidence-set schema for the D2 error estimator.

The honest uncertainty product is a coverage-calibrated *set*, not a scalar
U-width. Sets may be two-sided, one-sided, disconnected, or the full domain.

Legacy LHS ``u_width_*`` / ``u_is_identifiable_*`` fields are optimizer-geometry
diagnostics only and must not be treated as confidence intervals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

# Explicit observation-level name required by the ER measurement contract.
OBSERVATION_LEVEL = "calibrated_flux_quasi_likelihood"

# Max components stored in fixed-width NPZ arrays.
MAX_CI_COMPONENTS = 8


class GateReason(StrEnum):
    """Selection / regime decisions recorded separately from the fit mask."""

    OK = "ok"
    INSTRUMENT_INVALID = "instrument_invalid"
    POLARITY_UNKNOWN = "polarity_unknown"
    ENERGY_LEVERAGE_INSUFFICIENT = "energy_leverage_insufficient"
    BEAM_EDGE_INCONSISTENT = "beam_edge_inconsistent"
    NONIDENTIFIABLE = "nonidentifiable"
    FULL_DOMAIN = "full_domain"


@dataclass(frozen=True)
class ConfidenceSetComponent:
    """One contiguous interval of an admissible DeltaU (or U_surface) set."""

    lo: float
    hi: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.lo) or not np.isfinite(self.hi):
            raise ValueError("component endpoints must be finite")
        if self.lo > self.hi:
            raise ValueError(f"component lo={self.lo} > hi={self.hi}")

    @property
    def width(self) -> float:
        return float(self.hi - self.lo)


@dataclass(frozen=True)
class SweepConfidenceSet:
    """Honest per-sweep uncertainty record for spacecraft-relative DeltaU.

    Parameters use ``D = -DeltaU`` internally in the profiler; this record stores
    the reported surface-relative potential ``U`` (= -D when U_sc is absorbed
    into the relative definition, or the fitted U_surface when that is the
    profiled coordinate).
    """

    observation_level: str = OBSERVATION_LEVEL
    u_hat: float = float("nan")
    r_hat: float = float("nan")
    beam_amp_hat: float = float("nan")
    nll_min: float = float("nan")
    c_alpha: float = float("nan")
    alpha: float = 0.32
    bootstrap_n: int = 0
    components: tuple[ConfidenceSetComponent, ...] = ()
    domain_lo: float = float("nan")
    domain_hi: float = float("nan")
    touches_bound_lo: bool = False
    touches_bound_hi: bool = False
    is_full_domain: bool = False
    is_one_sided: bool = False
    gate_reason: GateReason = GateReason.OK
    notes: str = ""
    # Optional profile samples for diagnostics (not required for science claims).
    profile_u: tuple[float, ...] = ()
    profile_lambda: tuple[float, ...] = ()

    @property
    def n_components(self) -> int:
        return len(self.components)

    @property
    def is_empty(self) -> bool:
        return self.n_components == 0 and not self.is_full_domain

    def span_width(self) -> float:
        """Overall max-min span across components (diagnostic only)."""
        if self.is_full_domain:
            if np.isfinite(self.domain_lo) and np.isfinite(self.domain_hi):
                return float(self.domain_hi - self.domain_lo)
            return float("nan")
        if not self.components:
            return float("nan")
        return float(
            max(c.hi for c in self.components) - min(c.lo for c in self.components)
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["gate_reason"] = str(self.gate_reason)
        d["components"] = [{"lo": c.lo, "hi": c.hi} for c in self.components]
        return d


@dataclass
class ConfidenceSetBatch:
    """Fixed-width arrays for writing confidence sets into batch NPZs."""

    spec_no: np.ndarray
    observation_level: str = OBSERVATION_LEVEL
    u_hat: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    r_hat: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    beam_amp_hat: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    nll_min: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    c_alpha: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    alpha: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    bootstrap_n: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.int32)
    )
    n_components: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.int32)
    )
    component_lo: np.ndarray = field(
        default_factory=lambda: np.zeros((0, MAX_CI_COMPONENTS), dtype=np.float64)
    )
    component_hi: np.ndarray = field(
        default_factory=lambda: np.zeros((0, MAX_CI_COMPONENTS), dtype=np.float64)
    )
    domain_lo: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    domain_hi: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    touches_bound_lo: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=bool)
    )
    touches_bound_hi: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=bool)
    )
    is_full_domain: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    is_one_sided: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    gate_reason: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))

    @classmethod
    def from_sets(
        cls, spec_nos: np.ndarray, sets: list[SweepConfidenceSet]
    ) -> ConfidenceSetBatch:
        n = len(sets)
        if len(spec_nos) != n:
            raise ValueError("spec_nos and sets must have the same length")
        lo = np.full((n, MAX_CI_COMPONENTS), np.nan, dtype=np.float64)
        hi = np.full((n, MAX_CI_COMPONENTS), np.nan, dtype=np.float64)
        for i, s in enumerate(sets):
            for j, c in enumerate(s.components[:MAX_CI_COMPONENTS]):
                lo[i, j] = c.lo
                hi[i, j] = c.hi
        return cls(
            spec_no=np.asarray(spec_nos, dtype=np.int64),
            observation_level=OBSERVATION_LEVEL,
            u_hat=np.array([s.u_hat for s in sets], dtype=np.float64),
            r_hat=np.array([s.r_hat for s in sets], dtype=np.float64),
            beam_amp_hat=np.array([s.beam_amp_hat for s in sets], dtype=np.float64),
            nll_min=np.array([s.nll_min for s in sets], dtype=np.float64),
            c_alpha=np.array([s.c_alpha for s in sets], dtype=np.float64),
            alpha=np.array([s.alpha for s in sets], dtype=np.float64),
            bootstrap_n=np.array([s.bootstrap_n for s in sets], dtype=np.int32),
            n_components=np.array([s.n_components for s in sets], dtype=np.int32),
            component_lo=lo,
            component_hi=hi,
            domain_lo=np.array([s.domain_lo for s in sets], dtype=np.float64),
            domain_hi=np.array([s.domain_hi for s in sets], dtype=np.float64),
            touches_bound_lo=np.array([s.touches_bound_lo for s in sets], dtype=bool),
            touches_bound_hi=np.array([s.touches_bound_hi for s in sets], dtype=bool),
            is_full_domain=np.array([s.is_full_domain for s in sets], dtype=bool),
            is_one_sided=np.array([s.is_one_sided for s in sets], dtype=bool),
            gate_reason=np.array([str(s.gate_reason) for s in sets], dtype=object),
        )

    def to_npz_arrays(self, *, prefix: str = "spec_ci_") -> dict[str, np.ndarray]:
        """Serialize to NPZ-friendly arrays with a stable key prefix."""
        out: dict[str, np.ndarray] = {
            f"{prefix}observation_level": np.array(self.observation_level),
            f"{prefix}spec_no": np.asarray(self.spec_no, dtype=np.int64),
            f"{prefix}u_hat": self.u_hat,
            f"{prefix}r_hat": self.r_hat,
            f"{prefix}beam_amp_hat": self.beam_amp_hat,
            f"{prefix}nll_min": self.nll_min,
            f"{prefix}c_alpha": self.c_alpha,
            f"{prefix}alpha": self.alpha,
            f"{prefix}bootstrap_n": self.bootstrap_n,
            f"{prefix}n_components": self.n_components,
            f"{prefix}component_lo": self.component_lo,
            f"{prefix}component_hi": self.component_hi,
            f"{prefix}domain_lo": self.domain_lo,
            f"{prefix}domain_hi": self.domain_hi,
            f"{prefix}touches_bound_lo": self.touches_bound_lo,
            f"{prefix}touches_bound_hi": self.touches_bound_hi,
            f"{prefix}is_full_domain": self.is_full_domain,
            f"{prefix}is_one_sided": self.is_one_sided,
            f"{prefix}gate_reason": self.gate_reason.astype(str),
        }
        return out


def components_from_retained(
    u_grid: np.ndarray,
    retained: np.ndarray,
    *,
    domain_lo: float,
    domain_hi: float,
    atol: float = 1e-9,
) -> tuple[tuple[ConfidenceSetComponent, ...], bool, bool, bool, bool]:
    """Convert a boolean retention mask on a sorted U grid into components.

    Returns:
        components, touches_bound_lo, touches_bound_hi, is_full_domain, is_one_sided
    """
    u = np.asarray(u_grid, dtype=np.float64)
    keep = np.asarray(retained, dtype=bool)
    if u.ndim != 1 or keep.shape != u.shape:
        raise ValueError("u_grid and retained must be 1-D and aligned")
    if u.size == 0 or not np.any(keep):
        return (), False, False, False, False

    order = np.argsort(u)
    u = u[order]
    keep = keep[order]

    # Find contiguous True runs.
    padded = np.concatenate([[False], keep, [False]])
    edges = np.diff(padded.astype(np.int8))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1
    comps = tuple(
        ConfidenceSetComponent(lo=float(u[s]), hi=float(u[e]))
        for s, e in zip(starts, ends, strict=True)
    )

    touches_lo = bool(keep[0] and abs(u[0] - domain_lo) <= atol)
    touches_hi = bool(keep[-1] and abs(u[-1] - domain_hi) <= atol)
    is_full = bool(np.all(keep) and touches_lo and touches_hi)
    # One-sided: exactly one finite endpoint (touches exactly one domain bound,
    # or a single component that reaches only one bound).
    is_one_sided = bool(
        (len(comps) == 1) and (touches_lo != touches_hi) and not is_full
    )
    return comps, touches_lo, touches_hi, is_full, is_one_sided


def level_set_from_profile(
    u_grid: np.ndarray,
    lambda_profile: np.ndarray,
    c_alpha: float,
    *,
    domain_lo: float,
    domain_hi: float,
) -> tuple[tuple[ConfidenceSetComponent, ...], bool, bool, bool, bool]:
    """Build the confidence set ``{U : Lambda(U) <= c_alpha}`` from a profile.

    Endpoints are linearly interpolated between grid nodes so a sharp
    likelihood does not collapse to a zero-width single-node set.
    """
    u = np.asarray(u_grid, dtype=np.float64)
    lam = np.asarray(lambda_profile, dtype=np.float64)
    c = float(c_alpha)
    if u.ndim != 1 or lam.shape != u.shape:
        raise ValueError("u_grid and lambda_profile must be aligned 1-D arrays")
    if u.size == 0:
        return (), False, False, False, False

    order = np.argsort(u)
    u = u[order]
    lam = lam[order]
    finite = np.isfinite(lam)
    if not np.any(finite):
        return (), False, False, False, False

    # Treat non-finite Lambda as above threshold.
    above = np.where(finite, lam > c, True)
    inside = ~above

    if not np.any(inside):
        return (), False, False, False, False

    def _cross(i0: int, i1: int) -> float:
        """Interpolate U where Lambda crosses c between nodes i0 and i1."""
        y0, y1 = lam[i0] - c, lam[i1] - c
        if y0 == y1:
            return float(u[i0])
        t = y0 / (y0 - y1)
        t = float(np.clip(t, 0.0, 1.0))
        return float(u[i0] + t * (u[i1] - u[i0]))

    comps: list[ConfidenceSetComponent] = []
    n = u.size
    i = 0
    while i < n:
        if not inside[i]:
            i += 1
            continue
        start = i
        while i + 1 < n and inside[i + 1]:
            i += 1
        end = i
        lo = float(u[start])
        hi = float(u[end])
        if start > 0 and finite[start - 1] and above[start - 1]:
            lo = _cross(start - 1, start)
        if end + 1 < n and finite[end + 1] and above[end + 1]:
            hi = _cross(end, end + 1)
        comps.append(ConfidenceSetComponent(lo=lo, hi=hi))
        i += 1

    atol = 1e-9
    touches_lo = bool(inside[0] and abs(u[0] - domain_lo) <= atol)
    touches_hi = bool(inside[-1] and abs(u[-1] - domain_hi) <= atol)
    # Extend to domain bound when the retained run includes the boundary node.
    if touches_lo and comps:
        comps[0] = ConfidenceSetComponent(lo=float(domain_lo), hi=comps[0].hi)
    if touches_hi and comps:
        comps[-1] = ConfidenceSetComponent(lo=comps[-1].lo, hi=float(domain_hi))

    is_full = bool(np.all(inside) and touches_lo and touches_hi)
    is_one_sided = bool(
        (len(comps) == 1) and (touches_lo != touches_hi) and not is_full
    )
    return tuple(comps), touches_lo, touches_hi, is_full, is_one_sided
