"""Regime gates for loss-cone DeltaU inference (outside the likelihood).

Physics-validity conditions are logically prior to any confidence set. A
structurally wrong model can produce a narrow profile around a meaningless
number; within-model statistics cannot detect that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.losscone.confidence_set import GateReason


@dataclass(frozen=True)
class RegimeGateResult:
    ok: bool
    reason: GateReason
    n_usable_channels: int = 0
    energy_leverage: float = float("nan")
    beam_edge_delta_v: float = float("nan")
    message: str = ""


def expected_energy_leverage(
    energies: np.ndarray,
    usable: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> float:
    """Proxy for Fisher leverage: weighted variance of 1/K on usable channels.

    Fresh-look: det I ∝ Var_w(1/K). Zero leverage ⇒ non-identifiable.
    """
    e = np.asarray(energies, dtype=np.float64)
    m = np.asarray(usable, dtype=bool)
    if e.ndim != 1 or m.shape != e.shape:
        raise ValueError("energies and usable must be aligned 1-D arrays")
    if int(np.count_nonzero(m)) < 2:
        return 0.0
    x = 1.0 / np.maximum(e[m], 1e-6)
    if weights is None:
        w = np.ones_like(x)
    else:
        w = np.asarray(weights, dtype=np.float64)[m]
        w = np.maximum(w, 0.0)
    if not np.any(w > 0):
        return 0.0
    w = w / w.sum()
    mean = float(np.sum(w * x))
    var = float(np.sum(w * (x - mean) ** 2))
    return max(var, 0.0)


def prefit_regime_gate(
    energies: np.ndarray,
    *,
    polarity: float | int | None,
    flux: np.ndarray | None = None,
    min_usable_channels: int = 2,
    min_counts_proxy: float = 0.0,
    min_leverage: float = 1e-10,
) -> RegimeGateResult:
    """Mandatory pre-fit physical gate.

    Args:
        energies: (nE,) centre energies.
        polarity: +1 / -1 magnetic footpoint polarity; 0/None → fail.
        flux: optional (nE, nPitch) calibrated flux for channel population.
        min_usable_channels: require >= this many partially open channels.
        min_counts_proxy: minimum row-sum flux treated as "populated".
        min_leverage: minimum Var(1/K) on usable channels.
    """
    if polarity is None or not np.isfinite(float(polarity)) or float(polarity) == 0.0:
        return RegimeGateResult(
            ok=False,
            reason=GateReason.POLARITY_UNKNOWN,
            message="magnetic footpoint polarity missing or zero",
        )

    e = np.asarray(energies, dtype=np.float64)
    if flux is None:
        usable = np.ones(e.shape, dtype=bool)
        weights = None
    else:
        f = np.asarray(flux, dtype=np.float64)
        row_sum = np.nansum(np.where(np.isfinite(f), f, 0.0), axis=1)
        usable = row_sum > float(min_counts_proxy)
        weights = row_sum

    n_usable = int(np.count_nonzero(usable))
    if n_usable < int(min_usable_channels):
        return RegimeGateResult(
            ok=False,
            reason=GateReason.ENERGY_LEVERAGE_INSUFFICIENT,
            n_usable_channels=n_usable,
            energy_leverage=0.0,
            message=f"only {n_usable} usable channels (need >={min_usable_channels})",
        )

    lev = expected_energy_leverage(e, usable, weights=weights)
    if lev < float(min_leverage):
        return RegimeGateResult(
            ok=False,
            reason=GateReason.ENERGY_LEVERAGE_INSUFFICIENT,
            n_usable_channels=n_usable,
            energy_leverage=lev,
            message=f"energy leverage Var(1/K)={lev:.3e} below floor",
        )

    return RegimeGateResult(
        ok=True,
        reason=GateReason.OK,
        n_usable_channels=n_usable,
        energy_leverage=lev,
    )


def beam_edge_consistency(
    *,
    u_edge: float,
    beam_centroid_eV: float | None,
    u_spacecraft: float = 0.0,
    tolerance_v: float = 50.0,
) -> RegimeGateResult:
    """Post-fit cross-check: beam centroid should track D = U_sc - U_surface.

    Disagreement diagnoses nonmonotonic barrier, wrong polarity, or a
    misidentified beam - emit a limit / nonidentification, not a tight CI.
    """
    if beam_centroid_eV is None or not np.isfinite(beam_centroid_eV):
        return RegimeGateResult(
            ok=True,
            reason=GateReason.OK,
            message="no beam centroid available; skipped",
        )
    expected = float(u_spacecraft) - float(u_edge)
    if expected <= 0:
        return RegimeGateResult(
            ok=True,
            reason=GateReason.OK,
            message="no expected secondary acceleration; skipped",
        )
    delta = abs(float(beam_centroid_eV) - expected)
    if delta > float(tolerance_v):
        return RegimeGateResult(
            ok=False,
            reason=GateReason.BEAM_EDGE_INCONSISTENT,
            beam_edge_delta_v=delta,
            message=(
                f"beam centroid {beam_centroid_eV:.1f} eV vs expected "
                f"{expected:.1f} eV (|Delta|={delta:.1f} V > {tolerance_v})"
            ),
        )
    return RegimeGateResult(
        ok=True,
        reason=GateReason.OK,
        beam_edge_delta_v=delta,
    )
