#!/usr/bin/env python3
"""Run the executable pre-C1 certification of the frozen CPU estimator.

This is intentionally expensive: the nominal corpus contains 120 observations,
and every observation is fitted from all 12 frozen CPU starts.  Use the focused
unit tests for development; run this file only when producing the frozen pre-C1
certificate.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import beam_model
import c1_estimator
import fit_reference
import numpy as np
import simulator
import start_parameters
import support_data
from numpy.polynomial.legendre import leggauss

if TYPE_CHECKING:
    from collections.abc import Callable

NORMALIZATION_LABELS = (c1_estimator.INCIDENT_MEAN, c1_estimator.CONJUGATE)
RECOVERY_LIMIT_FWHM = 1e-4
SCALE_FACTOR = 2.0


@dataclass(frozen=True)
class CertificationFixture:
    """One matched noiseless canonical-model observation."""

    support_id: int
    anchor_ev: float
    normalization: str
    observation: fit_reference.Observation
    true_parameters: np.ndarray


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parameters(anchor_ev: float) -> np.ndarray:
    return np.array([8.0, anchor_ev, 24.0, *([0.0] * 12)], dtype=np.float64)


def make_fixture(
    support: support_data.GeometrySupport,
    anchor_ev: float,
    normalization: str,
) -> CertificationFixture:
    """Build a matched noiseless reflected-hemisphere canonical fixture."""
    if normalization not in NORMALIZATION_LABELS:
        raise ValueError(f"Unknown normalization label: {normalization}")
    pitches = np.asarray(support.pitches_deg, dtype=np.float64)
    selected = np.isfinite(pitches) & (pitches >= 90.0)
    row_ids, _sector_ids = np.nonzero(selected)
    parameters = _parameters(anchor_ev)
    observation = fit_reference.Observation(
        np.asarray(support.energies_ev, dtype=np.float64),
        np.asarray(support.channel_low_ev, dtype=np.float64),
        np.asarray(support.channel_high_ev, dtype=np.float64),
        row_ids.astype(np.int64),
        pitches[selected],
        np.empty(row_ids.size, dtype=np.float64),
    )
    observed = beam_model.model_values(
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
    return CertificationFixture(
        int(support.spec_no),
        float(anchor_ev),
        normalization,
        replace(observation, observed=observed),
        parameters,
    )


def make_fixtures(
    supports: list[support_data.GeometrySupport], anchors_ev: list[float]
) -> list[CertificationFixture]:
    """Enumerate anchors, all calibration supports, and both branch labels."""
    return [
        make_fixture(support, anchor, label)
        for anchor in anchors_ev
        for support in supports
        for label in NORMALIZATION_LABELS
    ]


def _fit_complete(
    observation: fit_reference.Observation, *, max_nfev: int
) -> fit_reference.FitResult:
    lower, upper = fit_reference.parameter_bounds()
    starts = start_parameters.construct_starts(observation, lower, upper)
    if starts.shape != (12, fit_reference.N_PARAMETERS):
        raise RuntimeError("Estimator certification requires exactly 12 CPU starts")
    return fit_reference.fit_reference(observation, starts, max_nfev=max_nfev)


def _center_error_fwhm(fitted_center: float, true_center: float) -> float:
    return abs(fitted_center - true_center) / (0.48 * true_center)


def _fit_fixture(fixture: CertificationFixture, max_nfev: int) -> tuple[float, bool]:
    result = _fit_complete(fixture.observation, max_nfev=max_nfev)
    replay = _fit_complete(fixture.observation, max_nfev=max_nfev)
    error = _center_error_fwhm(result.parameters[1], fixture.anchor_ev)
    return error, bool(np.array_equal(result.parameters, replay.parameters))


def _permutation_shift(fixture: CertificationFixture, max_nfev: int) -> float:
    baseline = _fit_complete(fixture.observation, max_nfev=max_nfev)
    permuted = _fit_complete(_permuted(fixture.observation), max_nfev=max_nfev)
    return abs(permuted.parameters[1] - baseline.parameters[1]) / (
        0.48 * fixture.anchor_ev
    )


def _permuted(observation: fit_reference.Observation) -> fit_reference.Observation:
    order = np.arange(observation.observed.size - 1, -1, -1)
    return replace(
        observation,
        row_ids=observation.row_ids[order],
        pitches_deg=observation.pitches_deg[order],
        observed=observation.observed[order],
    )


def common_native_flux_scale_invariant(
    support: support_data.GeometrySupport,
) -> bool:
    """Check that both normalized products cancel a common native-flux scale."""
    pitches = np.asarray(support.pitches_deg, dtype=np.float64)
    row = np.arange(pitches.shape[0], dtype=np.float64)[:, None]
    sector = np.arange(pitches.shape[1], dtype=np.float64)[None, :]
    native_flux = 1.0 + 0.125 * row + 0.0625 * sector
    native_valid = np.isfinite(pitches)
    baseline = simulator.normalize_native(native_flux, pitches, native_valid)
    scaled = simulator.normalize_native(
        SCALE_FACTOR * native_flux, pitches, native_valid
    )
    return all(
        np.array_equal(first, second, equal_nan=True)
        for first, second in zip(baseline, scaled, strict=True)
    )


def quadrature_refinement_error() -> float:
    """Compare the analytic channel response with refined independent quadrature."""
    low = np.geomspace(20.0, 1200.0, 31)
    high = low * 1.17
    center = 220.0
    analytic = beam_model.integrated_energy_response(low, high, center, 0.48)
    nodes, weights = leggauss(96)
    energy = 0.5 * (high - low)[:, None] * nodes + 0.5 * (high + low)[:, None]
    sigma = 0.48 * center / 2.355
    values = np.exp(-0.5 * np.square((energy - center) / sigma))
    numerical = 0.5 * np.sum(values * weights, axis=1)
    return float(np.max(np.abs(numerical - analytic)) / np.max(np.abs(analytic)))


def simulator_source_independent() -> bool:
    """Confirm the independent simulator cannot call the estimator model."""
    source = Path(simulator.__file__).read_text(encoding="ascii")
    return "beam_model" not in source and "model_values" not in source


def certify(
    supports: list[support_data.GeometrySupport],
    config: dict[str, Any],
    *,
    fit: Callable[..., fit_reference.FitResult] = _fit_complete,
    jobs: int = 1,
) -> dict[str, Any]:
    """Execute all matched fits and practical estimator invariance gates."""
    anchors = [float(value) for value in config["allocation"]["anchors_ev"]]
    expected_supports = int(config["allocation"]["calibration_support_count"])
    if len(supports) != expected_supports:
        raise ValueError(f"Expected {expected_supports} calibration supports")
    fixtures = make_fixtures(supports, anchors)
    max_nfev = int(config["estimator"]["max_nfev"])
    if jobs > 1 and fit is _fit_complete:
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
            fitted = list(
                executor.map(
                    _fit_fixture,
                    fixtures,
                    [max_nfev] * len(fixtures),
                    chunksize=1,
                )
            )
        errors = [error for error, _replay in fitted]
        replay_pass = all(replay for _error, replay in fitted)
    else:
        errors = []
        replay_pass = True
        for fixture in fixtures:
            result = fit(fixture.observation, max_nfev=max_nfev)
            replay = fit(fixture.observation, max_nfev=max_nfev)
            errors.append(_center_error_fwhm(result.parameters[1], fixture.anchor_ev))
            replay_pass &= np.array_equal(result.parameters, replay.parameters)

    # One fixture per support exercises ordering and units without tripling the
    # already large anchor/branch corpus.
    permutation_fixtures = fixtures[: 2 * len(supports) : 2]
    if jobs > 1 and fit is _fit_complete:
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
            permutation_errors = list(
                executor.map(
                    _permutation_shift,
                    permutation_fixtures,
                    [max_nfev] * len(permutation_fixtures),
                    chunksize=1,
                )
            )
    else:
        permutation_errors = []
        for fixture in permutation_fixtures:
            baseline = fit(fixture.observation, max_nfev=max_nfev)
            permuted = fit(_permuted(fixture.observation), max_nfev=max_nfev)
            width = 0.48 * fixture.anchor_ev
            permutation_errors.append(
                abs(permuted.parameters[1] - baseline.parameters[1]) / width
            )
    scale_pass = all(
        common_native_flux_scale_invariant(support) for support in supports
    )

    quadrature_error = quadrature_refinement_error()
    maximum_error = float(max(errors, default=np.inf))
    maximum_permutation = float(max(permutation_errors, default=np.inf))
    source_pass = simulator_source_independent()
    expected_cases = len(anchors) * expected_supports * len(NORMALIZATION_LABELS)
    full_pass = all(
        (
            len(fixtures) == expected_cases,
            maximum_error <= RECOVERY_LIMIT_FWHM,
            replay_pass,
            maximum_permutation <= RECOVERY_LIMIT_FWHM,
            scale_pass,
            quadrature_error
            <= float(config["response"]["quadrature_relative_tolerance"]),
            source_pass,
        )
    )
    return {
        "anchors_ev": anchors,
        "support_count": len(supports),
        "normalization_labels": list(NORMALIZATION_LABELS),
        "cases": len(fixtures),
        "fits_per_case": 12,
        "recovery_limit_nominal_fwhm": RECOVERY_LIMIT_FWHM,
        "maximum_recovery_error_nominal_fwhm": maximum_error,
        "recovery_pass": maximum_error <= RECOVERY_LIMIT_FWHM,
        "deterministic_replay_pass": replay_pass,
        "maximum_permutation_shift_nominal_fwhm": maximum_permutation,
        "sector_permutation_pass": maximum_permutation <= RECOVERY_LIMIT_FWHM,
        "common_native_flux_scale_factor": SCALE_FACTOR,
        "common_native_flux_scale_pass": scale_pass,
        "quadrature_refinement_relative_error": quadrature_error,
        "quadrature_refinement_pass": quadrature_error
        <= float(config["response"]["quadrature_relative_tolerance"]),
        "simulator_source_independence_pass": source_pass,
        "full_gate_pass": full_pass,
        "confirmation_flux_opened": False,
    }


def main() -> None:
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=directory / "c1-estimator-certification.json"
    )
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    config_path = directory / "configuration-c1.json"
    manifest_path = directory / "c1-case-manifest.csv"
    support_manifest_path = directory / "support_manifest.csv"
    config = json.loads(config_path.read_text(encoding="ascii"))
    supports = support_data.load_geometry_supports(
        Path("data/1999/091_120APR/3D990429.TAB"), support_manifest_path
    )
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    summary = certify(supports, config, jobs=args.jobs)
    summary["parallel_jobs"] = args.jobs
    summary["hashes"] = {
        path.name: sha256(path)
        for path in (
            config_path,
            manifest_path,
            support_manifest_path,
            directory / "beam_model.py",
            directory / "fit_reference.py",
            directory / "start_parameters.py",
            directory / "c1_estimator.py",
            directory / "simulator.py",
            Path(__file__),
        )
    }
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
