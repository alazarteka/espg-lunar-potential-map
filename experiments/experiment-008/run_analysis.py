#!/usr/bin/env python3
"""Run the frozen geometry-only identifiability analyses for Experiment 008.

Examples from the repository root::

    .venv/bin/python experiments/experiment-008/run_analysis.py prepare
    .venv/bin/python experiments/experiment-008/run_analysis.py primary
    .venv/bin/python experiments/experiment-008/run_analysis.py grid
    .venv/bin/python experiments/experiment-008/run_analysis.py static
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
import observation_operator as operator
from replay_context import ARTIFACT_DIR, RESULTS_DIR

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
GEOMETRY_PATH = ARTIFACT_DIR / "geometry.npz"
BASIS_PATH = ARTIFACT_DIR / "canonical-sweep-basis-l10.npy"
BASIS_METADATA_PATH = ARTIFACT_DIR / "canonical-sweep-basis-l10.json"
CONFIGURATION_PATH = EXPERIMENT_DIR / "configuration.json"
AMENDMENT_03_PATH = EXPERIMENT_DIR / "configuration-amendment-03.json"
DAY_NS = int(86_400 * 1e9)
_COVARIANCE_SQUARES: dict[tuple[str, int, int], np.ndarray] = {}


@dataclass(frozen=True)
class WindowResult:
    """Compact diagnostics for one nonempty independent window."""

    model: str
    lmax: int
    duration_hours: float
    window_id: int
    start_utc: str
    stop_utc: str
    rows: int
    parameters: int
    exact_rank: int
    effective_rank_1e3: int
    effective_rank_1e6: int
    effective_rank_1e9: int
    condition_number: float
    visibility_min: float
    visibility_max: float
    unstable: bool
    recovery_median_relative: float
    recovery_p90_relative: float
    recovery_pass: bool
    leverage_median: float | None = None
    leverage_p99: float | None = None
    resolution_trace_1e6: float | None = None


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""
    with path.open(encoding="ascii") as stream:
        return dict(json.load(stream))


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def iso_utc(utc_ns: int) -> str:
    """Format an integer nanosecond UTC timestamp."""
    return np.datetime_as_string(np.datetime64(utc_ns, "ns"), unit="s")


def seed_from_parts(*parts: object) -> int:
    """Derive a stable 128-bit integer seed from explicit labels."""
    payload = "\x1f".join(str(part) for part in parts).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "little")


def slice_geometry(
    geometry: operator.SweepGeometry, item: slice | np.ndarray
) -> operator.SweepGeometry:
    """Return a row subset while preserving sweep structure."""
    return operator.SweepGeometry(
        source_index=geometry.source_index[item],
        spec_no=geometry.spec_no[item],
        row_utc_ns=geometry.row_utc_ns[item],
        latitude=geometry.latitude[item],
        longitude=geometry.longitude[item],
        polarity=geometry.polarity[item],
        valid_rows=geometry.valid_rows[item],
        midpoint_utc_ns=geometry.midpoint_utc_ns[item],
    )


def iter_windows(
    midpoint_utc_ns: np.ndarray,
    duration_hours: float,
) -> tuple[list[tuple[int, np.ndarray]], int]:
    """Return nonempty epoch-anchored windows and scheduled empty count."""
    duration_ns = round(duration_hours * 3600.0 * 1e9)
    window_id = np.floor_divide(midpoint_utc_ns, duration_ns)
    unique = np.unique(window_id)
    windows = [(int(value), np.flatnonzero(window_id == value)) for value in unique]
    scheduled = int(unique[-1] - unique[0] + 1)
    return windows, scheduled - len(windows)


def covariance_recovery_quantiles(
    singular_values: np.ndarray,
    rows: int,
    parameters: int,
    *,
    noise_sigma_v: float,
    truth_rms_v: float,
    draws: int,
    seed_namespace: str,
) -> tuple[float, float]:
    """Evaluate global-RMS least-squares error from the exact SVD covariance."""
    singular_values = np.asarray(singular_values[:parameters], dtype=np.float64)
    if rows < parameters or np.any(singular_values <= 0.0):
        return float("inf"), float("inf")
    cache_key = (seed_namespace, parameters, draws)
    standard_squared = _COVARIANCE_SQUARES.get(cache_key)
    if standard_squared is None:
        seed = seed_from_parts(seed_namespace, "covariance", parameters, draws)
        rng = np.random.Generator(np.random.PCG64DXSM(seed))
        standard_squared = rng.standard_normal((draws, parameters)) ** 2
        _COVARIANCE_SQUARES[cache_key] = standard_squared
    scales = noise_sigma_v / (np.sqrt(rows) * singular_values)
    relative_error = np.sqrt(standard_squared @ (scales * scales)) / truth_rms_v
    return (
        float(np.quantile(relative_error, 0.5, method="linear")),
        float(np.quantile(relative_error, 0.9, method="linear")),
    )


def classify_spectrum(
    singular_values: np.ndarray,
    rows: int,
    parameters: int,
    configuration: dict[str, Any],
) -> tuple[int, dict[float, int], float, bool]:
    """Apply frozen rank, condition, and physical-visibility gates."""
    largest = float(singular_values[0]) if singular_values.size else 0.0
    tolerance = max(rows, parameters) * np.finfo(np.float64).eps * largest
    exact_rank = int(np.sum(singular_values > tolerance))
    thresholds = tuple(
        configuration["diagnostics"]["relative_effective_rank_thresholds"]
    )
    effective = {
        float(threshold): int(np.sum(singular_values > threshold * largest))
        for threshold in thresholds
    }
    condition = (
        largest / float(singular_values[parameters - 1])
        if exact_rank == parameters
        else float("inf")
    )
    unstable = (
        exact_rank < parameters
        or condition
        >= float(configuration["diagnostics"]["unstable_condition_number_min"])
        or effective[1e-6] < parameters
        or float(singular_values[parameters - 1])
        <= float(configuration["diagnostics"]["unstable_visibility_max"])
    )
    return exact_rank, effective, condition, unstable


def spectrum_only(operator_matrix: np.ndarray) -> np.ndarray:
    """Return physical static-metric singular values, padded to parameter count."""
    rows, parameters = operator_matrix.shape
    transformed = operator_matrix * np.sqrt(4.0 * np.pi / rows)
    compact = np.linalg.svd(transformed, compute_uv=False)
    singular_values = np.zeros(parameters, dtype=np.float64)
    singular_values[: compact.size] = compact
    return singular_values


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    """Write deterministic CSV records."""
    records = list(rows)
    if not records:
        raise ValueError(f"No records for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=records[0].keys(),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)


def write_json(path: Path, value: object) -> None:
    """Write deterministic, strict JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def prepare_basis() -> int:
    """Build and certify the canonical sweep-averaged basis through degree 10."""
    configuration = load_json(CONFIGURATION_PATH)
    lmax = max(int(value) for value in configuration["basis"]["window_lmax"])
    geometry = operator.load_sweep_geometry(GEOMETRY_PATH)
    started = perf_counter()
    basis = operator.sweep_harmonics(geometry, lmax, chunk_sweeps=512)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(BASIS_PATH, basis, allow_pickle=False)
    metadata = {
        "basis_path": BASIS_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "basis_sha256": sha256_file(BASIS_PATH),
        "dtype": str(basis.dtype),
        "elapsed_seconds": perf_counter() - started,
        "geometry_path": GEOMETRY_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "geometry_sha256": sha256_file(GEOMETRY_PATH),
        "lmax": lmax,
        "shape": list(basis.shape),
        "sweeps": geometry.size,
    }
    write_json(BASIS_METADATA_PATH, metadata)
    print(json.dumps(metadata, indent=2))
    return 0


def load_basis_and_geometry() -> tuple[np.ndarray, operator.SweepGeometry]:
    """Load a certified basis cache and matching canonical geometry."""
    metadata = load_json(BASIS_METADATA_PATH)
    if metadata["geometry_sha256"] != sha256_file(GEOMETRY_PATH):
        raise RuntimeError("Geometry cache hash changed after basis preparation")
    if metadata["basis_sha256"] != sha256_file(BASIS_PATH):
        raise RuntimeError("Basis cache hash does not match its metadata")
    basis = np.load(BASIS_PATH, mmap_mode="r", allow_pickle=False)
    geometry = operator.load_sweep_geometry(GEOMETRY_PATH)
    if basis.shape[0] != geometry.size:
        raise RuntimeError("Basis and geometry sweep counts differ")
    return basis, geometry


def make_window_result(
    *,
    model: str,
    lmax: int,
    duration_hours: float,
    window_id: int,
    rows: int,
    singular_values: np.ndarray,
    configuration: dict[str, Any],
    recovery_configuration: dict[str, Any],
    detailed: operator.SpectrumDiagnostics | None = None,
) -> WindowResult:
    """Apply all frozen per-window classifications."""
    parameters = (lmax + 1) ** 2
    exact_rank, effective, condition, unstable = classify_spectrum(
        singular_values, rows, parameters, configuration
    )
    median, p90 = covariance_recovery_quantiles(
        singular_values,
        rows,
        parameters,
        noise_sigma_v=float(recovery_configuration["gate_noise_sigma_v"]),
        truth_rms_v=float(configuration["injections"]["truth_global_rms_v"]),
        draws=int(recovery_configuration["gate_draws"]),
        seed_namespace=str(configuration["injections"]["seed_namespace"]),
    )
    recovery_pass = median < float(
        configuration["injections"]["median_relative_global_rms_error_max"]
    ) and p90 < float(configuration["injections"]["p90_relative_global_rms_error_max"])
    duration_ns = round(duration_hours * 3600.0 * 1e9)
    start = window_id * duration_ns
    leverage_median = None
    leverage_p99 = None
    resolution_trace = None
    if detailed is not None:
        leverage_median = float(np.median(detailed.leverage))
        leverage_p99 = float(np.quantile(detailed.leverage, 0.99))
        resolution_trace = detailed.resolution_trace[1e-6]
    return WindowResult(
        model=model,
        lmax=lmax,
        duration_hours=duration_hours,
        window_id=window_id,
        start_utc=iso_utc(start),
        stop_utc=iso_utc(start + duration_ns),
        rows=rows,
        parameters=parameters,
        exact_rank=exact_rank,
        effective_rank_1e3=effective[1e-3],
        effective_rank_1e6=effective[1e-6],
        effective_rank_1e9=effective[1e-9],
        condition_number=condition,
        visibility_min=float(singular_values[parameters - 1]),
        visibility_max=float(singular_values[0]),
        unstable=unstable,
        recovery_median_relative=median,
        recovery_p90_relative=p90,
        recovery_pass=recovery_pass,
        leverage_median=leverage_median,
        leverage_p99=leverage_p99,
        resolution_trace_1e6=resolution_trace,
    )


def summarize_windows(
    results: list[WindowResult],
    *,
    empty_windows: int,
    amendment: dict[str, Any],
) -> dict[str, object]:
    """Summarize window fractions and frozen class-level gates."""
    unstable = np.asarray([row.unstable for row in results])
    recovery_pass = np.asarray([row.recovery_pass for row in results])
    visibility = np.asarray([row.visibility_min for row in results])
    stable_fraction = float(np.mean(~unstable))
    recovery_fraction = float(np.mean(recovery_pass))
    witness_fraction = float(
        np.mean(
            visibility
            <= float(
                load_json(CONFIGURATION_PATH)["diagnostics"]["unstable_visibility_max"]
            )
        )
    )
    class_pass = (
        stable_fraction >= 0.9
        and recovery_fraction
        >= float(amendment["outcome_tree"]["recovery_pass_window_fraction_min"])
        and witness_fraction
        < float(amendment["outcome_tree"]["material_witness_window_fraction_min"])
    )
    return {
        "class_pass": class_pass,
        "eligible_nonempty_windows": len(results),
        "empty_scheduled_windows": empty_windows,
        "failed_recovery_fraction": float(np.mean(~recovery_pass)),
        "material_witness": witness_fraction
        >= float(amendment["outcome_tree"]["material_witness_window_fraction_min"]),
        "recovery_pass_fraction": recovery_fraction,
        "stable_fraction": stable_fraction,
        "unstable_fraction": float(np.mean(unstable)),
        "visibility_minimum": float(np.min(visibility)),
        "visibility_median": float(np.median(visibility)),
        "visibility_p90": float(np.quantile(visibility, 0.9)),
        "witness_window_fraction": witness_fraction,
    }


def run_primary() -> int:
    """Run direct physical-metric SVD for the frozen 24-hour degree-5 audit."""
    configuration = load_json(CONFIGURATION_PATH)
    amendment = load_json(AMENDMENT_03_PATH)
    basis, geometry = load_basis_and_geometry()
    lmax = int(configuration["basis"]["primary_lmax"])
    duration = float(configuration["basis"]["primary_window_hours"])
    parameters = (lmax + 1) ** 2
    field_gram = operator.static_field_gram(parameters)
    windows, empty_windows = iter_windows(geometry.midpoint_utc_ns, duration)
    results: list[WindowResult] = []
    witness_modes = np.empty((len(windows), parameters), dtype=np.float64)
    spectra = np.empty((len(windows), parameters), dtype=np.float64)
    for completed, (window_id, indices) in enumerate(windows, start=1):
        matrix = np.asarray(basis[indices, :parameters])
        diagnostics = operator.metric_spectrum(matrix, field_gram)
        if diagnostics.smallest_physical_mode is None:
            raise RuntimeError("Primary SVD did not return a witness mode")
        witness_modes[completed - 1] = diagnostics.smallest_physical_mode * 100.0
        spectra[completed - 1] = diagnostics.singular_values
        results.append(
            make_window_result(
                model="primary_independent_window",
                lmax=lmax,
                duration_hours=duration,
                window_id=window_id,
                rows=len(indices),
                singular_values=diagnostics.singular_values,
                configuration=configuration,
                recovery_configuration=amendment["recovery"],
                detailed=diagnostics,
            )
        )
        if completed % 50 == 0 or completed == len(windows):
            print(f"primary [{completed}/{len(windows)}]")
    write_csv(RESULTS_DIR / "primary-windows.csv", (asdict(row) for row in results))
    np.savez_compressed(
        RESULTS_DIR / "primary-spectra-witnesses.npz",
        window_id=np.asarray([row.window_id for row in results], dtype=np.int64),
        singular_values=spectra,
        witness_coefficients_100v=witness_modes,
    )
    summary = summarize_windows(
        results, empty_windows=empty_windows, amendment=amendment
    )
    summary.update(
        {
            "basis_cache_sha256": sha256_file(BASIS_PATH),
            "geometry_cache_sha256": sha256_file(GEOMETRY_PATH),
            "lmax": lmax,
            "parameters": parameters,
            "window_hours": duration,
        }
    )
    write_json(RESULTS_DIR / "primary-summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


def run_grid() -> int:
    """Run the complete frozen independent-window lmax/duration grid."""
    configuration = load_json(CONFIGURATION_PATH)
    amendment = load_json(AMENDMENT_03_PATH)
    basis, geometry = load_basis_and_geometry()
    summaries: list[dict[str, object]] = []
    for duration in configuration["basis"]["window_hours"]:
        windows, empty_windows = iter_windows(geometry.midpoint_utc_ns, float(duration))
        for lmax_value in configuration["basis"]["window_lmax"]:
            lmax = int(lmax_value)
            parameters = (lmax + 1) ** 2
            results: list[WindowResult] = []
            started = perf_counter()
            for window_id, indices in windows:
                matrix = np.asarray(basis[indices, :parameters])
                singular_values = spectrum_only(matrix)
                results.append(
                    make_window_result(
                        model="independent_window_grid",
                        lmax=lmax,
                        duration_hours=float(duration),
                        window_id=window_id,
                        rows=len(indices),
                        singular_values=singular_values,
                        configuration=configuration,
                        recovery_configuration=amendment["recovery"],
                    )
                )
            summary = summarize_windows(
                results, empty_windows=empty_windows, amendment=amendment
            )
            summary.update(
                {
                    "duration_hours": float(duration),
                    "elapsed_seconds": perf_counter() - started,
                    "lmax": lmax,
                    "parameters": parameters,
                }
            )
            summaries.append(summary)
            print(
                f"grid duration={duration:g}h lmax={lmax}: "
                f"stable={summary['stable_fraction']:.3f} "
                f"recovery={summary['recovery_pass_fraction']:.3f}"
            )
    write_csv(RESULTS_DIR / "window-grid-summary.csv", summaries)
    return 0


def tsqr_spectrum(chunks: Iterator[np.ndarray], rows: int) -> np.ndarray:
    """Compute a stable spectrum from bounded-memory transformed chunks."""
    reduced: list[np.ndarray] = []
    parameters: int | None = None
    for chunk in chunks:
        if parameters is None:
            parameters = chunk.shape[1]
        _, r_factor = np.linalg.qr(chunk, mode="reduced")
        reduced.append(r_factor)
        if sum(part.shape[0] for part in reduced) >= 8 * parameters:
            _, r_factor = np.linalg.qr(np.vstack(reduced), mode="reduced")
            reduced = [r_factor]
    if parameters is None:
        raise ValueError("No TSQR chunks")
    _, final_r = np.linalg.qr(np.vstack(reduced), mode="reduced")
    compact = np.linalg.svd(final_r / np.sqrt(rows), compute_uv=False)
    singular_values = np.zeros(parameters, dtype=np.float64)
    singular_values[: compact.size] = compact
    return singular_values


def cached_static_chunks(
    basis: np.ndarray,
    parameters: int,
    *,
    chunk_rows: int = 20_000,
) -> Iterator[np.ndarray]:
    """Yield globally whitened static basis chunks."""
    scale = np.sqrt(4.0 * np.pi)
    for start in range(0, basis.shape[0], chunk_rows):
        yield np.asarray(basis[start : start + chunk_rows, :parameters]) * scale


def generated_static_chunks(
    geometry: operator.SweepGeometry,
    lmax: int,
    *,
    chunk_rows: int = 2_000,
) -> Iterator[np.ndarray]:
    """Yield high-degree globally whitened chunks directly from geometry."""
    scale = np.sqrt(4.0 * np.pi)
    for start in range(0, geometry.size, chunk_rows):
        stop = min(start + chunk_rows, geometry.size)
        spatial = operator.sweep_harmonics(
            slice_geometry(geometry, slice(start, stop)),
            lmax,
            chunk_sweeps=512,
        )
        yield spatial * scale


def run_static() -> int:
    """Run whole-mission static controls through the frozen degree grid."""
    configuration = load_json(CONFIGURATION_PATH)
    amendment = load_json(AMENDMENT_03_PATH)
    basis, geometry = load_basis_and_geometry()
    rows: list[dict[str, object]] = []
    for lmax_value in configuration["basis"]["static_lmax"]:
        lmax = int(lmax_value)
        parameters = (lmax + 1) ** 2
        started = perf_counter()
        chunks = (
            cached_static_chunks(basis, parameters)
            if lmax <= 10
            else generated_static_chunks(geometry, lmax)
        )
        singular_values = tsqr_spectrum(chunks, geometry.size)
        exact_rank, effective, condition, unstable = classify_spectrum(
            singular_values, geometry.size, parameters, configuration
        )
        median, p90 = covariance_recovery_quantiles(
            singular_values,
            geometry.size,
            parameters,
            noise_sigma_v=float(amendment["recovery"]["gate_noise_sigma_v"]),
            truth_rms_v=float(configuration["injections"]["truth_global_rms_v"]),
            draws=int(amendment["recovery"]["gate_draws"]),
            seed_namespace=str(configuration["injections"]["seed_namespace"]),
        )
        recovery_pass = median < float(
            configuration["injections"]["median_relative_global_rms_error_max"]
        ) and p90 < float(
            configuration["injections"]["p90_relative_global_rms_error_max"]
        )
        record = {
            "condition_number": condition,
            "effective_rank_1e3": effective[1e-3],
            "effective_rank_1e6": effective[1e-6],
            "effective_rank_1e9": effective[1e-9],
            "elapsed_seconds": perf_counter() - started,
            "exact_rank": exact_rank,
            "lmax": lmax,
            "parameters": parameters,
            "recovery_median_relative": median,
            "recovery_p90_relative": p90,
            "recovery_pass": recovery_pass,
            "rows": geometry.size,
            "unstable": unstable,
            "visibility_max": float(singular_values[0]),
            "visibility_min": float(singular_values[-1]),
        }
        rows.append(record)
        print(json.dumps(record, indent=2))
    write_csv(RESULTS_DIR / "static-controls.csv", rows)
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("prepare", "primary", "grid", "static"),
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    handlers = {
        "prepare": prepare_basis,
        "primary": run_primary,
        "grid": run_grid,
        "static": run_static,
    }
    return handlers[args.phase]()


if __name__ == "__main__":
    raise SystemExit(main())
