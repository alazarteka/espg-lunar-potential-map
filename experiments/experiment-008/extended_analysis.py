#!/usr/bin/env python3
"""Run frozen temporal, sensitivity, injection, and blocked Experiment 008 phases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
import observation_operator as operator
import run_analysis

if TYPE_CHECKING:
    from collections.abc import Iterator

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "experiment-008"
RESULTS_DIR = EXPERIMENT_DIR / "results"
TEMPORAL_PATH = ARTIFACT_DIR / "canonical-sweep-temporal-synodic-l10.npy"
TEMPORAL_METADATA_PATH = ARTIFACT_DIR / "canonical-sweep-temporal-synodic-l10.json"
CONFIGURATION_PATH = EXPERIMENT_DIR / "configuration.json"
AMENDMENT_PATH = EXPERIMENT_DIR / "configuration-amendment-03.json"


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""
    with path.open(encoding="ascii") as stream:
        return dict(json.load(stream))


def metric_inverse_square_root(field_gram: np.ndarray) -> np.ndarray:
    """Return the symmetric inverse square root of a positive metric."""
    eigenvalues, eigenvectors = np.linalg.eigh(field_gram)
    tolerance = len(eigenvalues) * np.finfo(np.float64).eps * np.max(eigenvalues)
    if np.any(eigenvalues <= tolerance):
        raise ValueError("field metric must be positive definite")
    return (eigenvectors * np.reciprocal(np.sqrt(eigenvalues))) @ eigenvectors.T


def temporal_column_indices(lmax: int, source_lmax: int = 10) -> np.ndarray:
    """Select nested degree columns from a temporal-major cached operator."""
    target_spatial = (lmax + 1) ** 2
    source_spatial = (source_lmax + 1) ** 2
    return np.concatenate(
        [
            np.arange(block * source_spatial, block * source_spatial + target_spatial)
            for block in range(3)
        ]
    )


def contrast_indices(lmax: int, temporal_columns: int) -> np.ndarray:
    """Return all columns except the monopole in each temporal block."""
    spatial = (lmax + 1) ** 2
    return np.concatenate(
        [
            np.arange(block * spatial + 1, (block + 1) * spatial)
            for block in range(temporal_columns)
        ]
    )


def prepare_temporal() -> int:
    """Generate the exact row-product synodic degree-10 operator as a memmap."""
    configuration = load_json(CONFIGURATION_PATH)
    period_days = float(configuration["basis"]["synodic_period_days"])
    geometry = operator.load_sweep_geometry(run_analysis.GEOMETRY_PATH)
    lmax = max(int(value) for value in configuration["basis"]["temporal_lmax"])
    parameters = 3 * (lmax + 1) ** 2
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    cache = np.lib.format.open_memmap(
        TEMPORAL_PATH,
        mode="w+",
        dtype=np.float64,
        shape=(geometry.size, parameters),
    )
    started = perf_counter()
    chunk_rows = 2_000
    for start in range(0, geometry.size, chunk_rows):
        stop = min(start + chunk_rows, geometry.size)
        cache[start:stop] = operator.sweep_tensor_harmonics(
            run_analysis.slice_geometry(geometry, slice(start, stop)),
            lmax,
            "constant_synodic",
            period_days=period_days,
            epoch_ns=0,
            chunk_sweeps=500,
        )
        if stop % 20_000 == 0 or stop == geometry.size:
            print(f"temporal prepare [{stop}/{geometry.size}]")
    cache.flush()
    metadata = {
        "dtype": "float64",
        "elapsed_seconds": perf_counter() - started,
        "epoch_utc_ns": 0,
        "geometry_sha256": run_analysis.sha256_file(run_analysis.GEOMETRY_PATH),
        "lmax": lmax,
        "model": "constant_synodic",
        "path": TEMPORAL_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "period_days": period_days,
        "sha256": run_analysis.sha256_file(TEMPORAL_PATH),
        "shape": [geometry.size, parameters],
    }
    run_analysis.write_json(TEMPORAL_METADATA_PATH, metadata)
    print(json.dumps(metadata, indent=2))
    return 0


def load_temporal() -> np.ndarray:
    """Load and verify the exact temporal cache."""
    metadata = load_json(TEMPORAL_METADATA_PATH)
    if metadata["geometry_sha256"] != run_analysis.sha256_file(
        run_analysis.GEOMETRY_PATH
    ):
        raise RuntimeError("Temporal cache geometry hash mismatch")
    if metadata["sha256"] != run_analysis.sha256_file(TEMPORAL_PATH):
        raise RuntimeError("Temporal cache hash mismatch")
    return np.load(TEMPORAL_PATH, mmap_mode="r", allow_pickle=False)


def transformed_chunks(
    matrix: np.ndarray,
    columns: np.ndarray | slice,
    inverse_square_root: np.ndarray,
    *,
    chunk_rows: int = 20_000,
) -> Iterator[np.ndarray]:
    """Yield physical-metric-whitened matrix chunks for TSQR."""
    for start in range(0, matrix.shape[0], chunk_rows):
        chunk = np.asarray(matrix[start : start + chunk_rows, columns])
        yield chunk @ inverse_square_root


def temporal_interval() -> tuple[int, int]:
    """Return the frozen complete-sweep half-open metric interval."""
    complete = operator.load_sweep_geometry(
        run_analysis.GEOMETRY_PATH, complete_only=True
    )
    return int(np.min(complete.row_utc_ns)), int(np.max(complete.row_utc_ns)) + 1


def temporal_record(
    matrix: np.ndarray,
    columns: np.ndarray | slice,
    field_gram: np.ndarray,
    *,
    model: str,
    lmax: int,
    estimand: str,
    configuration: dict[str, Any],
    amendment: dict[str, Any],
) -> dict[str, object]:
    """Compute one whole-mission temporal/static TSQR record."""
    parameters = field_gram.shape[0]
    started = perf_counter()
    inverse_square_root = metric_inverse_square_root(field_gram)
    singular_values = run_analysis.tsqr_spectrum(
        transformed_chunks(matrix, columns, inverse_square_root),
        matrix.shape[0],
    )
    exact_rank, effective, condition, unstable = run_analysis.classify_spectrum(
        singular_values,
        matrix.shape[0],
        parameters,
        configuration,
    )
    median, p90 = run_analysis.covariance_recovery_quantiles(
        singular_values,
        matrix.shape[0],
        parameters,
        noise_sigma_v=float(amendment["recovery"]["gate_noise_sigma_v"]),
        truth_rms_v=float(configuration["injections"]["truth_global_rms_v"]),
        draws=int(amendment["recovery"]["gate_draws"]),
        seed_namespace=str(configuration["injections"]["seed_namespace"]),
    )
    recovery_pass = median < float(
        configuration["injections"]["median_relative_global_rms_error_max"]
    ) and p90 < float(configuration["injections"]["p90_relative_global_rms_error_max"])
    return {
        "condition_number": condition,
        "effective_rank_1e3": effective[1e-3],
        "effective_rank_1e6": effective[1e-6],
        "effective_rank_1e9": effective[1e-9],
        "elapsed_seconds": perf_counter() - started,
        "estimand": estimand,
        "exact_rank": exact_rank,
        "lmax": lmax,
        "model": model,
        "parameters": parameters,
        "recovery_median_relative": median,
        "recovery_p90_relative": p90,
        "recovery_pass": recovery_pass,
        "rows": matrix.shape[0],
        "unstable": unstable,
        "visibility_max": float(singular_values[0]),
        "visibility_min": float(singular_values[-1]),
    }


def run_temporal() -> int:
    """Run whole-mission constant/synodic and contrast-only controls."""
    configuration = load_json(CONFIGURATION_PATH)
    amendment = load_json(AMENDMENT_PATH)
    basis, _ = run_analysis.load_basis_and_geometry()
    temporal = load_temporal()
    start_utc_ns, stop_utc_ns = temporal_interval()
    period_days = float(configuration["basis"]["synodic_period_days"])
    rows: list[dict[str, object]] = []
    for model in configuration["basis"]["temporal_models"]:
        temporal_columns = 1 if model == "constant" else 3
        source = basis if model == "constant" else temporal
        for lmax_value in configuration["basis"]["temporal_lmax"]:
            lmax = int(lmax_value)
            spatial = (lmax + 1) ** 2
            if model == "constant":
                columns: np.ndarray | slice = slice(0, spatial)
            else:
                columns = temporal_column_indices(lmax)
            field_gram = operator.temporal_field_gram(
                lmax,
                model,
                start_utc_ns=start_utc_ns,
                stop_utc_ns=stop_utc_ns,
                period_days=period_days,
                epoch_ns=0,
            )
            record = temporal_record(
                source,
                columns,
                field_gram,
                model=model,
                lmax=lmax,
                estimand="oracle_with_monopole",
                configuration=configuration,
                amendment=amendment,
            )
            rows.append(record)
            print(json.dumps(record, indent=2))

            keep = contrast_indices(lmax, temporal_columns)
            source_columns = (
                keep
                if model == "constant"
                else np.asarray(columns, dtype=np.int64)[keep]
            )
            contrast_gram = field_gram[np.ix_(keep, keep)]
            contrast = temporal_record(
                source,
                source_columns,
                contrast_gram,
                model=model,
                lmax=lmax,
                estimand="contrast_without_l0",
                configuration=configuration,
                amendment=amendment,
            )
            rows.append(contrast)
            print(json.dumps(contrast, indent=2))
    run_analysis.write_csv(RESULTS_DIR / "temporal-controls.csv", rows)
    return 0


def window_class_summary(
    basis: np.ndarray,
    midpoint_utc_ns: np.ndarray,
    *,
    name: str,
    parameters: int,
    configuration: dict[str, Any],
    amendment: dict[str, Any],
    duration_hours: float = 24.0,
) -> dict[str, object]:
    """Summarize a static-metric window class with an arbitrary column count."""
    windows, empty = run_analysis.iter_windows(midpoint_utc_ns, duration_hours)
    unstable_flags: list[bool] = []
    recovery_flags: list[bool] = []
    visibility: list[float] = []
    conditions: list[float] = []
    for _, indices in windows:
        singular_values = run_analysis.spectrum_only(
            np.asarray(basis[indices, :parameters])
        )
        _, _, condition, unstable = run_analysis.classify_spectrum(
            singular_values,
            len(indices),
            parameters,
            configuration,
        )
        median, p90 = run_analysis.covariance_recovery_quantiles(
            singular_values,
            len(indices),
            parameters,
            noise_sigma_v=float(amendment["recovery"]["gate_noise_sigma_v"]),
            truth_rms_v=float(configuration["injections"]["truth_global_rms_v"]),
            draws=int(amendment["recovery"]["gate_draws"]),
            seed_namespace=str(configuration["injections"]["seed_namespace"]),
        )
        recovery = median < float(
            configuration["injections"]["median_relative_global_rms_error_max"]
        ) and p90 < float(
            configuration["injections"]["p90_relative_global_rms_error_max"]
        )
        unstable_flags.append(unstable)
        recovery_flags.append(recovery)
        visibility.append(float(singular_values[-1]))
        conditions.append(condition)
    unstable_array = np.asarray(unstable_flags)
    recovery_array = np.asarray(recovery_flags)
    visibility_array = np.asarray(visibility)
    witness_fraction = float(
        np.mean(
            visibility_array
            <= float(configuration["diagnostics"]["unstable_visibility_max"])
        )
    )
    stable_fraction = float(np.mean(~unstable_array))
    recovery_fraction = float(np.mean(recovery_array))
    class_pass = (
        stable_fraction
        >= float(configuration["outcomes"]["a_stable_window_fraction_min"])
        and recovery_fraction
        >= float(amendment["outcome_tree"]["recovery_pass_window_fraction_min"])
        and witness_fraction
        < float(amendment["outcome_tree"]["material_witness_window_fraction_min"])
    )
    return {
        "class_pass": class_pass,
        "condition_median": float(np.median(conditions)),
        "duration_hours": duration_hours,
        "eligible_nonempty_windows": len(windows),
        "empty_scheduled_windows": empty,
        "name": name,
        "parameters": parameters,
        "recovery_pass_fraction": recovery_fraction,
        "stable_fraction": stable_fraction,
        "visibility_median": float(np.median(visibility_array)),
        "visibility_minimum": float(np.min(visibility_array)),
        "witness_window_fraction": witness_fraction,
    }


def sweep_keys(geometry: operator.SweepGeometry) -> np.ndarray:
    """Return unique structured source/spec keys for cohort intersection."""
    keys = np.empty(
        geometry.size,
        dtype=np.dtype([("source", np.int64), ("spec", np.int64)]),
    )
    keys["source"] = geometry.source_index
    keys["spec"] = geometry.spec_no
    if np.unique(keys).size != geometry.size:
        raise RuntimeError("Sweep source/spec keys are not unique")
    return keys


def common_cohort(
    left: operator.SweepGeometry,
    right: operator.SweepGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned indices for sweeps present in both geometry variants."""
    _, left_index, right_index = np.intersect1d(
        sweep_keys(left),
        sweep_keys(right),
        assume_unique=True,
        return_indices=True,
    )
    return left_index, right_index


def build_spatial_basis(
    geometry: operator.SweepGeometry,
    label: str,
    *,
    reducer: operator.Reducer = "mean",
    longitude_offset_deg: np.ndarray | None = None,
) -> np.ndarray:
    """Build or load a replayable degree-5 sensitivity basis cache."""
    path = ARTIFACT_DIR / f"sensitivity-basis-{label}-l5.npy"
    if path.exists():
        cached = np.load(path, mmap_mode="r", allow_pickle=False)
        if cached.shape != (geometry.size, 36):
            raise RuntimeError(f"Sensitivity cache shape mismatch: {path}")
        return cached
    basis = operator.sweep_harmonics(
        geometry,
        5,
        reducer=reducer,
        chunk_sweeps=512,
        longitude_offset_deg=longitude_offset_deg,
    )
    np.save(path, basis, allow_pickle=False)
    return np.load(path, mmap_mode="r", allow_pickle=False)


def row_duplication_summary(
    geometry: operator.SweepGeometry,
    *,
    configuration: dict[str, Any],
    amendment: dict[str, Any],
) -> list[dict[str, object]]:
    """Compare total-weight and invalid independent-row duplication."""
    windows, empty = run_analysis.iter_windows(geometry.midpoint_utc_ns, 24.0)
    methods = ("row_footprints_total_sweep_weight", "invalid_independent_rows")
    unstable: dict[str, list[bool]] = {method: [] for method in methods}
    recovery: dict[str, list[bool]] = {method: [] for method in methods}
    visibility: dict[str, list[float]] = {method: [] for method in methods}
    for completed, (_, indices) in enumerate(windows, start=1):
        subset = run_analysis.slice_geometry(geometry, indices)
        valid = subset.valid_rows.ravel()
        row_basis = operator.real_spherical_harmonics(
            subset.latitude.ravel()[valid],
            subset.longitude.ravel()[valid],
            5,
        )
        counts = np.sum(subset.valid_rows, axis=1)
        row_sweep = np.repeat(np.arange(subset.size), subset.valid_rows.shape[1])[valid]
        weighted = row_basis / np.sqrt(counts[row_sweep, None])
        matrices = {
            methods[0]: (weighted, subset.size),
            methods[1]: (row_basis, row_basis.shape[0]),
        }
        for method, (matrix, noise_rows) in matrices.items():
            transformed = matrix * np.sqrt(4.0 * np.pi / noise_rows)
            singular_values = np.linalg.svd(transformed, compute_uv=False)
            _, _, _, failed = run_analysis.classify_spectrum(
                singular_values,
                noise_rows,
                36,
                configuration,
            )
            median, p90 = run_analysis.covariance_recovery_quantiles(
                singular_values,
                noise_rows,
                36,
                noise_sigma_v=float(amendment["recovery"]["gate_noise_sigma_v"]),
                truth_rms_v=float(configuration["injections"]["truth_global_rms_v"]),
                draws=int(amendment["recovery"]["gate_draws"]),
                seed_namespace=str(configuration["injections"]["seed_namespace"]),
            )
            passes = median < float(
                configuration["injections"]["median_relative_global_rms_error_max"]
            ) and p90 < float(
                configuration["injections"]["p90_relative_global_rms_error_max"]
            )
            unstable[method].append(failed)
            recovery[method].append(passes)
            visibility[method].append(float(singular_values[-1]))
        if completed % 100 == 0 or completed == len(windows):
            print(f"row duplication [{completed}/{len(windows)}]")
    return [
        (
            {
                "class_pass": False,
                "condition_median": "",
                "duration_hours": 24.0,
                "eligible_nonempty_windows": len(windows),
                "empty_scheduled_windows": empty,
                "name": method,
                "parameters": 36,
                "recovery_pass_fraction": float(np.mean(recovery[method])),
                "stable_fraction": float(np.mean(~np.asarray(unstable[method]))),
                "visibility_median": float(np.median(visibility[method])),
                "visibility_minimum": float(np.min(visibility[method])),
                "witness_window_fraction": float(
                    np.mean(
                        np.asarray(visibility[method])
                        <= float(
                            configuration["diagnostics"]["unstable_visibility_max"]
                        )
                    )
                ),
            }
        )
        for method in methods
    ]


def run_sensitivities() -> int:
    """Run frozen geometry, reducer, gauge, rotation, and duplication checks."""
    configuration = load_json(CONFIGURATION_PATH)
    amendment = load_json(AMENDMENT_PATH)
    canonical_basis, canonical = run_analysis.load_basis_and_geometry()
    records = [
        window_class_summary(
            canonical_basis,
            canonical.midpoint_utc_ns,
            name="canonical_native_mean",
            parameters=36,
            configuration=configuration,
            amendment=amendment,
        )
    ]

    midpoint_basis = build_spatial_basis(
        canonical, "corrected-midpoint", reducer="midpoint"
    )
    records.append(
        window_class_summary(
            midpoint_basis,
            canonical.midpoint_utc_ns,
            name="canonical_midpoint_row",
            parameters=36,
            configuration=configuration,
            amendment=amendment,
        )
    )

    complete = operator.load_sweep_geometry(
        run_analysis.GEOMETRY_PATH, complete_only=True
    )
    complete_basis = build_spatial_basis(complete, "corrected-complete")
    records.append(
        window_class_summary(
            complete_basis,
            complete.midpoint_utc_ns,
            name="canonical_complete_15_rows",
            parameters=36,
            configuration=configuration,
            amendment=amendment,
        )
    )

    for variant in ("legacy", "frame_mixed", "submitted_legacy"):
        geometry = operator.load_sweep_geometry(
            run_analysis.GEOMETRY_PATH, variant=variant
        )
        variant_basis = build_spatial_basis(geometry, variant)
        records.append(
            window_class_summary(
                variant_basis,
                geometry.midpoint_utc_ns,
                name=f"{variant}_native",
                parameters=36,
                configuration=configuration,
                amendment=amendment,
            )
        )
        canonical_index, variant_index = common_cohort(canonical, geometry)
        records.append(
            window_class_summary(
                np.asarray(canonical_basis[canonical_index, :36]),
                canonical.midpoint_utc_ns[canonical_index],
                name=f"canonical_paired_with_{variant}",
                parameters=36,
                configuration=configuration,
                amendment=amendment,
            )
        )
        records.append(
            window_class_summary(
                np.asarray(variant_basis[variant_index]),
                geometry.midpoint_utc_ns[variant_index],
                name=f"{variant}_paired_with_canonical",
                parameters=36,
                configuration=configuration,
                amendment=amendment,
            )
        )

    period_days = float(configuration["basis"]["synodic_period_days"])
    offset = -360.0 * (
        canonical.row_utc_ns.astype(np.float64) / (run_analysis.DAY_NS * period_days)
    )
    offset = (offset + 180.0) % 360.0 - 180.0
    corotating_basis = build_spatial_basis(
        canonical,
        "solar-corotating",
        longitude_offset_deg=offset,
    )
    records.append(
        window_class_summary(
            corotating_basis,
            canonical.midpoint_utc_ns,
            name="solar_corotating_native",
            parameters=36,
            configuration=configuration,
            amendment=amendment,
        )
    )
    records.append(
        window_class_summary(
            np.asarray(canonical_basis[:, 1:36]),
            canonical.midpoint_utc_ns,
            name="contrast_without_l0",
            parameters=35,
            configuration=configuration,
            amendment=amendment,
        )
    )
    records.extend(
        row_duplication_summary(
            canonical,
            configuration=configuration,
            amendment=amendment,
        )
    )
    run_analysis.write_csv(RESULTS_DIR / "sensitivity-summary.csv", records)
    return 0


def run_column_normalization() -> int:
    """Record the original empirical column-normalized secondary spectrum."""
    basis, geometry = run_analysis.load_basis_and_geometry()
    windows, _ = run_analysis.iter_windows(geometry.midpoint_utc_ns, 24.0)
    field_gram = operator.static_field_gram(36)
    rows: list[dict[str, object]] = []
    for window_id, indices in windows:
        matrix = np.asarray(basis[indices, :36])
        diagnostics = operator.direct_spectrum(
            matrix,
            include_explicit_null=False,
        )
        sampled_rms, visibility = operator.mode_visibility(
            matrix,
            diagnostics.smallest_physical_mode,
            field_gram,
        )
        rows.append(
            {
                "condition_number": diagnostics.condition_number,
                "effective_rank_1e3": diagnostics.effective_ranks[1e-3],
                "effective_rank_1e6": diagnostics.effective_ranks[1e-6],
                "effective_rank_1e9": diagnostics.effective_ranks[1e-9],
                "exact_rank": diagnostics.exact_rank,
                "physical_visibility_of_selected_weakest_mode": visibility,
                "rows": len(indices),
                "sampled_rms_for_100v_selected_mode": sampled_rms,
                "window_id": window_id,
            }
        )
    run_analysis.write_csv(RESULTS_DIR / "column-normalization-sensitivity.csv", rows)
    condition = np.asarray([float(row["condition_number"]) for row in rows])
    visibility = np.asarray(
        [float(row["physical_visibility_of_selected_weakest_mode"]) for row in rows]
    )
    summary = {
        "condition_maximum": float(np.max(condition)),
        "condition_median": float(np.median(condition)),
        "metric": "empirical_unit_l2_column_secondary",
        "physical_visibility_maximum": float(np.max(visibility)),
        "physical_visibility_median": float(np.median(visibility)),
        "physical_visibility_minimum": float(np.min(visibility)),
        "windows": len(rows),
    }
    run_analysis.write_json(RESULTS_DIR / "column-normalization-summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


def broadband_truths(
    lmax: int,
    replicates: int,
    *,
    seed: int,
    shell: int | None = None,
) -> np.ndarray:
    """Draw 100 V metric-coordinate truths with frozen degree power."""
    degree, _ = operator.harmonic_indices(lmax)
    rng = np.random.Generator(np.random.PCG64(seed))
    values = rng.standard_normal((replicates, degree.size))
    if shell is None:
        scale = np.empty(degree.size, dtype=np.float64)
        for ell in range(lmax + 1):
            scale[degree == ell] = 1.0 / ((ell + 1) * np.sqrt(2 * ell + 1))
        values *= scale
    else:
        values[:, degree != shell] = 0.0
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms == 0.0):
        raise RuntimeError("Injection truth has zero norm")
    return values * (100.0 / norms[:, None])


def recover_methods(
    matrix: np.ndarray,
    observations: np.ndarray,
    *,
    tsvd_thresholds: list[float],
    ridge_penalties: list[float],
) -> dict[str, np.ndarray]:
    """Recover metric coordinates with all frozen linear estimators."""
    u_matrix, singular_values, vh_matrix = np.linalg.svd(matrix, full_matrices=False)
    projected = u_matrix.T @ observations.T
    estimates: dict[str, np.ndarray] = {}
    unregularized = vh_matrix.T @ (projected / singular_values[:, None])
    estimates["unregularized"] = unregularized.T
    for threshold in tsvd_thresholds:
        keep = singular_values > threshold * singular_values[0]
        fitted = vh_matrix[keep].T @ (projected[keep] / singular_values[keep, None])
        estimates[f"tsvd_{threshold:g}"] = fitted.T

    column_norms = np.linalg.norm(matrix, axis=0)
    normalized = matrix / column_norms
    gram = normalized.T @ normalized
    rhs = normalized.T @ observations.T
    identity = np.eye(matrix.shape[1])
    for penalty in ridge_penalties:
        normalized_coefficients = np.linalg.solve(
            gram + penalty * identity,
            rhs,
        )
        estimates[f"ridge_{penalty:g}"] = (
            normalized_coefficients / column_norms[:, None]
        ).T
    return estimates


def run_injections() -> int:
    """Run explicit frozen primary-window estimator and regularization trials."""
    configuration = load_json(CONFIGURATION_PATH)
    basis, geometry = run_analysis.load_basis_and_geometry()
    namespace = str(configuration["injections"]["seed_namespace"])
    replicates = int(configuration["injections"]["replicates"])
    noise_levels = [
        float(value) for value in configuration["injections"]["noise_sigma_v"]
    ]
    tsvd = [
        float(value)
        for value in configuration["injections"]["tsvd_relative_thresholds"]
    ]
    ridge = [float(value) for value in configuration["injections"]["ridge_penalties"]]
    windows, _ = run_analysis.iter_windows(geometry.midpoint_utc_ns, 24.0)
    error_rows: list[dict[str, object]] = []
    window_pass_rows: list[dict[str, object]] = []
    for completed, (window_id, indices) in enumerate(windows, start=1):
        # Metric coordinates d have global RMS ||d||. For static harmonics,
        # c = sqrt(4 pi) d and y = A c = M d.
        matrix = np.asarray(basis[indices, :36]) * np.sqrt(4.0 * np.pi)
        _, _, vh_matrix = np.linalg.svd(matrix, full_matrices=False)
        truth_classes: list[tuple[str, np.ndarray]] = []
        root = run_analysis.seed_from_parts(namespace, "window", 24.0, 5, window_id)
        truth_classes.append(("broadband", broadband_truths(5, replicates, seed=root)))
        truth_classes.extend(
            (
                (
                    f"degree_shell_{ell}",
                    broadband_truths(
                        5,
                        replicates,
                        seed=run_analysis.seed_from_parts(root, "shell", ell),
                        shell=ell,
                    ),
                )
            )
            for ell in range(6)
        )
        truth_classes.append(
            (
                "weakest_mode",
                np.repeat(100.0 * vh_matrix[-1][None, :], replicates, axis=0),
            )
        )
        for truth_name, truths in truth_classes:
            active_noise = noise_levels if truth_name == "broadband" else [25.0]
            for noise_sigma in active_noise:
                rng = np.random.Generator(
                    np.random.PCG64(
                        run_analysis.seed_from_parts(
                            root,
                            truth_name,
                            noise_sigma,
                            "noise",
                        )
                    )
                )
                noise = rng.normal(0.0, noise_sigma, (replicates, matrix.shape[0]))
                observations = truths @ matrix.T + noise
                estimates = recover_methods(
                    matrix,
                    observations,
                    tsvd_thresholds=tsvd,
                    ridge_penalties=ridge,
                )
                for method, estimate in estimates.items():
                    errors = np.linalg.norm(estimate - truths, axis=1) / 100.0
                    error_rows.extend(
                        {
                            "error_relative": float(error),
                            "method": method,
                            "noise_sigma_v": noise_sigma,
                            "replicate": replicate,
                            "truth_class": truth_name,
                            "window_id": window_id,
                        }
                        for replicate, error in enumerate(errors)
                    )
                    if (
                        truth_name == "broadband"
                        and noise_sigma
                        == float(
                            configuration["injections"]["recovery_gate_noise_sigma_v"]
                        )
                        and method == "unregularized"
                    ):
                        window_pass_rows.append(
                            {
                                "median_relative": float(
                                    np.quantile(errors, 0.5, method="linear")
                                ),
                                "p90_relative": float(
                                    np.quantile(errors, 0.9, method="linear")
                                ),
                                "passes": bool(
                                    np.quantile(errors, 0.5, method="linear")
                                    < float(
                                        configuration["injections"][
                                            "median_relative_global_rms_error_max"
                                        ]
                                    )
                                    and np.quantile(errors, 0.9, method="linear")
                                    < float(
                                        configuration["injections"][
                                            "p90_relative_global_rms_error_max"
                                        ]
                                    )
                                ),
                                "window_id": window_id,
                            }
                        )
        if completed % 50 == 0 or completed == len(windows):
            print(f"injections [{completed}/{len(windows)}]")

    run_analysis.write_csv(ARTIFACT_DIR / "injection-errors.csv", error_rows)
    run_analysis.write_csv(RESULTS_DIR / "injection-window-gates.csv", window_pass_rows)
    # Compact aggregation is the interpretation table; raw errors remain for replay.
    grouped: dict[tuple[str, float, str], list[float]] = {}
    for row in error_rows:
        key = (
            str(row["truth_class"]),
            float(row["noise_sigma_v"]),
            str(row["method"]),
        )
        grouped.setdefault(key, []).append(float(row["error_relative"]))
    summaries = []
    for (truth_class, noise_sigma, method), errors in sorted(grouped.items()):
        summaries.append(
            {
                "error_median_relative": float(
                    np.quantile(errors, 0.5, method="linear")
                ),
                "error_p90_relative": float(np.quantile(errors, 0.9, method="linear")),
                "method": method,
                "noise_sigma_v": noise_sigma,
                "samples": len(errors),
                "truth_class": truth_class,
            }
        )
    run_analysis.write_csv(RESULTS_DIR / "injection-summary.csv", summaries)
    return 0


def circular_mean_longitude(geometry: operator.SweepGeometry) -> np.ndarray:
    """Return one connected-row circular mean longitude per sweep."""
    radians = np.deg2rad(geometry.longitude)
    valid = geometry.valid_rows
    sine = np.sum(np.where(valid, np.sin(radians), 0.0), axis=1)
    cosine = np.sum(np.where(valid, np.cos(radians), 0.0), axis=1)
    return np.rad2deg(np.arctan2(sine, cosine))


def gaussian_cap(
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
    *,
    center_latitude_deg: float,
    center_longitude_deg: float,
    fwhm_deg: float,
) -> np.ndarray:
    """Evaluate a unit-amplitude great-circle Gaussian cap."""
    latitude = np.deg2rad(latitude_deg)
    longitude = np.deg2rad(longitude_deg)
    center_latitude = np.deg2rad(center_latitude_deg)
    center_longitude = np.deg2rad(center_longitude_deg)
    cosine_angle = np.sin(latitude) * np.sin(center_latitude) + np.cos(
        latitude
    ) * np.cos(center_latitude) * np.cos(longitude - center_longitude)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    sigma = np.deg2rad(fwhm_deg) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return np.exp(-0.5 * (angle / sigma) ** 2)


def cap_projection(
    center_latitude_deg: float,
    center_longitude_deg: float,
    fwhm_deg: float,
    *,
    lmax: int = 5,
) -> tuple[float, np.ndarray, float]:
    """Return cap amplitude, metric projection coordinates, and bias RMS."""
    nodes, weights = np.polynomial.legendre.leggauss(181)
    latitude = np.rad2deg(np.arcsin(nodes))
    longitude = np.linspace(-180.0, 180.0, 360, endpoint=False)
    lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    cap = gaussian_cap(
        lat_grid,
        lon_grid,
        center_latitude_deg=center_latitude_deg,
        center_longitude_deg=center_longitude_deg,
        fwhm_deg=fwhm_deg,
    )
    longitude_weight = 2.0 * np.pi / longitude.size
    mean_square = float(
        longitude_weight * np.sum(weights[:, None] * cap * cap) / (4.0 * np.pi)
    )
    amplitude = 100.0 / np.sqrt(mean_square)
    cap *= amplitude
    basis = operator.real_spherical_harmonics(lat_grid.ravel(), lon_grid.ravel(), lmax)
    integration_weight = np.repeat(weights, longitude.size) * longitude_weight
    coefficients = basis.T @ (cap.ravel() * integration_weight)
    metric_coordinates = coefficients / np.sqrt(4.0 * np.pi)
    projected_rms = float(np.linalg.norm(metric_coordinates))
    bias_rms = np.sqrt(max(100.0**2 - projected_rms**2, 0.0))
    return amplitude, metric_coordinates, bias_rms


def sweep_cap_values(
    geometry: operator.SweepGeometry,
    *,
    amplitude: float,
    center_latitude_deg: float,
    center_longitude_deg: float,
    fwhm_deg: float,
) -> np.ndarray:
    """Average an out-of-basis cap over connected rows in each sweep."""
    row_values = gaussian_cap(
        geometry.latitude,
        geometry.longitude,
        center_latitude_deg=center_latitude_deg,
        center_longitude_deg=center_longitude_deg,
        fwhm_deg=fwhm_deg,
    )
    row_values = np.where(geometry.valid_rows, amplitude * row_values, 0.0)
    return np.sum(row_values, axis=1) / np.sum(geometry.valid_rows, axis=1)


def run_caps() -> int:
    """Decompose localized-cap approximation bias and sampling alias error."""
    configuration = load_json(AMENDMENT_PATH)
    basis, geometry = run_analysis.load_basis_and_geometry()
    windows, _ = run_analysis.iter_windows(geometry.midpoint_utc_ns, 24.0)
    rows: list[dict[str, object]] = []
    for center_latitude, center_longitude in configuration["recovery"]["caps"][
        "centers_lat_lon_degrees"
    ]:
        for fwhm in configuration["recovery"]["caps"]["fwhm_degrees"]:
            amplitude, projected_truth, bias_rms = cap_projection(
                float(center_latitude),
                float(center_longitude),
                float(fwhm),
            )
            sampled_truth = sweep_cap_values(
                geometry,
                amplitude=amplitude,
                center_latitude_deg=float(center_latitude),
                center_longitude_deg=float(center_longitude),
                fwhm_deg=float(fwhm),
            )
            estimation_errors = []
            total_errors = []
            sampled_residuals = []
            for _, indices in windows:
                matrix = np.asarray(basis[indices, :36]) * np.sqrt(4.0 * np.pi)
                fitted, _, _, _ = np.linalg.lstsq(
                    matrix,
                    sampled_truth[indices],
                    rcond=None,
                )
                estimation = float(np.linalg.norm(fitted - projected_truth))
                estimation_errors.append(estimation)
                total_errors.append(np.sqrt(bias_rms**2 + estimation**2))
                sampled_residuals.append(
                    float(
                        np.sqrt(
                            np.mean((matrix @ fitted - sampled_truth[indices]) ** 2)
                        )
                    )
                )
            rows.append(
                {
                    "approximation_bias_rms_v": bias_rms,
                    "cap_amplitude_v": amplitude,
                    "center_latitude_deg": center_latitude,
                    "center_longitude_deg": center_longitude,
                    "estimation_error_median_v": float(np.median(estimation_errors)),
                    "estimation_error_p90_v": float(
                        np.quantile(estimation_errors, 0.9)
                    ),
                    "fwhm_deg": fwhm,
                    "sampled_residual_median_v": float(np.median(sampled_residuals)),
                    "total_global_error_median_v": float(np.median(total_errors)),
                    "total_global_error_p90_v": float(np.quantile(total_errors, 0.9)),
                    "windows": len(windows),
                }
            )
            print(json.dumps(rows[-1], indent=2))
    run_analysis.write_csv(RESULTS_DIR / "localized-cap-summary.csv", rows)
    return 0


def covariance_draw_errors(
    train_gram: np.ndarray,
    test_gram: np.ndarray,
    *,
    train_rows: int,
    test_rows: int,
    noise_sigma_v: float,
    draws: int,
    seed: int,
) -> tuple[float, float, float, float, float, float]:
    """Return global, train, and test median/p90 errors from fit covariance."""
    eigenvalues, eigenvectors = np.linalg.eigh(train_gram)
    tolerance = len(eigenvalues) * np.finfo(np.float64).eps * np.max(eigenvalues)
    if np.any(eigenvalues <= tolerance):
        return (float("inf"),) * 6
    covariance_root = (
        eigenvectors * (noise_sigma_v / np.sqrt(eigenvalues))
    ) @ eigenvectors.T
    rng = np.random.Generator(np.random.PCG64(seed))
    errors = rng.standard_normal((draws, len(eigenvalues))) @ covariance_root.T
    global_rms = np.linalg.norm(errors, axis=1)
    train_rms = np.sqrt(
        np.maximum(
            np.einsum("ni,ij,nj->n", errors, train_gram, errors) / train_rows,
            0.0,
        )
    )
    test_rms = np.sqrt(
        np.maximum(
            np.einsum("ni,ij,nj->n", errors, test_gram, errors) / test_rows,
            0.0,
        )
    )
    return (
        float(np.quantile(global_rms, 0.5, method="linear")),
        float(np.quantile(global_rms, 0.9, method="linear")),
        float(np.quantile(train_rms, 0.5, method="linear")),
        float(np.quantile(train_rms, 0.9, method="linear")),
        float(np.quantile(test_rms, 0.5, method="linear")),
        float(np.quantile(test_rms, 0.9, method="linear")),
    )


def blocked_record(
    matrix: np.ndarray,
    test_mask: np.ndarray,
    *,
    model: str,
    split: str,
    seed: int,
    noise_sigma_v: float,
    draws: int,
) -> dict[str, object]:
    """Compute one shared-parameter blocked synthetic recovery record."""
    train = matrix[~test_mask]
    test = matrix[test_mask]
    train_gram = train.T @ train
    test_gram = test.T @ test
    singular = np.linalg.svd(train / np.sqrt(train.shape[0]), compute_uv=False)
    tolerance = max(train.shape) * np.finfo(np.float64).eps * singular[0]
    rank = int(np.sum(singular > tolerance))
    condition = singular[0] / singular[-1] if rank == matrix.shape[1] else float("inf")
    errors = covariance_draw_errors(
        train_gram,
        test_gram,
        train_rows=train.shape[0],
        test_rows=test.shape[0],
        noise_sigma_v=noise_sigma_v,
        draws=draws,
        seed=seed,
    )
    return {
        "condition_number": condition,
        "global_error_median_v": errors[0],
        "global_error_p90_v": errors[1],
        "model": model,
        "parameters": matrix.shape[1],
        "rank": rank,
        "split": split,
        "test_error_median_v": errors[4],
        "test_error_p90_v": errors[5],
        "test_rows": test.shape[0],
        "train_error_median_v": errors[2],
        "train_error_p90_v": errors[3],
        "train_rows": train.shape[0],
    }


def run_blocked() -> int:
    """Run descriptive shared-parameter blocked synthetic recovery checks."""
    configuration = load_json(CONFIGURATION_PATH)
    basis, geometry = run_analysis.load_basis_and_geometry()
    temporal = load_temporal()
    start_utc_ns, stop_utc_ns = temporal_interval()
    period_days = float(configuration["basis"]["synodic_period_days"])
    longitude = circular_mean_longitude(geometry)
    longitude_sector = np.floor((longitude + 180.0) / 30.0).astype(int) % 12
    cutoff = start_utc_ns + int(0.75 * (stop_utc_ns - start_utc_ns))
    seven_day_fold = (
        np.floor_divide(geometry.midpoint_utc_ns, 7 * run_analysis.DAY_NS) % 4
    )
    namespace = str(configuration["injections"]["seed_namespace"])
    noise_sigma = float(configuration["injections"]["recovery_gate_noise_sigma_v"])
    draws = int(configuration["injections"]["replicates"])
    records: list[dict[str, object]] = []
    for model in ("constant", "constant_synodic"):
        if model == "constant":
            source = np.asarray(basis[:, :36])
        else:
            source = np.asarray(temporal[:, temporal_column_indices(5)])
        field_gram = operator.temporal_field_gram(
            5,
            model,
            start_utc_ns=start_utc_ns,
            stop_utc_ns=stop_utc_ns,
            period_days=period_days,
            epoch_ns=0,
        )
        matrix = source @ metric_inverse_square_root(field_gram)
        masks: list[tuple[str, np.ndarray]] = [
            ("mission_tail", geometry.midpoint_utc_ns >= cutoff)
        ]
        masks.extend(
            (f"longitude_{sector:02d}", longitude_sector == sector)
            for sector in range(12)
        )
        masks.extend(
            (f"seven_day_fold_{fold}", seven_day_fold == fold) for fold in range(4)
        )
        for split, mask in masks:
            record = blocked_record(
                matrix,
                mask,
                model=model,
                split=split,
                seed=run_analysis.seed_from_parts(namespace, "blocked", model, split),
                noise_sigma_v=noise_sigma,
                draws=draws,
            )
            records.append(record)
            print(json.dumps(record, indent=2))
    run_analysis.write_csv(RESULTS_DIR / "blocked-recovery.csv", records)

    independent_records: list[dict[str, object]] = []
    windows, _ = run_analysis.iter_windows(geometry.midpoint_utc_ns, 24.0)
    for completed, (window_id, indices) in enumerate(windows, start=1):
        matrix = np.asarray(basis[indices, :36]) * np.sqrt(4.0 * np.pi)
        times = geometry.midpoint_utc_ns[indices]
        cutoff = int(np.min(times) + 0.75 * (np.max(times) - np.min(times)))
        masks: list[tuple[str, np.ndarray]] = [("contiguous_time", times >= cutoff)]
        sectors = longitude_sector[indices]
        masks.extend(
            (f"longitude_{sector:02d}", sectors == sector)
            for sector in np.unique(sectors)
        )
        for split, mask in masks:
            if not np.any(mask) or np.all(mask):
                continue
            independent_records.append(
                blocked_record(
                    matrix,
                    mask,
                    model="independent_24h_l5",
                    split=split,
                    seed=run_analysis.seed_from_parts(
                        namespace,
                        "blocked",
                        "independent_24h_l5",
                        window_id,
                        split,
                    ),
                    noise_sigma_v=noise_sigma,
                    draws=draws,
                )
                | {"window_id": window_id}
            )
        if completed % 100 == 0 or completed == len(windows):
            print(f"independent blocked [{completed}/{len(windows)}]")
    run_analysis.write_csv(
        RESULTS_DIR / "blocked-independent-windows.csv", independent_records
    )
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "prepare-temporal",
            "temporal",
            "sensitivities",
            "injections",
            "blocked",
            "caps",
            "column-normalization",
        ),
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    handlers = {
        "blocked": run_blocked,
        "caps": run_caps,
        "column-normalization": run_column_normalization,
        "injections": run_injections,
        "prepare-temporal": prepare_temporal,
        "sensitivities": run_sensitivities,
        "temporal": run_temporal,
    }
    return handlers[args.phase]()


if __name__ == "__main__":
    raise SystemExit(main())
