"""Observation-operator helpers for Experiment 008.

The module deliberately accepts geometry arrays, never fitted potentials.  Its
canonical row is one energy sweep: valid row-level spherical harmonics are
averaged before any inverse-problem diagnostic is computed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.special import sph_harm_y, sph_harm_y_all

if TYPE_CHECKING:
    from pathlib import Path

NAT_INT64 = np.iinfo(np.int64).min
Reducer = Literal["mean", "midpoint"]


@dataclass(frozen=True)
class SweepGeometry:
    """Geometry for structurally valid, surface-connected sweeps."""

    source_index: np.ndarray
    spec_no: np.ndarray
    row_utc_ns: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    polarity: np.ndarray
    valid_rows: np.ndarray
    midpoint_utc_ns: np.ndarray

    @property
    def size(self) -> int:
        """Return the number of retained sweeps."""
        return int(self.source_index.size)


@dataclass(frozen=True)
class SpectrumDiagnostics:
    """Direct-SVD diagnostics for one normalized observation operator."""

    rows: int
    parameters: int
    singular_values: np.ndarray
    exact_rank: int
    effective_ranks: dict[float, int]
    condition_number: float
    column_norms: np.ndarray
    leverage: np.ndarray
    resolution_diagonal: dict[float, np.ndarray]
    resolution_trace: dict[float, float]
    smallest_physical_mode: np.ndarray | None


def harmonic_indices(lmax: int) -> tuple[np.ndarray, np.ndarray]:
    """Return degree and order arrays in degree-major, order-minor order."""
    if lmax < 0:
        raise ValueError("lmax must be nonnegative")
    degree: list[int] = []
    order: list[int] = []
    for ell in range(lmax + 1):
        for emm in range(-ell, ell + 1):
            degree.append(ell)
            order.append(emm)
    return np.asarray(degree, dtype=np.int16), np.asarray(order, dtype=np.int16)


def real_to_complex_coefficients(coefficients: np.ndarray, lmax: int) -> np.ndarray:
    """Convert real-basis coefficients to repository complex coefficients."""
    coefficients = np.asarray(coefficients, dtype=np.float64)
    expected = (lmax + 1) ** 2
    if coefficients.shape[-1] != expected:
        raise ValueError(f"Expected {expected} coefficients")
    result = np.zeros(coefficients.shape, dtype=np.complex128)
    for ell in range(lmax + 1):
        zero = ell * ell + ell
        result[..., zero] = coefficients[..., zero]
        for emm in range(1, ell + 1):
            negative = zero - emm
            positive = zero + emm
            phase = -1.0 if emm % 2 else 1.0
            result[..., positive] = (
                phase
                * (coefficients[..., positive] - 1j * coefficients[..., negative])
                / np.sqrt(2.0)
            )
            result[..., negative] = (
                coefficients[..., positive] + 1j * coefficients[..., negative]
            ) / np.sqrt(2.0)
    return result


def real_spherical_harmonics(
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
    lmax: int,
) -> np.ndarray:
    """Evaluate an orthonormal real spherical-harmonic basis.

    The convention is ``sqrt(2) (-1)^m Im(Y_l_|m|)`` for negative orders,
    ``Y_l_0`` for zero, and ``sqrt(2) (-1)^m Re(Y_l_m)`` for positive orders.
    """
    latitude = np.asarray(latitude_deg, dtype=np.float64)
    longitude = np.asarray(longitude_deg, dtype=np.float64)
    latitude, longitude = np.broadcast_arrays(latitude, longitude)
    flat_latitude = latitude.ravel()
    flat_longitude = longitude.ravel()
    colatitude = np.deg2rad(90.0 - flat_latitude)
    azimuth = np.deg2rad(flat_longitude)
    degree, order = harmonic_indices(lmax)
    result = np.empty((flat_latitude.size, degree.size), dtype=np.float64)
    root_two = np.sqrt(2.0)
    for column, (ell, emm) in enumerate(zip(degree, order, strict=True)):
        if emm == 0:
            result[:, column] = np.real(sph_harm_y(ell, 0, colatitude, azimuth))
        else:
            harmonic = sph_harm_y(ell, abs(int(emm)), colatitude, azimuth)
            phase = -1.0 if abs(int(emm)) % 2 else 1.0
            if emm < 0:
                result[:, column] = root_two * phase * np.imag(harmonic)
            else:
                result[:, column] = root_two * phase * np.real(harmonic)
    return result.reshape((*latitude.shape, degree.size))


def _real_basis_from_all(complex_basis: np.ndarray, lmax: int) -> np.ndarray:
    """Convert ``sph_harm_y_all`` output to the experiment's real basis."""
    sample_shape = complex_basis.shape[2:]
    result = np.empty((*sample_shape, (lmax + 1) ** 2), dtype=np.float64)
    root_two = np.sqrt(2.0)
    column = 0
    for ell in range(lmax + 1):
        for emm in range(-ell, ell + 1):
            if emm == 0:
                values = np.real(complex_basis[ell, 0])
            else:
                # scipy orders sph_harm_y_all as 0, 1, ..., m, -m, ..., -1.
                harmonic = complex_basis[ell, abs(emm)]
                phase = -1.0 if abs(emm) % 2 else 1.0
                values = (
                    root_two
                    * phase
                    * (np.imag(harmonic) if emm < 0 else np.real(harmonic))
                )
            result[..., column] = values
            column += 1
    return result


def load_sweep_geometry(
    path: Path,
    variant: str = "corrected",
    *,
    complete_only: bool = False,
) -> SweepGeometry:
    """Load one geometry variant and retain its connected sweep population."""
    with np.load(path, allow_pickle=False) as data:
        row_utc_ns = np.asarray(data["row_utc_ns"], dtype=np.int64)
        latitude = np.asarray(data[f"{variant}_latitude"], dtype=np.float64)
        longitude = np.asarray(data[f"{variant}_longitude"], dtype=np.float64)
        polarity = np.asarray(data[f"{variant}_polarity"], dtype=np.int8)
        structural = np.asarray(data["structural_valid"], dtype=bool)
        source_index = np.asarray(data["source_file_index"], dtype=np.int64)
        spec_no = np.asarray(data["sweep_spec_no"], dtype=np.int64)

    valid_rows = (
        np.isfinite(latitude)
        & np.isfinite(longitude)
        & (row_utc_ns != NAT_INT64)
        & structural[:, None]
    )
    valid_count = np.sum(valid_rows, axis=1)
    keep = valid_count == row_utc_ns.shape[1] if complete_only else valid_count >= 1
    row_utc_ns = row_utc_ns[keep]
    valid_rows = valid_rows[keep]
    # Structural validity guarantees finite UTC for every row. Window
    # assignment uses the complete sweep, not only the connected subset.
    minimum = np.min(row_utc_ns, axis=1)
    maximum = np.max(row_utc_ns, axis=1)
    midpoint = minimum + (maximum - minimum) // 2
    return SweepGeometry(
        source_index=source_index[keep],
        spec_no=spec_no[keep],
        row_utc_ns=row_utc_ns,
        latitude=latitude[keep],
        longitude=longitude[keep],
        polarity=polarity[keep],
        valid_rows=valid_rows,
        midpoint_utc_ns=midpoint,
    )


def sweep_harmonics(
    geometry: SweepGeometry,
    lmax: int,
    *,
    reducer: Reducer = "mean",
    chunk_sweeps: int = 1024,
    longitude_offset_deg: np.ndarray | None = None,
) -> np.ndarray:
    """Build one real-harmonic row per sweep in bounded-memory chunks."""
    if chunk_sweeps < 1:
        raise ValueError("chunk_sweeps must be positive")
    parameters = (lmax + 1) ** 2
    result = np.empty((geometry.size, parameters), dtype=np.float64)
    if longitude_offset_deg is not None:
        longitude_offset_deg = np.broadcast_to(
            longitude_offset_deg, geometry.latitude.shape
        )

    for start in range(0, geometry.size, chunk_sweeps):
        stop = min(start + chunk_sweeps, geometry.size)
        valid = geometry.valid_rows[start:stop]
        latitude = geometry.latitude[start:stop]
        longitude = geometry.longitude[start:stop]
        if longitude_offset_deg is not None:
            longitude = longitude + longitude_offset_deg[start:stop]

        if reducer == "midpoint":
            utc_distance = np.abs(
                geometry.row_utc_ns[start:stop]
                - geometry.midpoint_utc_ns[start:stop, None]
            )
            utc_distance = np.where(valid, utc_distance, np.iinfo(np.int64).max)
            row_index = np.argmin(utc_distance, axis=1)
            selected = real_spherical_harmonics(
                latitude[np.arange(stop - start), row_index],
                longitude[np.arange(stop - start), row_index],
                lmax,
            )
            result[start:stop] = selected
            continue
        if reducer != "mean":
            raise ValueError(f"Unknown reducer: {reducer}")

        flat_valid = valid.ravel()
        colatitude = np.deg2rad(90.0 - latitude.ravel()[flat_valid])
        azimuth = np.deg2rad(longitude.ravel()[flat_valid])
        complex_basis = sph_harm_y_all(lmax, lmax, colatitude, azimuth)
        row_basis = _real_basis_from_all(complex_basis, lmax)
        local_sweep = np.repeat(
            np.arange(stop - start, dtype=np.int64), valid.shape[1]
        )[flat_valid]
        accumulated = np.zeros((stop - start, parameters), dtype=np.float64)
        np.add.at(accumulated, local_sweep, row_basis)
        accumulated /= np.sum(valid, axis=1, keepdims=True)
        result[start:stop] = accumulated
    return result


def temporal_factors(
    utc_ns: np.ndarray,
    model: str,
    *,
    period_days: float,
    epoch_ns: int | None = None,
) -> np.ndarray:
    """Evaluate frozen constant or constant-plus-synodic temporal factors."""
    utc_ns = np.asarray(utc_ns, dtype=np.int64)
    if epoch_ns is None:
        epoch_ns = int(np.min(utc_ns))
    days = (utc_ns - epoch_ns) / (86_400.0 * 1e9)
    if model == "constant":
        return np.ones((utc_ns.size, 1), dtype=np.float64)
    if model == "constant_synodic":
        phase = 2.0 * np.pi * days / period_days
        return np.column_stack((np.ones(utc_ns.size), np.cos(phase), np.sin(phase)))
    raise ValueError(f"Unknown temporal model: {model}")


def tensor_operator(spatial: np.ndarray, temporal: np.ndarray) -> np.ndarray:
    """Form row-wise temporal/spatial tensor products."""
    if spatial.shape[0] != temporal.shape[0]:
        raise ValueError("spatial and temporal rows must match")
    return np.einsum("nq,nk->nqk", temporal, spatial).reshape(spatial.shape[0], -1)


def sweep_tensor_harmonics(
    geometry: SweepGeometry,
    lmax: int,
    model: str,
    *,
    period_days: float,
    epoch_ns: int | None = None,
    chunk_sweeps: int = 512,
    longitude_offset_deg: np.ndarray | None = None,
) -> np.ndarray:
    """Average exact row-level ``T(t) Y(x)`` products within each sweep."""
    if epoch_ns is None:
        epoch_ns = int(np.min(geometry.row_utc_ns))
    temporal_columns = 1 if model == "constant" else 3
    parameters = temporal_columns * (lmax + 1) ** 2
    result = np.empty((geometry.size, parameters), dtype=np.float64)
    if longitude_offset_deg is not None:
        longitude_offset_deg = np.broadcast_to(
            longitude_offset_deg, geometry.latitude.shape
        )

    for start in range(0, geometry.size, chunk_sweeps):
        stop = min(start + chunk_sweeps, geometry.size)
        valid = geometry.valid_rows[start:stop]
        latitude = geometry.latitude[start:stop].ravel()[valid.ravel()]
        longitude = geometry.longitude[start:stop]
        if longitude_offset_deg is not None:
            longitude = longitude + longitude_offset_deg[start:stop]
        longitude = longitude.ravel()[valid.ravel()]
        utc_ns = geometry.row_utc_ns[start:stop].ravel()[valid.ravel()]
        spatial = real_spherical_harmonics(latitude, longitude, lmax)
        temporal = temporal_factors(
            utc_ns,
            model,
            period_days=period_days,
            epoch_ns=epoch_ns,
        )
        row_operator = tensor_operator(spatial, temporal)
        local_sweep = np.repeat(
            np.arange(stop - start, dtype=np.int64), valid.shape[1]
        )[valid.ravel()]
        accumulated = np.zeros((stop - start, parameters), dtype=np.float64)
        np.add.at(accumulated, local_sweep, row_operator)
        accumulated /= np.sum(valid, axis=1, keepdims=True)
        result[start:stop] = accumulated
    return result


def normalize_columns(operator: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize nonzero operator columns to unit L2 norm."""
    norms = np.linalg.norm(operator, axis=0)
    normalized = np.zeros_like(operator, dtype=np.float64)
    nonzero = norms > 0.0
    normalized[:, nonzero] = operator[:, nonzero] / norms[nonzero]
    return normalized, norms


def direct_spectrum(
    operator: np.ndarray,
    *,
    relative_thresholds: tuple[float, ...] = (1e-3, 1e-6, 1e-9),
    include_explicit_null: bool = True,
) -> SpectrumDiagnostics:
    """Compute direct-SVD rank, conditioning, leverage, and resolution."""
    rows, parameters = operator.shape
    normalized, norms = normalize_columns(operator)
    full_matrices = include_explicit_null and rows < parameters
    u_matrix, singular_values, vh_matrix = np.linalg.svd(
        normalized, full_matrices=full_matrices
    )
    largest = float(singular_values[0]) if singular_values.size else 0.0
    tolerance = max(rows, parameters) * np.finfo(np.float64).eps * largest
    exact_rank = int(np.sum(singular_values > tolerance))
    effective = {
        threshold: int(np.sum(singular_values > threshold * largest))
        for threshold in relative_thresholds
    }
    if rows < parameters or exact_rank < parameters or not singular_values.size:
        condition = float("inf")
    else:
        condition = largest / float(singular_values[-1])
    leverage = np.sum(u_matrix[:, :exact_rank] ** 2, axis=1)
    resolution_diagonal: dict[float, np.ndarray] = {}
    resolution_trace: dict[float, float] = {}
    for threshold in relative_thresholds:
        keep = singular_values > threshold * largest
        diagonal = np.sum(vh_matrix[: singular_values.size][keep] ** 2, axis=0)
        resolution_diagonal[threshold] = diagonal
        resolution_trace[threshold] = float(np.sum(diagonal))

    smallest_mode: np.ndarray | None
    if vh_matrix.shape[0] == parameters:
        normalized_mode = vh_matrix[-1]
        smallest_mode = np.divide(
            normalized_mode,
            norms,
            out=np.zeros_like(normalized_mode),
            where=norms > 0.0,
        )
    else:
        smallest_mode = None
    return SpectrumDiagnostics(
        rows=rows,
        parameters=parameters,
        singular_values=singular_values,
        exact_rank=exact_rank,
        effective_ranks=effective,
        condition_number=condition,
        column_norms=norms,
        leverage=leverage,
        resolution_diagonal=resolution_diagonal,
        resolution_trace=resolution_trace,
        smallest_physical_mode=smallest_mode,
    )


def metric_spectrum(
    operator: np.ndarray,
    field_gram: np.ndarray,
    *,
    relative_thresholds: tuple[float, ...] = (1e-3, 1e-6, 1e-9),
    include_explicit_null: bool = True,
) -> SpectrumDiagnostics:
    """Compute SVD diagnostics in a physical global-RMS field metric.

    If ``c.T @ field_gram @ c`` is global mean-square field amplitude, the
    singular values returned here are sampled-RMS/global-RMS visibility ratios.
    """
    rows, parameters = operator.shape
    if field_gram.shape != (parameters, parameters):
        raise ValueError("field_gram shape must match operator parameters")
    eigenvalues, eigenvectors = np.linalg.eigh(field_gram)
    tolerance = parameters * np.finfo(np.float64).eps * np.max(eigenvalues)
    if np.any(eigenvalues <= tolerance):
        raise ValueError("field_gram must be positive definite")
    square_root = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    inverse_square_root = (
        eigenvectors * np.reciprocal(np.sqrt(eigenvalues))
    ) @ eigenvectors.T
    transformed = operator @ inverse_square_root / np.sqrt(rows)
    full_matrices = include_explicit_null and rows < parameters
    u_matrix, compact_singular, vh_matrix = np.linalg.svd(
        transformed, full_matrices=full_matrices
    )
    largest = float(compact_singular[0]) if compact_singular.size else 0.0
    rank_tolerance = max(rows, parameters) * np.finfo(np.float64).eps * largest
    exact_rank = int(np.sum(compact_singular > rank_tolerance))
    singular_values = np.zeros(parameters, dtype=np.float64)
    singular_values[: compact_singular.size] = compact_singular
    effective = {
        threshold: int(np.sum(singular_values > threshold * largest))
        for threshold in relative_thresholds
    }
    condition = (
        largest / float(compact_singular[-1])
        if exact_rank == parameters and compact_singular.size == parameters
        else float("inf")
    )
    leverage = np.sum(u_matrix[:, :exact_rank] ** 2, axis=1)
    resolution_diagonal: dict[float, np.ndarray] = {}
    resolution_trace: dict[float, float] = {}
    compact_vh = vh_matrix[: compact_singular.size]
    for threshold in relative_thresholds:
        keep = compact_singular > threshold * largest
        metric_projector = compact_vh[keep].T @ compact_vh[keep]
        physical_resolution = inverse_square_root @ metric_projector @ square_root
        resolution_diagonal[threshold] = np.diag(physical_resolution)
        resolution_trace[threshold] = float(np.trace(physical_resolution))

    physical_mode = (
        inverse_square_root @ vh_matrix[-1]
        if vh_matrix.shape[0] == parameters
        else None
    )
    return SpectrumDiagnostics(
        rows=rows,
        parameters=parameters,
        singular_values=singular_values,
        exact_rank=exact_rank,
        effective_ranks=effective,
        condition_number=condition,
        column_norms=np.linalg.norm(operator, axis=0),
        leverage=leverage,
        resolution_diagonal=resolution_diagonal,
        resolution_trace=resolution_trace,
        smallest_physical_mode=physical_mode,
    )


def static_field_gram(parameters: int) -> np.ndarray:
    """Return the analytic surface-area mean-square metric."""
    return np.eye(parameters, dtype=np.float64) / (4.0 * np.pi)


def temporal_field_gram(
    lmax: int,
    model: str,
    *,
    start_utc_ns: int,
    stop_utc_ns: int,
    period_days: float,
    epoch_ns: int = 0,
) -> np.ndarray:
    """Return a uniform continuous space-time mean-square field metric."""
    if stop_utc_ns <= start_utc_ns:
        raise ValueError("stop_utc_ns must be later than start_utc_ns")
    if model == "constant":
        temporal_gram = np.ones((1, 1), dtype=np.float64)
    elif model == "constant_synodic":
        day_ns = 86_400.0 * 1e9
        angular_rate = 2.0 * np.pi / period_days
        start = (start_utc_ns - epoch_ns) / day_ns
        stop = (stop_utc_ns - epoch_ns) / day_ns
        span = stop - start
        phase_start = angular_rate * start
        phase_stop = angular_rate * stop
        mean_cos = (np.sin(phase_stop) - np.sin(phase_start)) / (angular_rate * span)
        mean_sin = (-np.cos(phase_stop) + np.cos(phase_start)) / (angular_rate * span)
        double_term = (np.sin(2.0 * phase_stop) - np.sin(2.0 * phase_start)) / (
            4.0 * angular_rate * span
        )
        cross_term = (-np.cos(2.0 * phase_stop) + np.cos(2.0 * phase_start)) / (
            4.0 * angular_rate * span
        )
        temporal_gram = np.array(
            [
                [1.0, mean_cos, mean_sin],
                [mean_cos, 0.5 + double_term, cross_term],
                [mean_sin, cross_term, 0.5 - double_term],
            ]
        )
    else:
        raise ValueError(f"Unknown temporal model: {model}")
    spatial_gram = static_field_gram((lmax + 1) ** 2)
    return np.kron(temporal_gram, spatial_gram)


def global_basis(lmax: int, step_degrees: float = 2.0) -> np.ndarray:
    """Evaluate the frozen equal-angle global diagnostic grid."""
    latitude = np.arange(-90.0, 90.0 + 0.5 * step_degrees, step_degrees)
    longitude = np.arange(-180.0, 180.0, step_degrees)
    lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    return real_spherical_harmonics(lat_grid.ravel(), lon_grid.ravel(), lmax)


def rms_from_gram(coefficients: np.ndarray, gram: np.ndarray) -> np.ndarray:
    """Compute RMS fields from coefficient rows and a basis Gram matrix."""
    coefficients = np.atleast_2d(np.asarray(coefficients, dtype=np.float64))
    variance = np.einsum("ni,ij,nj->n", coefficients, gram, coefficients)
    return np.sqrt(np.maximum(variance, 0.0))


def mode_visibility(
    operator: np.ndarray,
    physical_mode: np.ndarray | None,
    global_gram: np.ndarray,
) -> tuple[float, float]:
    """Scale a mode to 100 V global RMS and report sampled RMS and ratio."""
    if physical_mode is None:
        return 0.0, 0.0
    global_rms = float(rms_from_gram(physical_mode, global_gram)[0])
    if not np.isfinite(global_rms) or global_rms == 0.0:
        return 0.0, 0.0
    scaled = physical_mode * (100.0 / global_rms)
    sampled_rms = float(np.sqrt(np.mean((operator @ scaled) ** 2)))
    return sampled_rms, sampled_rms / 100.0
