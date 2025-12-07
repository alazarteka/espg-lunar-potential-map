"""
Compute time-dependent spherical harmonic coefficients a_lm(t) for lunar surface potential.

Implements the expansion:
    U(φ, θ, t) = Σ_{l,m} a_lm(t) Y_lm(φ, θ)

where a_lm(t) are fitted in temporal windows with spatial coverage validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from numpy.linalg import LinAlgError, lstsq

from scipy.special import sph_harm_y


def _sph_harm(m: int, l: int, phi, theta):
    """Evaluate spherical harmonics using SciPy's sph_harm_y (θ=colat, φ=azimuth)."""
    return sph_harm_y(l, m, theta, phi)


# Default cache directory
DEFAULT_CACHE_DIR = Path("artifacts/potential_cache")
DEFAULT_SYNODIC_PERIOD_DAYS = 29.530588


@dataclass(slots=True)
class TimeWindow:
    """Single time window with measurements."""

    start_time: np.datetime64
    end_time: np.datetime64
    midpoint: np.datetime64
    utc: np.ndarray  # datetime64[ns]
    lat: np.ndarray  # degrees
    lon: np.ndarray  # degrees
    potential: np.ndarray  # V
    chi2: np.ndarray | None  # fit quality if available


@dataclass(slots=True)
class HarmonicCoefficients:
    """Spherical harmonic coefficients for a single time window."""

    time: np.datetime64  # midpoint of window
    lmax: int
    coeffs: np.ndarray  # complex, shape ((lmax+1)^2,)
    n_samples: int
    spatial_coverage: float  # fraction of bins with data
    rms_residual: float  # RMS fit residual





def _discover_npz(cache_dir: Path) -> list[Path]:
    """Find all NPZ cache files."""
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory {cache_dir} does not exist")
    return sorted(p for p in cache_dir.rglob("*.npz") if p.is_file())


def _load_all_data(
    files: list[Path],
    start_ts: np.datetime64,
    end_ts_exclusive: np.datetime64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load all measurements in the time range."""
    utc_parts: list[np.ndarray] = []
    lat_parts: list[np.ndarray] = []
    lon_parts: list[np.ndarray] = []
    pot_parts: list[np.ndarray] = []

    start_str = str(start_ts.astype("datetime64[s]"))
    end_str = str(end_ts_exclusive.astype("datetime64[s]"))

    for path in files:
        with np.load(path) as data:
            utc = data["rows_utc"]
            if utc.size == 0:
                continue

            valid_time = utc != ""
            if not np.any(valid_time):
                continue

            mask = valid_time & (utc >= start_str) & (utc < end_str)
            if not np.any(mask):
                continue

            try:
                utc_vals = np.array(utc[mask], dtype="datetime64[ns]")
            except ValueError:
                logging.debug("Failed to parse UTC in %s", path)
                continue

            lat = data["rows_projection_latitude"].astype(np.float64)
            lon = data["rows_projection_longitude"].astype(np.float64)
            pot = data["rows_projected_potential"].astype(np.float64)

            finite_mask = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(pot)
            mask_refined = mask & finite_mask
            if not np.any(mask_refined):
                continue

            utc_parts.append(utc_vals[finite_mask[mask]])
            lat_parts.append(lat[mask_refined])
            lon_parts.append(lon[mask_refined])
            pot_parts.append(pot[mask_refined])

    if not utc_parts:
        return (
            np.array([], dtype="datetime64[ns]"),
            np.array([]),
            np.array([]),
            np.array([]),
        )

    return (
        np.concatenate(utc_parts),
        np.concatenate(lat_parts),
        np.concatenate(lon_parts),
        np.concatenate(pot_parts),
    )


def _partition_into_windows(
    utc: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    potential: np.ndarray,
    window_hours: float,
    stride_hours: float | None = None,
    start_anchor: np.datetime64 | None = None,
) -> Iterator[TimeWindow]:
    """
    Partition data into temporal windows of specified duration.

    Args:
        utc: Array of UTC timestamps
        lat: Array of latitudes
        lon: Array of longitudes
        potential: Array of surface potentials
        window_hours: Duration of each window
        stride_hours: Step size between window starts (enables overlap).
                      If None, defaults to window_hours (non-overlapping).
        start_anchor: Optional reference timestamp (UTC). Window starts snap to
                      this anchor instead of the first measurement time.

    Yields TimeWindow objects with spatially-contiguous measurements.
    """
    if utc.size == 0:
        return

    if stride_hours is None:
        stride_hours = window_hours

    # Sort by time
    sort_idx = np.argsort(utc)
    utc_sorted = utc[sort_idx]
    lat_sorted = lat[sort_idx]
    lon_sorted = lon[sort_idx]
    pot_sorted = potential[sort_idx]

    window_delta = np.timedelta64(int(window_hours * 3600), "s")
    stride_delta = np.timedelta64(int(stride_hours * 3600), "s")

    if start_anchor is None:
        current_start = utc_sorted[0]
    else:
        anchor = np.datetime64(start_anchor)
        current_start = anchor.astype(utc_sorted.dtype)

    while current_start <= utc_sorted[-1]:
        current_end = current_start + window_delta

        mask = (utc_sorted >= current_start) & (utc_sorted < current_end)
        n_samples = np.sum(mask)

        if n_samples > 0:
            window_utc = utc_sorted[mask]
            midpoint = window_utc[0] + (window_utc[-1] - window_utc[0]) // 2

            yield TimeWindow(
                start_time=current_start,
                end_time=current_end,
                midpoint=midpoint,
                utc=window_utc,
                lat=lat_sorted[mask],
                lon=lon_sorted[mask],
                potential=pot_sorted[mask],
                chi2=None,
            )

        # Slide window by stride
        current_start = current_start + stride_delta


def _compute_spatial_coverage(
    lat: np.ndarray, lon: np.ndarray, bin_size_deg: float = 10.0
) -> float:
    """
    Compute fraction of spatial bins that contain at least one measurement.

    Returns value in [0, 1] indicating global coverage quality.
    """
    lat_bins = int(180 / bin_size_deg)
    lon_bins = int(360 / bin_size_deg)

    lat_indices = ((lat + 90) / bin_size_deg).astype(int)
    lon_indices = ((lon + 180) / bin_size_deg).astype(int)

    lat_indices = np.clip(lat_indices, 0, lat_bins - 1)
    lon_indices = np.clip(lon_indices, 0, lon_bins - 1)

    occupied = np.zeros((lat_bins, lon_bins), dtype=bool)
    occupied[lat_indices, lon_indices] = True

    return float(np.sum(occupied)) / occupied.size


def _harmonic_coefficient_count(lmax: int) -> int:
    """Number of spherical harmonic coefficients up to degree lmax."""
    return (lmax + 1) ** 2


def _z_rotation_diagonal(lmax: int, angle_rad: float) -> np.ndarray:
    """Diagonal elements for rotating coefficients about the z-axis by angle."""
    diag = np.empty(_harmonic_coefficient_count(lmax), dtype=np.complex128)
    idx = 0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            diag[idx] = np.exp(1j * m * angle_rad)
            idx += 1
    return diag


def _build_degree_weight_vector(lmax: int, exponent: float | None) -> np.ndarray:
    """Per-(l,m) weights for degree-dependent spatial regularization."""
    n_coeffs = _harmonic_coefficient_count(lmax)
    if exponent is None:
        return np.ones(n_coeffs, dtype=np.float64)
    if exponent < 0.0:
        raise ValueError("spatial weight exponent must be non-negative")
    if np.isclose(exponent, 0.0):
        return np.ones(n_coeffs, dtype=np.float64)

    weights = np.empty(n_coeffs, dtype=np.float64)
    idx = 0
    for l in range(lmax + 1):
        if l == 0:
            weight = 0.0
        else:
            weight = float((l * (l + 1)) ** exponent)
        for _ in range(-l, l + 1):
            weights[idx] = weight
            idx += 1
    return weights


def _build_harmonic_design(
    lat_deg: np.ndarray, lon_deg: np.ndarray, lmax: int
) -> np.ndarray:
    """Build design matrix of spherical harmonics."""
    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)
    colatitudes = (np.pi / 2.0) - lat_rad

    n_rows = lat_rad.size
    n_cols = _harmonic_coefficient_count(lmax)
    design = np.empty((n_rows, n_cols), dtype=np.complex128)

    col_idx = 0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            design[:, col_idx] = _sph_harm(m, l, lon_rad, colatitudes)
            col_idx += 1

    return design


def _enforce_reality_condition(coeffs: np.ndarray, lmax: int) -> np.ndarray:
    """
    Enforce reality condition: a_{l,-m} = (-1)^m * conj(a_{l,m}).

    For a real-valued field reconstructed from complex spherical harmonics,
    the coefficients must satisfy this relationship. This function projects
    the solution onto the real subspace.
    """
    coeffs_real = coeffs.copy()

    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            if m == 0:
                idx_zero = l * l + l
                coeffs_real[idx_zero] = complex(coeffs_real[idx_zero].real, 0.0)
            elif m > 0:
                # Find positive m coefficient
                idx_pos = l * l + l + m
                # Find corresponding negative m coefficient
                idx_neg = l * l + l - m

                # Average to enforce symmetry
                a_pos = coeffs[idx_pos]
                a_neg = coeffs[idx_neg]

                # Enforce: a_{l,-m} = (-1)^m * conj(a_{l,m})
                phase = (-1) ** m
                a_avg_pos = 0.5 * (a_pos + phase * np.conj(a_neg))
                a_avg_neg = phase * np.conj(a_avg_pos)

                coeffs_real[idx_pos] = a_avg_pos
                coeffs_real[idx_neg] = a_avg_neg

    return coeffs_real


def _build_temporal_derivative_matrix(
    delta_hours: np.ndarray,
    lmax: int,
    rotation_angles: np.ndarray | None = None,
) -> "scipy.sparse.csr_matrix":
    """
    Build finite-difference operator for temporal derivatives.

    Returns sparse matrix D of shape:
        ((n_windows-1)*n_coeffs, n_windows*n_coeffs)

    Encoding: D @ a_stacked ≈ [(a_2 - a_1)/Δt_1, ..., (a_N - a_{N-1})/Δt_{N-1}]
    """
    from scipy.sparse import bmat, diags, eye

    if delta_hours.ndim != 1:
        raise ValueError("delta_hours must be a 1D array of time-step lengths in hours")
    if delta_hours.size == 0:
        raise ValueError(
            "delta_hours must contain at least one entry for temporal coupling"
        )
    if rotation_angles is not None and rotation_angles.shape != delta_hours.shape:
        raise ValueError("rotation_angles must match delta_hours shape")

    safe_delta = np.clip(delta_hours, np.finfo(np.float64).eps, None)
    inv_delta = 1.0 / safe_delta
    n_windows = safe_delta.size + 1
    n_coeffs = _harmonic_coefficient_count(lmax)
    identity = eye(n_coeffs, dtype=np.complex128, format="csr")

    row_blocks = []
    for i in range(n_windows - 1):
        row = [None] * n_windows
        row[i] = -inv_delta[i] * identity
        if rotation_angles is None:
            rot = identity
        else:
            diag = _z_rotation_diagonal(lmax, rotation_angles[i])
            rot = diags(diag, offsets=0, format="csr")
        row[i + 1] = inv_delta[i] * rot
        row_blocks.append(row)

    return bmat(row_blocks, format="csr")


def _fit_window_harmonics(
    window: TimeWindow,
    lmax: int,
    l2_penalty: float = 0.0,
    min_samples: int = 100,
    min_coverage: float = 0.1,
    degree_weights: np.ndarray | None = None,
) -> HarmonicCoefficients | None:
    """
    Fit spherical harmonics to a single time window.

    Returns None if the window lacks sufficient spatial coverage.
    """
    if window.lat.size < min_samples:
        logging.debug(
            "Window at %s: insufficient samples (%d < %d)",
            window.midpoint,
            window.lat.size,
            min_samples,
        )
        return None

    coverage = _compute_spatial_coverage(window.lat, window.lon)
    if coverage < min_coverage:
        logging.debug(
            "Window at %s: insufficient coverage (%.2f%% < %.2f%%)",
            window.midpoint,
            coverage * 100,
            min_coverage * 100,
        )
        return None

    n_coeffs = _harmonic_coefficient_count(lmax)
    if window.lat.size < n_coeffs:
        logging.debug(
            "Window at %s: too few samples (%d) for lmax=%d (%d coeffs)",
            window.midpoint,
            window.lat.size,
            lmax,
            n_coeffs,
        )
        return None

    # Build design matrix
    design = _build_harmonic_design(window.lat, window.lon, lmax)
    potential_complex = window.potential.astype(np.complex128)
    n_cols = design.shape[1]

    # Solve with optional regularization
    try:
        if l2_penalty > 0.0:
            lam = np.sqrt(l2_penalty)
            if degree_weights is None:
                weight_vec = np.ones(n_cols, dtype=np.float64)
            else:
                weight_vec = np.asarray(degree_weights, dtype=np.float64)
                if weight_vec.size != n_cols:
                    raise ValueError(
                        "degree_weights must have length equal to coefficient count"
                    )
            sqrt_weights = np.sqrt(weight_vec, out=np.empty_like(weight_vec))
            identity = np.eye(n_cols, dtype=design.dtype)
            weighted_identity = sqrt_weights[:, None] * identity
            zeros = np.zeros(n_cols, dtype=potential_complex.dtype)
            design_aug = np.vstack([design, lam * weighted_identity])
            rhs_aug = np.concatenate([potential_complex, zeros])
            coeffs, *_ = lstsq(design_aug, rhs_aug, rcond=None)
        else:
            coeffs, *_ = lstsq(design, potential_complex, rcond=None)
    except LinAlgError as exc:
        logging.warning(
            "Harmonic fit failed for window at %s: %s", window.midpoint, exc
        )
        return None

    # Enforce reality condition for physical solution
    coeffs = _enforce_reality_condition(coeffs, lmax)

    # Compute RMS residual
    predicted = np.real(design @ coeffs)
    residuals = window.potential - predicted
    rms_residual = float(np.sqrt(np.mean(residuals**2)))

    return HarmonicCoefficients(
        time=window.midpoint,
        lmax=lmax,
        coeffs=coeffs,
        n_samples=window.lat.size,
        spatial_coverage=coverage,
        rms_residual=rms_residual,
    )


def _fit_coupled_windows(
    windows: list[TimeWindow],
    lmax: int,
    spatial_lambda: float,
    temporal_lambda: float,
    min_samples: int,
    min_coverage: float,
    *,
    degree_weights: np.ndarray | None = None,
    co_rotate: bool = False,
    rotation_period_hours: float | None = None,
) -> list[HarmonicCoefficients]:
    """
    Fit all windows jointly with temporal continuity constraint.

    Solves augmented least-squares system:
        [X_block      ]       [Φ    ]
        [√λ_s * I     ] @ a = [0    ]
        [√λ_t * D_t   ]       [0    ]
    """
    from scipy.sparse import vstack, csr_matrix, diags, block_diag
    from scipy.sparse.linalg import lsqr

    # Step 1: Filter windows by coverage and sample count
    valid_windows = []
    for w in windows:
        if w.lat.size < min_samples:
            logging.debug(
                "Skipping window at %s: insufficient samples (%d < %d)",
                w.midpoint,
                w.lat.size,
                min_samples,
            )
            continue
        coverage = _compute_spatial_coverage(w.lat, w.lon)
        if coverage < min_coverage:
            logging.debug(
                "Skipping window at %s: insufficient coverage (%.2f%% < %.2f%%)",
                w.midpoint,
                coverage * 100,
                min_coverage * 100,
            )
            continue
        valid_windows.append(w)

    n_windows = len(valid_windows)
    n_coeffs = _harmonic_coefficient_count(lmax)
    if degree_weights is None:
        degree_weights = np.ones(n_coeffs, dtype=np.float64)
    else:
        degree_weights = np.asarray(degree_weights, dtype=np.float64)
        if degree_weights.size != n_coeffs:
            raise ValueError("degree_weights length mismatch for coupled solve")

    if n_windows < 2:
        # Not enough windows for temporal coupling - fall back to independent
        logging.warning(
            "Only %d valid windows; temporal coupling requires at least 2. "
            "Falling back to independent fitting.",
            n_windows,
        )
        fallback_results: list[HarmonicCoefficients] = []
        for window in valid_windows:
            result = _fit_window_harmonics(
                window,
                lmax,
                spatial_lambda,
                min_samples,
                min_coverage,
                degree_weights=degree_weights,
            )
            if result is not None:
                fallback_results.append(result)
        return fallback_results

    logging.info(
        "Fitting %d windows jointly with temporal_lambda=%.2e",
        n_windows,
        temporal_lambda,
    )

    # Compute time deltas between consecutive window midpoints (hours)
    delta_hours = np.empty(n_windows - 1, dtype=np.float64)
    for i in range(n_windows - 1):
        delta = valid_windows[i + 1].midpoint - valid_windows[i].midpoint
        dt_hours = float(delta / np.timedelta64(1, "s")) / 3600.0
        if dt_hours <= 0.0:
            logging.warning(
                "Non-positive Δt between windows at %s and %s; clamping to eps.",
                valid_windows[i].midpoint,
                valid_windows[i + 1].midpoint,
            )
            dt_hours = np.finfo(np.float64).eps
        delta_hours[i] = dt_hours

    rotation_angles = None
    if co_rotate:
        period_hours = rotation_period_hours
        if period_hours is None:
            period_hours = DEFAULT_SYNODIC_PERIOD_DAYS * 24.0
        if period_hours == 0.0:
            raise ValueError(
                "rotation_period_hours must be non-zero when co_rotate=True"
            )
        direction = 1.0 if period_hours > 0 else -1.0
        omega = 2.0 * np.pi / abs(period_hours)
        rotation_angles = direction * omega * delta_hours
        logging.info(
            "Applying co-rotating temporal coupling with period %.3f days (%s)",
            abs(period_hours) / 24.0,
            "forward" if direction > 0 else "reverse",
        )

    # Step 2: Build block-diagonal design matrix X_block
    X_blocks = []
    Phi_parts = []

    for window in valid_windows:
        X_i = _build_harmonic_design(window.lat, window.lon, lmax)
        Phi_i = window.potential.astype(np.complex128)

        X_blocks.append(csr_matrix(X_i))
        Phi_parts.append(Phi_i)

    X_block = block_diag(X_blocks, format="csr")
    Phi_stack = np.concatenate(Phi_parts)

    # Step 3: Build spatial regularization (optional, diagonal)
    n_total = n_windows * n_coeffs
    reg_matrices = [X_block]
    rhs_parts = [Phi_stack]

    if spatial_lambda > 0.0:
        tiled_weights = np.tile(degree_weights, n_windows)
        sqrt_weights = np.sqrt(tiled_weights, out=np.empty_like(tiled_weights))
        diag_entries = sqrt_weights.astype(np.complex128, copy=False)
        R_spatial = np.sqrt(spatial_lambda) * diags(
            diag_entries,
            offsets=0,
            shape=(n_total, n_total),
            format="csr",
        )
        reg_matrices.append(R_spatial)
        rhs_parts.append(np.zeros(n_total, dtype=np.complex128))

    # Step 4: Build temporal regularization
    D_temporal = _build_temporal_derivative_matrix(
        delta_hours,
        lmax,
        rotation_angles=rotation_angles,
    )
    if temporal_lambda > 0.0:
        R_temporal = np.sqrt(temporal_lambda) * D_temporal
        reg_matrices.append(R_temporal)
        rhs_parts.append(np.zeros(D_temporal.shape[0], dtype=np.complex128))

    # Step 5: Stack into augmented system
    A_aug = vstack(reg_matrices, format="csr")
    b_aug = np.concatenate(rhs_parts)

    logging.info(
        "Solving augmented system: %d equations, %d unknowns, %d non-zeros",
        A_aug.shape[0],
        A_aug.shape[1],
        A_aug.nnz,
    )

    # Step 6: Solve sparse least-squares
    try:
        coeffs_stacked, istop, itn, *_ = lsqr(A_aug, b_aug, atol=1e-8, btol=1e-8)
        if istop not in [1, 2]:
            logging.warning(
                "lsqr converged with status %d after %d iterations", istop, itn
            )
    except Exception as exc:
        logging.error("Failed to solve augmented system: %s", exc)
        raise

    # Step 7: Reshape and compute residuals
    results = []
    for i, window in enumerate(valid_windows):
        start_idx = i * n_coeffs
        end_idx = (i + 1) * n_coeffs

        coeffs_i = coeffs_stacked[start_idx:end_idx]

        # Enforce reality condition for physical solution
        coeffs_i = _enforce_reality_condition(coeffs_i, lmax)

        # Compute RMS residual for this window
        X_i = X_blocks[i]
        predicted = np.real(X_i @ coeffs_i)
        residuals = window.potential - predicted
        rms = float(np.sqrt(np.mean(residuals**2)))

        coverage = _compute_spatial_coverage(window.lat, window.lon)

        results.append(
            HarmonicCoefficients(
                time=window.midpoint,
                lmax=lmax,
                coeffs=coeffs_i,
                n_samples=window.lat.size,
                spatial_coverage=coverage,
                rms_residual=rms,
            )
        )

    return results


def compute_temporal_harmonics(
    cache_dir: Path,
    start_date: np.datetime64,
    end_date: np.datetime64,
    lmax: int,
    window_hours: float = 24.0,
    stride_hours: float | None = None,
    l2_penalty: float = 0.0,
    temporal_lambda: float = 0.0,
    min_samples: int = 100,
    min_coverage: float = 0.1,
    co_rotate: bool = False,
    rotation_period_days: float = DEFAULT_SYNODIC_PERIOD_DAYS,
    spatial_weight_exponent: float | None = None,
) -> list[HarmonicCoefficients]:
    """
    Compute time-dependent spherical harmonic coefficients a_lm(t).

    Args:
        cache_dir: Directory containing NPZ potential cache files
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        lmax: Maximum spherical harmonic degree
        window_hours: Duration of each temporal window
        stride_hours: Temporal stride in hours (None = non-overlapping)
        l2_penalty: Ridge regularization strength (spatial)
        temporal_lambda: Temporal continuity regularization strength
        min_samples: Minimum measurements per window
        min_coverage: Minimum spatial coverage fraction (0-1)
        co_rotate: Rotate the temporal derivative into a solar co-rotating frame
        rotation_period_days: Period (days) used when co_rotate=True
        spatial_weight_exponent: Optional exponent for degree-weighted spatial damping

    Returns:
        List of HarmonicCoefficients, one per valid time window.
    """
    if co_rotate and rotation_period_days == 0.0:
        raise ValueError("rotation_period_days must be non-zero when co_rotate=True")
    if spatial_weight_exponent is not None and spatial_weight_exponent < 0.0:
        raise ValueError("spatial_weight_exponent must be non-negative")

    logging.info("Discovering NPZ files in %s", cache_dir)
    files = _discover_npz(cache_dir)
    logging.info("Found %d NPZ files", len(files))

    start_ts = start_date.astype("datetime64[s]")
    end_ts_exclusive = (end_date + np.timedelta64(1, "D")).astype("datetime64[s]")

    logging.info("Loading measurements from %s to %s", start_ts, end_ts_exclusive)
    utc, lat, lon, potential = _load_all_data(files, start_ts, end_ts_exclusive)
    logging.info("Loaded %d measurements", utc.size)

    if utc.size == 0:
        logging.warning("No measurements found in date range")
        return []

    stride_info = (
        f" with {stride_hours:.1f}h stride"
        if stride_hours and stride_hours != window_hours
        else ""
    )
    logging.info("Partitioning into %.1f-hour windows%s", window_hours, stride_info)
    windows = list(
        _partition_into_windows(
            utc,
            lat,
            lon,
            potential,
            window_hours,
            stride_hours,
            start_anchor=start_ts,
        )
    )
    logging.info("Created %d time windows", len(windows))

    degree_weights = _build_degree_weight_vector(lmax, spatial_weight_exponent)

    # Choose fitting strategy based on temporal_lambda
    if temporal_lambda > 0.0:
        logging.info("Using coupled fitting with temporal regularization")
        results = _fit_coupled_windows(
            windows,
            lmax=lmax,
            spatial_lambda=l2_penalty,
            temporal_lambda=temporal_lambda,
            min_samples=min_samples,
            min_coverage=min_coverage,
            degree_weights=degree_weights,
            co_rotate=co_rotate,
            rotation_period_hours=rotation_period_days * 24.0,
        )
    else:
        logging.info("Using independent window fitting (no temporal coupling)")
        results: list[HarmonicCoefficients] = []
        for i, window in enumerate(windows):
            logging.debug(
                "Fitting window %d/%d: %s (%d samples)",
                i + 1,
                len(windows),
                window.midpoint,
                window.lat.size,
            )

            result = _fit_window_harmonics(
                window,
                lmax=lmax,
                l2_penalty=l2_penalty,
                min_samples=min_samples,
                min_coverage=min_coverage,
                degree_weights=degree_weights,
            )

            if result is not None:
                results.append(result)
                logging.info(
                    "Window %s: %d samples, %.1f%% coverage, RMS=%.2f V",
                    result.time,
                    result.n_samples,
                    result.spatial_coverage * 100,
                    result.rms_residual,
                )

    logging.info("Successfully fitted %d/%d windows", len(results), len(windows))
    return results


def save_temporal_coefficients(
    results: list[HarmonicCoefficients], output_path: Path
) -> None:
    """
    Save time-dependent coefficients to NPZ file.

    Storage format:
        times: array of datetime64[ns] midpoints
        lmax: scalar integer
        coeffs: complex array of shape (n_windows, (lmax+1)^2)
        n_samples: integer array of sample counts
        spatial_coverage: float array of coverage fractions
        rms_residuals: float array of RMS fit residuals
    """
    if not results:
        raise ValueError("No results to save")

    times = np.array([r.time for r in results], dtype="datetime64[ns]")
    lmax = results[0].lmax
    n_coeffs = _harmonic_coefficient_count(lmax)

    coeffs = np.array([r.coeffs for r in results], dtype=np.complex128)
    n_samples = np.array([r.n_samples for r in results], dtype=np.int32)
    coverage = np.array([r.spatial_coverage for r in results], dtype=np.float64)
    rms = np.array([r.rms_residual for r in results], dtype=np.float64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        times=times,
        lmax=lmax,
        coeffs=coeffs,
        n_samples=n_samples,
        spatial_coverage=coverage,
        rms_residuals=rms,
    )
    logging.info("Saved temporal coefficients to %s", output_path)



