"""U-surface identifiability QC and D2 confidence-set NPZ helpers.

Legacy LHS ``u_width_*`` / ``u_is_identifiable_*`` fields measure optimizer
geometry only — they are **not** confidence intervals. Prefer the D2
profile-likelihood confidence-set arrays from
:func:`augment_batch_arrays_with_confidence_sets`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.losscone.confidence_set import ConfidenceSetBatch, SweepConfidenceSet
from src.potential_mapper.npz_io import write_npz_atomic


def _delta_key(delta_reduced: float) -> str:
    """Format delta for use in NPZ keys (e.g., 0.001 -> 0p001)."""
    if delta_reduced <= 0:
        raise ValueError("delta_reduced must be > 0")
    return f"{float(delta_reduced):g}".replace(".", "p")


def align_by_spec_no(
    *,
    target_spec_no: np.ndarray,
    source_spec_no: np.ndarray,
    source_values: np.ndarray,
    fill_value: Any,
) -> np.ndarray:
    """
    Align `source_values` to `target_spec_no` by matching spec numbers.

    Args:
        target_spec_no: (N_target,) target spec numbers
        source_spec_no: (N_source,) source spec numbers
        source_values: (N_source, ...) values aligned with source_spec_no
        fill_value: fill used where no source exists

    Returns:
        (N_target, ...) values aligned to target_spec_no
    """
    target_spec_no = np.asarray(target_spec_no, dtype=np.int64)
    source_spec_no = np.asarray(source_spec_no, dtype=np.int64)
    source_values = np.asarray(source_values)

    if source_spec_no.shape[0] != source_values.shape[0]:
        raise ValueError(
            "source_spec_no and source_values must have the same leading dimension"
        )

    order = np.argsort(source_spec_no)
    source_sorted = source_spec_no[order]
    values_sorted = source_values[order]

    idx = np.searchsorted(source_sorted, target_spec_no)
    match = (idx < source_sorted.size) & (source_sorted[idx] == target_spec_no)

    out_shape = (target_spec_no.size, *source_values.shape[1:])
    out = np.full(out_shape, fill_value, dtype=source_values.dtype)
    if match.any():
        out[match] = values_sorted[idx[match]]
    return out


def broadcast_spec_to_rows(
    *,
    rows_spec_no: np.ndarray,
    spec_spec_no: np.ndarray,
    spec_values: np.ndarray,
    fill_value: Any,
) -> np.ndarray:
    """
    Broadcast spec-level values to row-level by matching `rows_spec_no`.

    Thin wrapper over :func:`align_by_spec_no` (the rows are the alignment
    target, the spectra are the source).

    Args:
        rows_spec_no: (N_rows,) row spec numbers
        spec_spec_no: (N_specs,) spec numbers corresponding to spec_values
        spec_values: (N_specs, ...) values per spec
        fill_value: fill used where row spec does not exist in spec list

    Returns:
        (N_rows, ...) values broadcast per row
    """
    return align_by_spec_no(
        target_spec_no=rows_spec_no,
        source_spec_no=spec_spec_no,
        source_values=spec_values,
        fill_value=fill_value,
    )


def augment_batch_arrays_with_u_width(
    *,
    batch_arrays: dict[str, np.ndarray],
    profile_arrays: dict[str, np.ndarray],
    delta_reduced: float = 0.001,
    identifiable_width_max_v: float = 200.0,
    include_rows: bool = True,
) -> dict[str, np.ndarray]:
    """
    Add U-profile width + identifiability QC fields to a batch NPZ payload.

    The profile NPZ is expected to be the output of
    `scripts/diagnostics/losscone_u_profile.py` and contain:
    - spec_spec_no
    - u_width_dchi2red_<delta>

    The batch NPZ payload is expected to contain:
    - spec_spec_no
    - rows_spec_no

    Added keys:
    - spec_u_width_dchi2red_<delta>
    - spec_u_is_identifiable_dchi2red_<delta>
    - rows_u_width_dchi2red_<delta> (optional)
    - rows_u_is_identifiable_dchi2red_<delta> (optional)
    - u_identifiable_width_max_v_dchi2red_<delta> (scalar)
    - u_identifiable_delta_reduced_dchi2red_<delta> (scalar)
    """
    delta_str = _delta_key(float(delta_reduced))
    profile_width_key = f"u_width_dchi2red_{delta_str}"

    if "spec_spec_no" not in batch_arrays or "rows_spec_no" not in batch_arrays:
        raise KeyError("batch_arrays must contain 'spec_spec_no' and 'rows_spec_no'")
    if "spec_spec_no" not in profile_arrays:
        raise KeyError("profile_arrays must contain 'spec_spec_no'")
    if profile_width_key not in profile_arrays:
        raise KeyError(
            f"profile_arrays is missing '{profile_width_key}' "
            f"(available: {sorted(profile_arrays.keys())})"
        )

    batch_spec_no = batch_arrays["spec_spec_no"]
    prof_spec_no = profile_arrays["spec_spec_no"]
    prof_width = profile_arrays[profile_width_key]

    spec_width = align_by_spec_no(
        target_spec_no=batch_spec_no,
        source_spec_no=prof_spec_no,
        source_values=prof_width,
        fill_value=np.nan,
    ).astype(np.float64, copy=False)

    is_identifiable = np.isfinite(spec_width) & (
        spec_width <= float(identifiable_width_max_v)
    )

    out = dict(batch_arrays)
    out[f"spec_u_width_dchi2red_{delta_str}"] = spec_width
    out[f"spec_u_is_identifiable_dchi2red_{delta_str}"] = is_identifiable.astype(bool)
    out[f"u_identifiable_width_max_v_dchi2red_{delta_str}"] = np.array(
        float(identifiable_width_max_v), dtype=np.float64
    )
    out[f"u_identifiable_delta_reduced_dchi2red_{delta_str}"] = np.array(
        float(delta_reduced), dtype=np.float64
    )

    if include_rows:
        rows_spec_no = batch_arrays["rows_spec_no"]
        out[f"rows_u_width_dchi2red_{delta_str}"] = broadcast_spec_to_rows(
            rows_spec_no=rows_spec_no,
            spec_spec_no=batch_spec_no,
            spec_values=spec_width,
            fill_value=np.nan,
        ).astype(np.float64, copy=False)
        out[f"rows_u_is_identifiable_dchi2red_{delta_str}"] = broadcast_spec_to_rows(
            rows_spec_no=rows_spec_no,
            spec_spec_no=batch_spec_no,
            spec_values=is_identifiable.astype(bool),
            fill_value=False,
        ).astype(bool, copy=False)

    return out


def augment_batch_npz_with_u_width(
    *,
    batch_npz_path: Path,
    profile_npz_path: Path,
    out_npz_path: Path,
    delta_reduced: float = 0.001,
    identifiable_width_max_v: float = 200.0,
    include_rows: bool = True,
    overwrite: bool = False,
) -> None:
    """
    Read batch + profile NPZs, attach U-width QC metrics, and write a new batch NPZ.
    """
    batch_npz_path = Path(batch_npz_path)
    profile_npz_path = Path(profile_npz_path)
    out_npz_path = Path(out_npz_path)

    if out_npz_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {out_npz_path} (use --overwrite)"
        )

    batch_npz = np.load(batch_npz_path, allow_pickle=False)
    profile_npz = np.load(profile_npz_path, allow_pickle=False)
    batch_arrays = {k: batch_npz[k] for k in batch_npz.files}
    profile_arrays = {k: profile_npz[k] for k in profile_npz.files}

    payload = augment_batch_arrays_with_u_width(
        batch_arrays=batch_arrays,
        profile_arrays=profile_arrays,
        delta_reduced=float(delta_reduced),
        identifiable_width_max_v=float(identifiable_width_max_v),
        include_rows=bool(include_rows),
    )
    write_npz_atomic(out_npz_path, payload)


def augment_batch_arrays_with_confidence_sets(
    *,
    batch_arrays: dict[str, np.ndarray],
    spec_nos: np.ndarray,
    confidence_sets: list[SweepConfidenceSet],
    include_rows: bool = True,
    prefix: str = "spec_ci_",
) -> dict[str, np.ndarray]:
    """Attach D2 profile-likelihood confidence-set fields to a batch payload.

    Unlike :func:`augment_batch_arrays_with_u_width`, this stores endpoints,
    component counts, bound-touch flags, and gate reasons — never a lone
    scalar width presented as a CI.
    """
    if "spec_spec_no" not in batch_arrays or "rows_spec_no" not in batch_arrays:
        raise KeyError("batch_arrays must contain 'spec_spec_no' and 'rows_spec_no'")

    ci_batch = ConfidenceSetBatch.from_sets(np.asarray(spec_nos), confidence_sets)
    ci_arrays = ci_batch.to_npz_arrays(prefix=prefix)

    # Align CI arrays onto the batch's spec_spec_no order.
    batch_spec = np.asarray(batch_arrays["spec_spec_no"], dtype=np.int64)
    src_spec = np.asarray(ci_arrays[f"{prefix}spec_no"], dtype=np.int64)
    out = dict(batch_arrays)
    for key, values in ci_arrays.items():
        if key == f"{prefix}observation_level":
            out[key] = values
            continue
        if key == f"{prefix}spec_no":
            out[key] = batch_spec
            continue
        out[key] = align_by_spec_no(
            target_spec_no=batch_spec,
            source_spec_no=src_spec,
            source_values=values,
            fill_value=np.nan if np.asarray(values).dtype.kind == "f" else 0,
        )

    if include_rows:
        rows_spec = batch_arrays["rows_spec_no"]
        for field in (
            "u_hat",
            "r_hat",
            "c_alpha",
            "n_components",
            "is_full_domain",
            "is_one_sided",
            "touches_bound_lo",
            "touches_bound_hi",
        ):
            spec_key = f"{prefix}{field}"
            if spec_key not in out:
                continue
            out[f"rows_ci_{field}"] = broadcast_spec_to_rows(
                rows_spec_no=rows_spec,
                spec_spec_no=batch_spec,
                spec_values=out[spec_key],
                fill_value=np.nan if out[spec_key].dtype.kind == "f" else 0,
            )
    return out
