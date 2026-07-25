"""Export legacy (this-repo) golden numbers for the new ``lpref`` parity tests.

This is repo-maintenance tooling, not part of the science pipeline. It freezes
the *current* ``main``-line behaviour of the loss-cone forward model, the two
chi2 objectives, flux normalization, pitch-angle geometry, SPICE footprints,
the Halekas J(U) curve, the kappa flux, and a block of real per-sweep fits, so
that a rewrite can be checked against it bin-for-bin.

Everything runs on the **CPU float64 path** (``use_torch=False``). Torch is
never imported.

Reference: ``scratch/review-2026-07-25/07-salvage-map.md`` section 4
(parity-harness spec) and sections 3.3-3.4 (golden numbers / empirical facts).

Outputs (into ``--out``, default ``scratch/goldens-export/``):

===========================  ==================================================
``legacy_forward_model.npz`` ``synth_losscone`` on sweep 0's (E, pitch) grid for
                             a seeded Sobol grid of (U, Bs/Bm, beam_amp) triples,
                             in both boundary modes (tau=5 sigmoid, tau=0 step).
``legacy_objectives.npz``    Halekas log-chi2 and Lillis masked-linear reduced
                             chi2 for the same triples, both tau modes.
``legacy_norm2d.npz``        Normalized flux for the first 20 sweeps.
``legacy_pitch_angles.npz``  Pitch angles for the first 20 sweeps, computed with
                             BOTH ``polarity=None`` and the SPICE-derived
                             polarity, each labelled with its argument.
``legacy_footprints.npz``    lat/lon/polarity for the first 500 connected rows.
``legacy_ju_curve.npz``      J(U) on 200 points plus the brentq inverse at 20 J.
``legacy_kappa_flux.npz``    Omnidirectional flux on a small (n, T, kappa) grid,
                             numba-fast path and pint reference path.
``legacy_sweep_fits.npz``    Full per-sweep Lillis fits for sweeps 0-49.
``manifest.json``            git sha/branch/UTC time, config snapshot, spot
                             checks, and a sha256 for every npz.
===========================  ==================================================

The npz writer is deterministic: entries are written in sorted order with a
fixed zip timestamp, so two runs produce byte-identical files (``manifest.json``
itself carries a wall-clock timestamp and is expected to differ).

Usage::

    uv run python tools/export_legacy_goldens.py
    uv run python tools/export_legacy_goldens.py --out /tmp/goldens --n-params 256
    uv run python tools/export_legacy_goldens.py --skip-spice   # no kernels
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.lib import format as npformat
from scipy.stats import qmc

from src import config
from src.losscone.chi2 import (
    compute_halekas_chi2_batch,
    compute_lillis_chi2_batch,
)
from src.losscone.cpu import LossConeFitter
from src.losscone.er_data import ERData
from src.losscone.model import _compute_loss_cone_angle, synth_losscone
from src.losscone.normalization import build_norm2d_batch
from src.losscone.pitch_angle import PitchAngle
from src.losscone.types import FitMethod, NormalizationMode
from src.physics.jucurve import DEFAULT_JU_COEFFICIENTS, J_of_U, U_from_J
from src.physics.kappa import (
    KappaParams,
    omnidirectional_flux,
    omnidirectional_flux_fast,
    temperature_ev_to_theta,
)
from src.utils import thetas as thetas_module
from src.utils.units import ureg

logger = logging.getLogger("export_legacy_goldens")

# ---------------------------------------------------------------------------
# Frozen export settings (see salvage map section 4)
# ---------------------------------------------------------------------------
DEFAULT_ER_FILE = config.DATA_DIR / "1999" / "091_120APR" / "3D990429.TAB"
DEFAULT_OUT_DIR = config.PROJECT_ROOT / "scratch" / "goldens-export"

PARAM_GRID_SEED = 42
N_PARAM_TRIPLES = 256  # power of two: balanced Sobol sequence
TAU_MODES = (5.0, 0.0)  # sigmoid (new-repo reference) and hard step (drift)
N_NORM_SWEEPS = 20
N_FOOTPRINT_ROWS = 500
N_POLARITY_CONTRAST_ROWS = 150  # rows per polarity value in the contrast block
N_SWEEP_FITS = 50
FORWARD_SWEEP = 0

# Fixed inputs for the analytic / physics exports.
JU_N_U = 200
JU_U_MIN = 0.0
JU_U_MAX = 150.0
JU_N_INVERSE = 20

KAPPA_DENSITIES_M3 = (1.0e6, 5.0e6, 2.0e7, 1.0e8)
KAPPA_TEMPERATURES_EV = (5.0, 12.0, 24.0, 60.0)
KAPPA_VALUES = (2.0, 3.5, 8.0)
KAPPA_ENERGY_MIN_EV = 7.0
KAPPA_ENERGY_MAX_EV = 20000.0
KAPPA_N_ENERGY = 40

# Lillis identifiability proxy band used by the production Lillis fitter.
LILLIS_DELTA_REDUCED = 0.001

# Fixed zip timestamp so repeated runs are byte-identical.
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def save_npz_deterministic(path: Path, arrays: dict[str, Any]) -> None:
    """Write a ``.npz`` whose bytes depend only on the array contents.

    ``numpy.savez`` stamps each member with the current wall-clock time, which
    makes byte-level reproducibility impossible. This writes the same container
    with entries in sorted order and a fixed zip timestamp.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as zf:
        for name in sorted(arrays):
            value = np.asanyarray(arrays[name])
            if value.dtype == object:
                raise TypeError(f"array {name!r} has object dtype; not exportable")
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            with zf.open(info, "w", force_zip64=True) as fid:
                npformat.write_array(fid, value, allow_pickle=False)


def sha256_of(path: Path) -> str:
    """Return the hex sha256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(cmd: list[str]) -> str:
    """Run a command in the repo root and return stripped stdout ('' on error)."""
    try:
        out = subprocess.run(
            cmd,
            cwd=config.PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        logger.warning("command %s failed: %s", " ".join(cmd), exc)
        return ""
    return out.stdout.strip()


# ---------------------------------------------------------------------------
# Shared inputs
# ---------------------------------------------------------------------------
def build_param_grid(n_params: int, seed: int) -> np.ndarray:
    """Seeded Sobol grid of ``(U_surface, bs_over_bm, beam_amp)`` triples.

    Spans the **full** config bounds, unlike ``utils.losscone_lhs``, which
    silently censors ``U`` to ``[-1000, 0]`` (salvage map section 1.6).
    """
    lower = np.array(
        [
            config.LOSS_CONE_U_SURFACE_MIN,
            config.LOSS_CONE_BS_OVER_BM_MIN,
            config.LOSS_CONE_BEAM_AMP_MIN,
        ],
        dtype=np.float64,
    )
    upper = np.array(
        [
            config.LOSS_CONE_U_SURFACE_MAX,
            config.LOSS_CONE_BS_OVER_BM_MAX,
            config.LOSS_CONE_BEAM_AMP_MAX,
        ],
        dtype=np.float64,
    )
    sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    unit = sampler.random(n_params)
    return np.asarray(qmc.scale(unit, lower, upper), dtype=np.float64)


def sweep_slice(er_data: ERData, sweep: int) -> slice:
    """Row slice for one 15-row ER sweep."""
    start = sweep * config.SWEEP_ROWS
    return slice(start, start + config.SWEEP_ROWS)


def spec_numbers(er_data: ERData, n_sweeps: int) -> np.ndarray:
    """First spec_no of each of the first ``n_sweeps`` sweeps."""
    spec = pd.to_numeric(er_data.data[config.SPEC_NO_COLUMN], errors="coerce").to_numpy(
        dtype=np.float64
    )
    return np.array(
        [spec[s * config.SWEEP_ROWS] for s in range(n_sweeps)], dtype=np.float64
    )


# ---------------------------------------------------------------------------
# Geometry (SPICE)
# ---------------------------------------------------------------------------
class Geometry:
    """SPICE-derived per-row geometry for one ER file."""

    def __init__(
        self,
        polarity: np.ndarray,
        connected: np.ndarray,
        points: np.ndarray,
        lp_positions: np.ndarray,
        projected_b: np.ndarray,
    ):
        self.polarity = polarity
        self.connected = connected
        self.points = points
        self.lp_positions = lp_positions
        self.projected_b = projected_b


def compute_geometry(er_data: ERData) -> Geometry:
    """Run the production geometry path (attitude + SPICE + field-line trace)."""
    from src.potential_mapper.coordinates import (
        CoordinateCalculator,
        find_surface_intersection_with_polarity,
        project_magnetic_fields,
    )
    from src.potential_mapper.spice import load_spice_files
    from src.utils.attitude import load_attitude_data

    load_spice_files()
    et_spin, ra_vals, dec_vals = load_attitude_data(
        config.DATA_DIR / config.ATTITUDE_FILE
    )
    if et_spin is None or ra_vals is None or dec_vals is None:
        raise RuntimeError("attitude data unavailable")

    calculator = CoordinateCalculator(et_spin, ra_vals, dec_vals)
    coords = calculator.calculate_coordinate_transformation(er_data)
    projected_b = project_magnetic_fields(er_data, coords)
    points, mask, polarity = find_surface_intersection_with_polarity(
        coords, projected_b
    )
    return Geometry(
        polarity=polarity.astype(np.int8, copy=False),
        connected=np.asarray(mask, dtype=bool),
        points=np.asarray(points, dtype=np.float64),
        lp_positions=np.asarray(coords.lp_positions, dtype=np.float64),
        projected_b=np.asarray(projected_b, dtype=np.float64),
    )


def _lat_lon(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Latitude/longitude in degrees from IAU_MOON cartesian points."""
    radius = np.linalg.norm(xyz, axis=1)
    lat = np.full(xyz.shape[0], np.nan)
    lon = np.full(xyz.shape[0], np.nan)
    good = radius > 0
    lat[good] = np.rad2deg(np.arcsin(xyz[good, 2] / radius[good]))
    lon[good] = np.rad2deg(np.arctan2(xyz[good, 1], xyz[good, 0]))
    return lat, lon


# ---------------------------------------------------------------------------
# Exporters
# ---------------------------------------------------------------------------
def export_forward_model(
    out_dir: Path,
    er_file: Path,
    energies: np.ndarray,
    pitches: np.ndarray,
    params: np.ndarray,
) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray]]:
    """(a) ``synth_losscone`` outputs for every triple, in both tau modes."""
    arrays: dict[str, Any] = {
        "params": params,
        "param_names": np.array(["U_surface", "bs_over_bm", "beam_amp"]),
        "energy_grid_ev": energies,
        "pitch_grid_deg": pitches,
        "u_spacecraft_v": np.float64(0.0),
        "beam_width_ev": np.float64(config.LOSS_CONE_BEAM_WIDTH_EV),
        "beam_pitch_sigma_deg": np.float64(config.LOSS_CONE_BEAM_PITCH_SIGMA_DEG),
        "background": np.float64(config.LOSS_CONE_BACKGROUND),
        "tau_values_deg": np.array(TAU_MODES, dtype=np.float64),
        "source_file": np.array(er_file.name),
        "source_sweep": np.int64(FORWARD_SWEEP),
        "param_grid_seed": np.int64(PARAM_GRID_SEED),
        "param_grid_kind": np.array("scipy.stats.qmc.Sobol(scramble=True)"),
    }

    models: dict[float, np.ndarray] = {}
    masks: dict[float, np.ndarray] = {}
    for tau in TAU_MODES:
        stack = np.empty(
            (params.shape[0], pitches.shape[0], pitches.shape[1]), dtype=np.float64
        )
        mask = None
        for i, (u_surface, bs_over_bm, beam_amp) in enumerate(params):
            model, valid = synth_losscone(
                energy_grid=energies,
                pitch_grid=pitches,
                U_surface=float(u_surface),
                U_spacecraft=0.0,
                bs_over_bm=float(bs_over_bm),
                beam_width_eV=config.LOSS_CONE_BEAM_WIDTH_EV,
                beam_amp=float(beam_amp),
                beam_pitch_sigma_deg=config.LOSS_CONE_BEAM_PITCH_SIGMA_DEG,
                background=config.LOSS_CONE_BACKGROUND,
                boundary_tau_deg=tau,
                return_mask=True,
            )
            stack[i] = model
            mask = valid
        models[tau] = stack
        masks[tau] = np.asarray(mask, dtype=bool)
        arrays[f"model_tau{tau:g}"] = stack
        arrays[f"valid_mask_tau{tau:g}"] = masks[tau]

    save_npz_deterministic(out_dir / "legacy_forward_model.npz", arrays)
    return models, masks


def export_objectives(
    out_dir: Path,
    er_file: Path,
    energies: np.ndarray,
    pitches: np.ndarray,
    raw_flux: np.ndarray,
    norm2d: np.ndarray,
    params: np.ndarray,
    models: dict[float, np.ndarray],
    masks: dict[float, np.ndarray],
) -> None:
    """(b) Halekas log-chi2 and Lillis masked-linear reduced chi2."""
    halekas_mask = np.isfinite(norm2d) & (norm2d > 0)
    arrays: dict[str, Any] = {
        "params": params,
        "param_names": np.array(["U_surface", "bs_over_bm", "beam_amp"]),
        "energy_grid_ev": energies,
        "pitch_grid_deg": pitches,
        "norm2d": norm2d,
        "raw_flux": raw_flux,
        "eps": np.float64(config.EPS),
        "normalization_mode": np.array(str(NormalizationMode.RATIO)),
        "incident_flux_stat": np.array("mean"),
        "u_spacecraft_v": np.float64(0.0),
        "tau_values_deg": np.array(TAU_MODES, dtype=np.float64),
        "source_file": np.array(er_file.name),
        "source_sweep": np.int64(FORWARD_SWEEP),
        "halekas_mask_retention": np.float64(float(halekas_mask.mean())),
        "lillis_min_valid_bins": np.int64(config.LILLIS_MIN_VALID_BINS),
    }

    for tau in TAU_MODES:
        stack = models[tau]
        # Same E >= U_spacecraft model mask the fitters pass (all-true at U_sc=0).
        valid = masks[tau]
        arrays[f"model_mask_tau{tau:g}"] = valid
        arrays[f"halekas_chi2_tau{tau:g}"] = compute_halekas_chi2_batch(
            norm2d=norm2d, models=stack, model_mask=valid, eps=config.EPS
        )
        arrays[f"lillis_chi2_tau{tau:g}"] = compute_lillis_chi2_batch(
            norm2d=norm2d,
            models=stack,
            raw_flux=raw_flux,
            pitches=pitches,
            model_mask=valid,
        )

    save_npz_deterministic(out_dir / "legacy_objectives.npz", arrays)


def export_norm2d(
    out_dir: Path, er_file: Path, er_data: ERData, fitter: LossConeFitter
) -> None:
    """(c) Normalized flux for the first ``N_NORM_SWEEPS`` sweeps."""
    sweeps = list(range(N_NORM_SWEEPS))
    norm2d = build_norm2d_batch(
        er_data,
        fitter.pitch_angle,
        fitter.normalization_mode,
        fitter.incident_flux_stat,
        sweeps,
    )
    n_rows = N_NORM_SWEEPS * config.SWEEP_ROWS
    raw_flux = er_data.data[config.FLUX_COLS].to_numpy(dtype=np.float64)[:n_rows]
    energies = er_data.data[config.ENERGY_COLUMN].to_numpy(dtype=np.float64)[:n_rows]
    mask = np.isfinite(norm2d) & (norm2d > 0)

    save_npz_deterministic(
        out_dir / "legacy_norm2d.npz",
        {
            "norm2d": norm2d,
            "raw_flux": raw_flux.reshape(N_NORM_SWEEPS, config.SWEEP_ROWS, -1),
            "energies_ev": energies.reshape(N_NORM_SWEEPS, config.SWEEP_ROWS),
            "sweep_indices": np.array(sweeps, dtype=np.int64),
            "spec_no": spec_numbers(er_data, N_NORM_SWEEPS),
            "normalization_mode": np.array(str(fitter.normalization_mode)),
            "incident_flux_stat": np.array(fitter.incident_flux_stat),
            "eps": np.float64(config.EPS),
            # First-class per-sweep Halekas mask retention (salvage map section 4).
            "halekas_mask_retention_per_sweep": mask.mean(axis=(1, 2)),
            "halekas_mask_retention": np.float64(float(mask.mean())),
            "source_file": np.array(er_file.name),
        },
    )


def export_pitch_angles(
    out_dir: Path, er_file: Path, er_data: ERData, geometry: Geometry | None
) -> None:
    """(d) Pitch angles for the first sweeps, with polarity recorded per row."""
    n_rows = N_NORM_SWEEPS * config.SWEEP_ROWS
    none_pitch = PitchAngle(er_data, polarity=None).pitch_angles[:n_rows]

    arrays: dict[str, Any] = {
        "pitch_polarity_none": np.asarray(none_pitch, dtype=np.float64),
        # Explicit record of the polarity ARGUMENT used for each stored variant.
        "polarity_argument_none": np.array("None"),
        # polarity=None takes the -B branch for every row (pitch_angle.py:101-102).
        "polarity_effective_none": np.full(n_rows, -1, dtype=np.int8),
        "row_index": np.arange(n_rows, dtype=np.int64),
        "spec_no": pd.to_numeric(
            er_data.data[config.SPEC_NO_COLUMN], errors="coerce"
        ).to_numpy(dtype=np.float64)[:n_rows],
        "thetas_deg": np.asarray(thetas_module.get_thetas(), dtype=np.float64),
        "phi_deg": er_data.data[config.PHI_COLS].to_numpy(dtype=np.float64)[:n_rows],
        "mag_scd": er_data.data[config.MAG_COLS].to_numpy(dtype=np.float64)[:n_rows],
        "n_sweeps": np.int64(N_NORM_SWEEPS),
        "source_file": np.array(er_file.name),
        "has_spice_variant": np.bool_(geometry is not None),
    }

    if geometry is not None:
        polarity = geometry.polarity
        all_spice_pitch = np.asarray(
            PitchAngle(er_data, polarity=polarity).pitch_angles, dtype=np.float64
        )
        arrays["pitch_polarity_spice"] = all_spice_pitch[:n_rows]
        arrays["polarity_argument_spice"] = np.array(
            "SPICE field-line trace (find_surface_intersection_with_polarity)"
        )
        arrays["polarity_effective_spice"] = polarity[:n_rows].astype(np.int8)
        arrays["variants_identical_in_window"] = np.bool_(
            np.array_equal(none_pitch, all_spice_pitch[:n_rows])
        )

        # The first 20 sweeps can be entirely polarity=-1, in which case the two
        # variants coincide and the golden cannot detect the polarity=+1 branch
        # (pitch_angle.py:103-114). Add a contrast block sampling rows with
        # polarity=+1 and polarity=0 (disconnected -> NaN) so both branches are
        # pinned.
        contrast_idx = np.concatenate(
            [
                np.flatnonzero(polarity == 1)[:N_POLARITY_CONTRAST_ROWS],
                np.flatnonzero(polarity == 0)[:N_POLARITY_CONTRAST_ROWS],
            ]
        ).astype(np.int64)
        all_none_pitch = np.asarray(
            PitchAngle(er_data, polarity=None).pitch_angles, dtype=np.float64
        )
        arrays["contrast_row_index"] = contrast_idx
        arrays["contrast_polarity_effective_spice"] = polarity[contrast_idx].astype(
            np.int8
        )
        arrays["contrast_pitch_polarity_none"] = all_none_pitch[contrast_idx]
        arrays["contrast_pitch_polarity_spice"] = all_spice_pitch[contrast_idx]

    save_npz_deterministic(out_dir / "legacy_pitch_angles.npz", arrays)


def export_footprints(
    out_dir: Path, er_file: Path, er_data: ERData, geometry: Geometry
) -> None:
    """(e) lat/lon/polarity for the first ``N_FOOTPRINT_ROWS`` connected rows."""
    connected_idx = np.flatnonzero(geometry.connected)[:N_FOOTPRINT_ROWS]
    points = geometry.points[connected_idx]
    proj_lat, proj_lon = _lat_lon(points)
    sc_lat, sc_lon = _lat_lon(geometry.lp_positions[connected_idx])

    utc = er_data.data[config.UTC_COLUMN].to_numpy()[connected_idx]
    save_npz_deterministic(
        out_dir / "legacy_footprints.npz",
        {
            "row_index": connected_idx.astype(np.int64),
            "utc": np.asarray(utc, dtype="<U32"),
            "spec_no": pd.to_numeric(
                er_data.data[config.SPEC_NO_COLUMN], errors="coerce"
            ).to_numpy(dtype=np.float64)[connected_idx],
            "projection_latitude_deg": proj_lat,
            "projection_longitude_deg": proj_lon,
            "spacecraft_latitude_deg": sc_lat,
            "spacecraft_longitude_deg": sc_lon,
            "polarity": geometry.polarity[connected_idx].astype(np.int8),
            "intersection_iau_moon_km": points,
            "lp_position_iau_moon_km": geometry.lp_positions[connected_idx],
            "projected_b_iau_moon": geometry.projected_b[connected_idx],
            "lunar_radius_km": np.float64(
                config.LUNAR_RADIUS.to(ureg.kilometer).magnitude
            ),
            "n_connected_total": np.int64(int(geometry.connected.sum())),
            "n_rows_total": np.int64(len(er_data.data)),
            "source_file": np.array(er_file.name),
        },
    )


def export_ju_curve(out_dir: Path) -> None:
    """(f) J(U) on a 200-point grid plus the brentq inverse at 20 J values."""
    u_grid = np.linspace(JU_U_MIN, JU_U_MAX, JU_N_U, dtype=np.float64)
    j_of_u = np.array([J_of_U(float(u)) for u in u_grid], dtype=np.float64)

    # Invert at 20 J values drawn from the curve's own interior range, so the
    # brentq bracket [0, 150] is always valid (salvage map F14).
    u_probe = np.linspace(0.5, 149.5, JU_N_INVERSE, dtype=np.float64)
    j_targets = np.array([J_of_U(float(u)) for u in u_probe], dtype=np.float64)
    u_from_j = np.array(
        [U_from_J(float(j), U_min=JU_U_MIN, U_max=JU_U_MAX) for j in j_targets],
        dtype=np.float64,
    )

    coeff = DEFAULT_JU_COEFFICIENTS
    save_npz_deterministic(
        out_dir / "legacy_ju_curve.npz",
        {
            "u_grid_v": u_grid,
            "j_of_u_a_per_m2": j_of_u,
            "j_targets_a_per_m2": j_targets,
            "u_probe_v": u_probe,
            "u_from_j_v": u_from_j,
            "coefficient_a": np.float64(coeff.A),
            "coefficient_b": np.float64(coeff.B),
            "coefficient_c": np.float64(coeff.C),
            "coefficient_d": np.float64(coeff.D),
            "bracket_u_min_v": np.float64(JU_U_MIN),
            "bracket_u_max_v": np.float64(JU_U_MAX),
        },
    )


def export_kappa_flux(out_dir: Path) -> float:
    """(g) Omnidirectional flux over a small (n, T, kappa) grid.

    Returns the max relative difference between the numba-fast path and the
    pint reference path (a genuine drift check, not a self-comparison).
    """
    densities = np.array(KAPPA_DENSITIES_M3, dtype=np.float64)
    temperatures = np.array(KAPPA_TEMPERATURES_EV, dtype=np.float64)
    kappas = np.array(KAPPA_VALUES, dtype=np.float64)
    energies = np.geomspace(
        KAPPA_ENERGY_MIN_EV, KAPPA_ENERGY_MAX_EV, KAPPA_N_ENERGY
    ).astype(np.float64)
    energy_q = energies * ureg.electron_volt

    shape = (densities.size, temperatures.size, kappas.size, energies.size)
    fast = np.empty(shape, dtype=np.float64)
    ref = np.empty(shape, dtype=np.float64)
    theta_grid = np.empty((temperatures.size, kappas.size), dtype=np.float64)

    flux_units = ureg.particle / (ureg.centimeter**2 * ureg.second * ureg.electron_volt)
    for it, temperature in enumerate(temperatures):
        for ik, kappa in enumerate(kappas):
            theta = temperature_ev_to_theta(float(temperature), float(kappa))
            theta_grid[it, ik] = theta
            for idx, density in enumerate(densities):
                params = KappaParams(
                    density=density * ureg.particle / ureg.meter**3,
                    kappa=float(kappa),
                    theta=theta * ureg.meter / ureg.second,
                )
                fast[idx, it, ik] = (
                    omnidirectional_flux_fast(params, energy_q).to(flux_units).magnitude
                )
                ref[idx, it, ik] = (
                    omnidirectional_flux(params, energy_q).to(flux_units).magnitude
                )

    rel = np.max(np.abs(fast - ref) / np.abs(ref))
    save_npz_deterministic(
        out_dir / "legacy_kappa_flux.npz",
        {
            "density_m3": densities,
            "temperature_ev": temperatures,
            "kappa": kappas,
            "theta_m_per_s": theta_grid,
            "energy_ev": energies,
            "omni_flux_fast": fast,
            "omni_flux_reference": ref,
            "flux_units": np.array("particle / (cm^2 s eV)"),
            "axis_order": np.array("density, temperature, kappa, energy"),
            "fast_vs_reference_max_rel_diff": np.float64(float(rel)),
        },
    )
    return float(rel)


def export_sweep_fits(
    out_dir: Path,
    er_file: Path,
    er_data: ERData,
    geometry: Geometry | None,
    n_fits: int,
) -> None:
    """(h) Full per-sweep Lillis fits, U_sc = 0, tau = 5.0, CPU float64."""
    n_rows = len(er_data.data)
    polarity = geometry.polarity if geometry is not None else None
    pitch_angle = PitchAngle(er_data, polarity=polarity)
    fitter = LossConeFitter(
        er_data,
        pitch_angle=pitch_angle,
        spacecraft_potential=np.zeros(n_rows, dtype=np.float64),
        fit_method=FitMethod.LILLIS,
    )

    fields = {
        key: np.full(n_fits, np.nan, dtype=np.float64)
        for key in ("u_surface", "bs_over_bm", "beam_amp", "chi2", "u_width_lhs")
    }
    n_bins_used = np.zeros(n_fits, dtype=np.int64)
    n_bins_total = np.zeros(n_fits, dtype=np.int64)

    for sweep in range(n_fits):
        # fit_chunk_full() delegates here for FitMethod.LILLIS; calling the QC
        # variant directly gets the identical fit plus the U-width proxy.
        result, u_width = fitter._fit_surface_potential_lillis_with_u_width_qc(
            sweep, delta_reduced=LILLIS_DELTA_REDUCED
        )
        fields["u_surface"][sweep] = result.u_surface
        fields["bs_over_bm"][sweep] = result.bs_over_bm
        fields["beam_amp"][sweep] = result.beam_amp
        fields["chi2"][sweep] = result.chi2
        fields["u_width_lhs"][sweep] = u_width

        chunk = fitter._prepare_chunk_data(sweep)
        if chunk is not None:
            n_bins_used[sweep] = int(
                np.count_nonzero(chunk.combined_mask(FitMethod.LILLIS))
            )
            n_bins_total[sweep] = int(chunk.norm2d.size)

    save_npz_deterministic(
        out_dir / "legacy_sweep_fits.npz",
        {
            "sweep_index": np.arange(n_fits, dtype=np.int64),
            "spec_no": spec_numbers(er_data, n_fits),
            "u_surface": fields["u_surface"],
            "bs_over_bm": fields["bs_over_bm"],
            "beam_amp": fields["beam_amp"],
            "chi2": fields["chi2"],
            "u_width_lhs": fields["u_width_lhs"],
            "n_bins_used": n_bins_used,
            "n_bins_total": n_bins_total,
            "fit_method": np.array(str(FitMethod.LILLIS)),
            "u_spacecraft_v": np.float64(0.0),
            "boundary_tau_deg": np.float64(config.LOSS_CONE_BOUNDARY_TAU_DEG),
            "normalization_mode": np.array(str(fitter.normalization_mode)),
            "incident_flux_stat": np.array(fitter.incident_flux_stat),
            "lhs_seed": np.int64(config.LOSS_CONE_LHS_SEED),
            "lhs_samples": np.int64(config.LOSS_CONE_LHS_SAMPLES),
            "de_seed": np.int64(config.LOSS_CONE_DE_SEED),
            "de_popsize": np.int64(config.LOSS_CONE_DE_POPSIZE),
            "lillis_delta_reduced": np.float64(LILLIS_DELTA_REDUCED),
            "pitch_polarity_source": np.array(
                "spice" if polarity is not None else "none"
            ),
            "use_torch": np.bool_(False),
            "dtype": np.array("float64"),
            "source_file": np.array(er_file.name),
        },
    )


# ---------------------------------------------------------------------------
# Spot checks
# ---------------------------------------------------------------------------
def run_spot_checks(
    er_file: Path,
    energies: np.ndarray,
    pitches: np.ndarray,
    norm2d_all: np.ndarray,
) -> dict[str, Any]:
    """Verify the salvage map's stated empirical facts (section 3.4)."""
    checks: dict[str, Any] = {}

    # alpha_c == 45 degrees exactly for bs/bm = 1, U_sc = 0, U = -E/2.
    angles = [
        float(
            _compute_loss_cone_angle(
                np.array([[[e]]]),
                np.array([[[-0.5 * e]]]),
                np.array([[[0.0]]]),
                np.array([[[1.0]]]),
            )[0, 0, 0]
        )
        for e in energies
    ]
    checks["alpha_c_half_energy_deg"] = angles
    checks["alpha_c_exactly_45"] = all(a == 45.0 for a in angles)

    # The tau = 0 hard-step model must switch at pitch = 180 - 45 = 135 deg.
    probe_e = float(energies[0])
    probe_pitch = np.array([[134.999, 135.0, 135.001]], dtype=np.float64)
    step = synth_losscone(
        energy_grid=np.array([probe_e]),
        pitch_grid=probe_pitch,
        U_surface=-0.5 * probe_e,
        U_spacecraft=0.0,
        bs_over_bm=1.0,
        background=config.LOSS_CONE_BACKGROUND,
        boundary_tau_deg=0.0,
    )
    checks["hard_step_at_135_deg"] = [float(v) for v in np.asarray(step).ravel()]
    checks["hard_step_boundary_ok"] = bool(
        step[0, 0] == 1.0
        and step[0, 1] == 1.0
        and step[0, 2] == config.LOSS_CONE_BACKGROUND
    )

    # Halekas mask retention. The salvage map's 64.45% is measured on the RAW
    # (uncleaned) TAB file's flux bins; the fitters see the sweep-cleaned frame
    # and the normalized array, which retains marginally fewer bins.
    mask = np.isfinite(norm2d_all) & (norm2d_all > 0)
    checks["halekas_mask_retention_norm2d_cleaned"] = float(mask.mean())
    raw = pd.read_csv(
        er_file,
        sep=" ",
        engine="c",
        skipinitialspace=True,
        header=None,
        names=config.ALL_COLS,
    )[config.FLUX_COLS].to_numpy(dtype=np.float64)
    checks["halekas_mask_retention_raw_file"] = float(
        (np.isfinite(raw) & (raw > 0)).mean()
    )
    checks["halekas_mask_retention_expected"] = 0.6445
    checks["raw_rows"] = int(raw.shape[0])
    return checks


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Export legacy golden numbers (CPU float64) for the new repo's "
            "parity tests."
        )
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--er-file",
        type=Path,
        default=DEFAULT_ER_FILE,
        help=f"ER TAB file to snapshot (default: {DEFAULT_ER_FILE})",
    )
    parser.add_argument(
        "--n-params",
        type=int,
        default=N_PARAM_TRIPLES,
        help="number of Sobol parameter triples (default: 256)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=PARAM_GRID_SEED,
        help="parameter-grid seed (default: 42)",
    )
    parser.add_argument(
        "--n-sweep-fits",
        type=int,
        default=N_SWEEP_FITS,
        help="number of sweeps to fit (default: 50)",
    )
    parser.add_argument(
        "--skip-spice",
        action="store_true",
        help="skip SPICE-dependent exports (footprints, polarity pitch angles)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser.parse_args(argv)


def config_snapshot() -> dict[str, Any]:
    """Snapshot every config value the parity harness must freeze."""
    wanted = {"EPS", "N_BG", "FIT_ERROR_THRESHOLD"}
    snapshot: dict[str, Any] = {}
    for name in sorted(dir(config)):
        prefixed = name.startswith("LOSS_CONE_") or name.startswith("LILLIS_")
        if not (prefixed or name in wanted):
            continue
        value = getattr(config, name)
        if isinstance(value, bool | int | float | str):
            snapshot[name] = value
        else:
            snapshot[name] = repr(value)
    return snapshot


def main(argv: list[str] | None = None) -> int:
    """Export every golden artifact and write the manifest."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    er_file: Path = args.er_file
    if not er_file.exists():
        raise FileNotFoundError(f"ER file {er_file} not found")

    logger.info("loading %s", er_file)
    er_data = ERData(str(er_file))
    if er_data.data.empty:
        raise RuntimeError(f"{er_file} loaded empty")
    n_rows = len(er_data.data)
    n_sweeps = n_rows // config.SWEEP_ROWS
    logger.info("loaded %d rows / %d sweeps (post-clean)", n_rows, n_sweeps)

    geometry: Geometry | None = None
    spice_error: str | None = None
    if args.skip_spice:
        spice_error = "skipped via --skip-spice"
    else:
        try:
            geometry = compute_geometry(er_data)
            logger.info(
                "geometry: %d/%d rows connected",
                int(geometry.connected.sum()),
                n_rows,
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            spice_error = f"{type(exc).__name__}: {exc}"
            logger.warning("SPICE geometry unavailable (%s)", spice_error)

    # Base fitter: polarity=None pitch angles, config-default normalization.
    base_pitch = PitchAngle(er_data, polarity=None)
    base_fitter = LossConeFitter(er_data, pitch_angle=base_pitch)

    sl = sweep_slice(er_data, FORWARD_SWEEP)
    energies = er_data.data[config.ENERGY_COLUMN].to_numpy(dtype=np.float64)[sl]
    pitches = np.asarray(base_pitch.pitch_angles[sl], dtype=np.float64)
    raw_flux = er_data.data[config.FLUX_COLS].to_numpy(dtype=np.float64)[sl]
    norm2d_sweep0 = base_fitter.build_norm2d(FORWARD_SWEEP)

    params = build_param_grid(args.n_params, args.seed)

    logger.info("(a) forward model")
    models, masks = export_forward_model(out_dir, er_file, energies, pitches, params)
    logger.info("(b) objectives")
    export_objectives(
        out_dir,
        er_file,
        energies,
        pitches,
        raw_flux,
        norm2d_sweep0,
        params,
        models,
        masks,
    )
    logger.info("(c) norm2d")
    export_norm2d(out_dir, er_file, er_data, base_fitter)
    logger.info("(d) pitch angles")
    export_pitch_angles(out_dir, er_file, er_data, geometry)
    if geometry is not None:
        logger.info("(e) footprints")
        export_footprints(out_dir, er_file, er_data, geometry)
    else:
        logger.warning("(e) footprints SKIPPED: %s", spice_error)
    logger.info("(f) J(U) curve")
    export_ju_curve(out_dir)
    logger.info("(g) kappa flux")
    kappa_rel_diff = export_kappa_flux(out_dir)
    logger.info("(h) sweep fits (%d sweeps)", args.n_sweep_fits)
    export_sweep_fits(out_dir, er_file, er_data, geometry, args.n_sweep_fits)

    logger.info("spot checks")
    all_norm2d = build_norm2d_batch(
        er_data,
        base_pitch,
        base_fitter.normalization_mode,
        base_fitter.incident_flux_stat,
        list(range(n_sweeps)),
    )
    checks = run_spot_checks(er_file, energies, pitches, all_norm2d)
    checks["kappa_fast_vs_reference_max_rel_diff"] = kappa_rel_diff

    files = sorted(p for p in out_dir.glob("*.npz"))
    manifest = {
        "generated_by": "tools/export_legacy_goldens.py",
        "spec": "scratch/review-2026-07-25/07-salvage-map.md section 4",
        "git_sha": _run(["git", "rev-parse", "HEAD"]),
        "git_branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(_run(["git", "status", "--porcelain"])),
        "utc_timestamp": _run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"]),
        "python_path": "CPU float64 (use_torch=False); torch never imported",
        "source_er_file": str(er_file.relative_to(config.PROJECT_ROOT)),
        "source_rows_post_clean": n_rows,
        "source_sweeps_post_clean": n_sweeps,
        "param_grid": {
            "kind": "scipy.stats.qmc.Sobol(d=3, scramble=True)",
            "seed": args.seed,
            "n": args.n_params,
            "bounds": {
                "U_surface": [
                    config.LOSS_CONE_U_SURFACE_MIN,
                    config.LOSS_CONE_U_SURFACE_MAX,
                ],
                "bs_over_bm": [
                    config.LOSS_CONE_BS_OVER_BM_MIN,
                    config.LOSS_CONE_BS_OVER_BM_MAX,
                ],
                "beam_amp": [
                    config.LOSS_CONE_BEAM_AMP_MIN,
                    config.LOSS_CONE_BEAM_AMP_MAX,
                ],
            },
        },
        "boundary_tau_modes_deg": list(TAU_MODES),
        "spice_available": geometry is not None,
        "spice_error": spice_error,
        "config": config_snapshot(),
        "spot_checks": checks,
        "files": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_of(path),
            }
            for path in files
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    logger.info("wrote %d npz + manifest.json to %s", len(files), out_dir)
    for entry in manifest["files"]:
        logger.info(
            "  %-28s %9d B  %s", entry["name"], entry["bytes"], entry["sha256"][:16]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
