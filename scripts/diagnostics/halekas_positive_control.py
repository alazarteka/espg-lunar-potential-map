#!/usr/bin/env python3
"""
Re-run the published Halekas et al. (2008) positive control for the loss-cone fit.

Halekas et al. (2008) report, for the Lunar Prospector interval nearest
1999-04-29 ~14:45 UT:

    U_M (lunar surface potential) = -160 V
    U_sc (spacecraft potential)   = +11 V
    B_s/B_m (magnetic ratio)      = 0.975

This script fits the sweeps bracketing that time straight from the PDS ER file
and prints the recovered parameters, so the check can be reproduced on demand
instead of by hand. It deliberately needs no SPICE kernels and no batch NPZ: it
loads the .TAB, builds pitch angles from the ER magnetic-field columns, pins the
spacecraft potential to a fixed value, and calls the CPU fitter directly. That
keeps it a test of the loss-cone model + fit, not of the geometry pipeline.

Reference values recovered on the merged smooth-boundary code
(tau = 5 deg, fit_method=lillis, normalization=ratio, U_sc = 0):

    sweep 650, 1999-04-29T14:44:08 UT -> U = -161.4 V, B_s/B_m = 0.993

matching the -157 V / 0.990 quoted in docs/resubmission_scientific_audit.md
("Published Positive Control") and the published -160 V.

Note on --polarity: the sign convention of the field direction used for pitch
angles changes this result completely. `none` (the default, B-hat negated)
reproduces the published value; `plus` rails U at about -1800 V with B_s/B_m
pinned to its lower bound. Anything that changes the polarity convention must
re-run this control.

Usage:
  uv run python scripts/diagnostics/halekas_positive_control.py

  # Report in the paper's absolute convention using the published U_sc
  uv run python scripts/diagnostics/halekas_positive_control.py --u-spacecraft 11

  # Another file / time, wider bracket
  uv run python scripts/diagnostics/halekas_positive_control.py \
    --er-file data/1999/091_120APR/3D990429.TAB --target-utc 14:45 --half-window 5
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from src import config
from src.losscone.cpu import LossConeFitter, PitchAngle
from src.potential_mapper.pipeline import load_all_data

DEFAULT_ER_FILE = Path("data/1999/091_120APR/3D990429.TAB")
DEFAULT_TARGET_UTC = "14:45"

# Halekas et al. (2008), Lunar Prospector interval nearest 1999-04-29 14:45 UT.
PUBLISHED_U_SURFACE_V = -160.0
PUBLISHED_U_SPACECRAFT_V = 11.0
PUBLISHED_BS_OVER_BM = 0.975


def _parse_target_utc(value: str) -> int:
    """Parse ``HH:MM`` or ``HH:MM:SS`` into seconds past midnight UT."""
    parts = value.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"--target-utc must be HH:MM or HH:MM:SS, got {value!r}")
    hours, minutes = int(parts[0]), int(parts[1])
    seconds = int(parts[2]) if len(parts) == 3 else 0
    return hours * 3600 + minutes * 60 + seconds


def _sweep_start_seconds(utc: np.ndarray, n_sweeps: int) -> np.ndarray:
    """Seconds past midnight UT at the first row of each 15-row sweep."""
    times = np.asarray(utc, dtype="datetime64[s]")
    past_midnight = (times - times.astype("datetime64[D]")).astype("timedelta64[s]")
    return past_midnight.astype(np.int64)[: n_sweeps * config.SWEEP_ROWS :][
        :: config.SWEEP_ROWS
    ]


def _resolve_polarity(mode: str, n_rows: int) -> np.ndarray | None:
    """Build the per-row field polarity array for :class:`PitchAngle`."""
    if mode == "none":
        return None
    if mode == "plus":
        return np.ones(n_rows, dtype=np.int8)
    return -np.ones(n_rows, dtype=np.int8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run the Halekas 2008 positive control for the loss-cone fit."
    )
    parser.add_argument(
        "--er-file",
        type=Path,
        default=DEFAULT_ER_FILE,
        help=f"PDS ER .TAB file (default: {DEFAULT_ER_FILE})",
    )
    parser.add_argument(
        "--target-utc",
        type=str,
        default=DEFAULT_TARGET_UTC,
        help=f"Target time as HH:MM[:SS] (default: {DEFAULT_TARGET_UTC})",
    )
    parser.add_argument(
        "--half-window",
        type=int,
        default=3,
        help="Number of sweeps to fit on each side of the target (default: 3)",
    )
    parser.add_argument(
        "--u-spacecraft",
        type=float,
        default=0.0,
        help=(
            "Spacecraft potential [V] held fixed during the fit. 0 reports "
            "U relative to the spacecraft (the manuscript convention); pass 11 to "
            "use the published U_sc (default: 0)"
        ),
    )
    parser.add_argument(
        "--fit-method",
        choices=["halekas", "lillis", "cash"],
        default="lillis",
        help="Loss-cone fitting method (default: lillis, as used for the control)",
    )
    parser.add_argument(
        "--normalization",
        choices=["ratio", "ratio2"],
        default="ratio",
        help="Loss-cone normalization mode (default: ratio)",
    )
    parser.add_argument(
        "--polarity",
        choices=["none", "plus", "minus"],
        default="none",
        help=(
            "Field polarity convention for pitch angles. 'none' matches the "
            "control and the fitter default (default: none)"
        ),
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = parse_args()

    er_file = Path(args.er_file)
    if not er_file.exists():
        raise FileNotFoundError(f"ER file not found: {er_file}")

    er_data = load_all_data([er_file])
    if er_data.data.empty:
        raise RuntimeError(f"No rows loaded from {er_file}")

    n_rows = len(er_data.data)
    n_sweeps = n_rows // config.SWEEP_ROWS
    utc = er_data.data[config.UTC_COLUMN].to_numpy()
    sweep_seconds = _sweep_start_seconds(utc, n_sweeps)

    target_seconds = _parse_target_utc(args.target_utc)
    target_sweep = int(np.argmin(np.abs(sweep_seconds - target_seconds)))

    fitter = LossConeFitter(
        er_data,
        pitch_angle=PitchAngle(
            er_data, polarity=_resolve_polarity(args.polarity, n_rows)
        ),
        spacecraft_potential=np.full(n_rows, float(args.u_spacecraft)),
        normalization_mode=str(args.normalization),
        fit_method=str(args.fit_method),
    )

    print(f"file            {er_file}")
    print(f"rows / sweeps   {n_rows} / {n_sweeps}")
    print(
        f"fit             method={args.fit_method} norm={args.normalization} "
        f"polarity={args.polarity} tau={config.LOSS_CONE_BOUNDARY_TAU_DEG} deg "
        f"U_sc={args.u_spacecraft:g} V"
    )
    print(
        f"published       U_M={PUBLISHED_U_SURFACE_V:g} V  "
        f"U_sc={PUBLISHED_U_SPACECRAFT_V:g} V  Bs/Bm={PUBLISHED_BS_OVER_BM:g} "
        f"(Halekas et al. 2008)"
    )
    print()
    header = f"{'sweep':>6} {'UTC':>21} {'U_surface[V]':>13} {'Bs/Bm':>8} {'beam':>7} {'chi2red':>9}"
    print(header)

    low = max(0, target_sweep - int(args.half_window))
    high = min(n_sweeps, target_sweep + int(args.half_window) + 1)
    recovered: list[float] = []
    for sweep in range(low, high):
        result = fitter.fit_chunk_full(sweep)
        marker = "  <== nearest target" if sweep == target_sweep else ""
        print(
            f"{sweep:>6} {utc[sweep * config.SWEEP_ROWS]!s:>21} "
            f"{result.u_surface:>13.3f} {result.bs_over_bm:>8.4f} "
            f"{result.beam_amp:>7.3f} {result.chi2:>9.4f}{marker}"
        )
        if np.isfinite(result.u_surface):
            recovered.append(float(result.u_surface))

    if recovered:
        median = float(np.median(recovered))
        print(f"\nmedian U_surface over {len(recovered)} fitted sweeps: {median:.3f} V")
        print(
            "absolute U_M = U_surface - U_sc = "
            f"{median - float(args.u_spacecraft):.3f} V"
        )
    else:
        print("\nNo sweep in the window produced a finite fit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
