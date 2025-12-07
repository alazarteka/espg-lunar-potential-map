#!/usr/bin/env python
"""
Compare loss-cone fits with and without the secondary beam component.

Example:
    uv run python scripts/dev/losscone_compare_beam.py \
        --file data/1998/060_090MAR/3D980323.TAB \
        --chunks 0 100 200 300 400 \
        --output scratch/perf_runs/1998-03-23/beam_comparison.csv
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from src import config
from src.flux import ERData, LossConeFitter


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run loss-cone fits with the beam enabled vs disabled and compare metrics."
        )
    )
    parser.add_argument(
        "--file",
        required=True,
        type=Path,
        help="Path to ER .TAB file.",
    )
    parser.add_argument(
        "--theta-file",
        type=Path,
        default=config.DATA_DIR / config.THETA_FILE,
        help="Theta table used for pitch-angle calculations (default: data/theta.tab).",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        nargs="*",
        help="Specific chunk indices (0-based) to evaluate. Mutually exclusive with --max-chunks.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=10,
        help="Evaluate the first N chunks when --chunks is not provided (default: 10).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output path for the comparison table.",
    )
    return parser.parse_args()


def _select_chunks(total_chunks: int, args: argparse.Namespace) -> Iterable[int]:
    if args.chunks:
        for idx in args.chunks:
            if 0 <= idx < total_chunks:
                yield idx
            else:
                raise IndexError(
                    f"Chunk index {idx} out of range (0 <= idx < {total_chunks})."
                )
    else:
        limit = min(args.max_chunks, total_chunks)
        yield from range(limit)


def _make_fitter(
    er: ERData, theta_file: Path, beam_min: float, beam_max: float
) -> LossConeFitter:
    fitter = LossConeFitter(er, str(theta_file))
    fitter.beam_amp_min = beam_min
    fitter.beam_amp_max = beam_max
    fitter.lhs = fitter._generate_latin_hypercube()
    return fitter


def main() -> int:
    args = _parse_args()

    er = ERData(str(args.file))
    total_rows = len(er.data)
    total_chunks = total_rows // config.SWEEP_ROWS

    if total_chunks == 0:
        raise RuntimeError(
            "File contains fewer rows than a single sweep; nothing to fit."
        )

    chunk_indices = list(_select_chunks(total_chunks, args))
    if not chunk_indices:
        raise RuntimeError(
            "No chunk indices selected. Provide --chunks or increase --max-chunks."
        )

    fitter_beam = _make_fitter(
        er,
        args.theta_file,
        config.LOSS_CONE_BEAM_AMP_MIN,
        config.LOSS_CONE_BEAM_AMP_MAX,
    )
    fitter_nobeam = _make_fitter(er, args.theta_file, 0.0, 0.0)

    comparisons: list[dict[str, float]] = []

    for idx in chunk_indices:
        U_surface_b, bs_over_bm_b, beam_amp_b, chi2_b = (
            fitter_beam._fit_surface_potential(idx)
        )
        U_surface_nb, bs_over_bm_nb, beam_amp_nb, chi2_nb = (
            fitter_nobeam._fit_surface_potential(idx)
        )

        comparisons.append(
            {
                "chunk": idx,
                "U_surface_with_beam": U_surface_b,
                "Bs/Bm_with_beam": bs_over_bm_b,
                "beam_amp": beam_amp_b,
                "chi2_with_beam": chi2_b,
                "U_surface_no_beam": U_surface_nb,
                "Bs/Bm_no_beam": bs_over_bm_nb,
                "chi2_no_beam": chi2_nb,
                "chi2_improvement": chi2_nb - chi2_b,
                "U_surface_diff": U_surface_b - U_surface_nb,
                "Bs/Bm_diff": bs_over_bm_b - bs_over_bm_nb,
            }
        )

    df = pd.DataFrame(comparisons).sort_values("chunk")
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", None)
    print(df.to_string(index=False))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"\nSaved comparison to {args.output}")

    avg_improvement = df["chi2_improvement"].mean()
    print(f"\nAverage χ² improvement (no-beam − with-beam): {avg_improvement:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
