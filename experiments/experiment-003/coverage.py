"""Matched-simulation voltage-profile coverage for Experiment 003.

The simulation fixes nuisance physics at a common known truth and calibrates
one-dimensional U-profile thresholds on independent realizations. It tests each
objective under its own observation assumptions; it does not test forward-model
misspecification or nuisance-parameter coverage.
"""

from __future__ import annotations

import argparse
import csv
import gc
from dataclasses import replace
from pathlib import Path

import numpy as np
from run import (
    METHOD_CONDITIONAL,
    METHOD_INTERVAL,
    METHOD_LILLIS,
    STUDENT_DF,
    ExperimentContext,
    SpectrumData,
    _conditional_objective,
    _interval_objective,
    _lillis_objective,
    _models,
    _profile_count_mean,
)

HERE = Path(__file__).resolve().parent
TRUTH = np.array([-160.0, 0.975, 0.621])
U_GRID = np.arange(-500.0, 20.1, 2.0)


def _compression_intervals(
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from compression import count_interval, decode_lower

    lower_table = np.array([decode_lower(code) for code in range(256)])
    code = np.searchsorted(lower_table, counts, side="right") - 1
    code = np.clip(code, 0, 255).astype(np.uint8)
    lower = np.empty(counts.shape, dtype=np.float64)
    upper = np.empty(counts.shape, dtype=np.float64)
    for index in np.ndindex(counts.shape):
        interval = count_interval(int(code[index]))
        lower[index] = interval.lower
        upper[index] = np.inf if interval.upper is None else interval.upper
    return code, lower, upper


def _profile_models(data: SpectrumData) -> tuple[np.ndarray, np.ndarray, int]:
    params = np.column_stack(
        (
            U_GRID,
            np.full(len(U_GRID), TRUTH[1]),
            np.full(len(U_GRID), TRUTH[2]),
        )
    )
    models, model_masks = _models(data.ratio, params)
    truth_index = int(np.flatnonzero(TRUTH[0] == U_GRID)[0])
    return models, model_masks, truth_index


def _simulated_data(
    method: str,
    base: SpectrumData,
    truth_model: np.ndarray,
    alpha: float,
    sigma: float,
    lillis_sigma: float,
    rng: np.random.Generator,
) -> SpectrumData:
    if method == METHOD_INTERVAL:
        mu = _profile_count_mean(truth_model[None, :, :], base)[0][0]
        size = 1.0 / alpha
        probability = size / (size + mu)
        counts = rng.negative_binomial(size, probability)
        code, lower, upper = _compression_intervals(counts)
        return replace(
            base,
            count_code=code,
            count_lower=lower,
            count_upper=upper,
        )

    if method == METHOD_CONDITIONAL:
        row_scale = np.nanmedian(base.ratio.raw_flux, axis=1)
        noise = rng.standard_t(STUDENT_DF, size=truth_model.shape) * sigma
        flux = row_scale[:, None] * truth_model * np.exp(noise)
        ratio = replace(base.ratio, raw_flux=flux)
        return replace(base, ratio=ratio)

    noise = rng.normal(0.0, lillis_sigma, size=truth_model.shape)
    norm2d = np.maximum(truth_model + noise, 1e-8)
    ratio2 = replace(base.ratio2, norm2d=norm2d)
    return replace(base, ratio2=ratio2)


def _curve(
    method: str,
    data: SpectrumData,
    models: np.ndarray,
    model_masks: np.ndarray,
    alpha: float,
    sigma: float,
) -> np.ndarray:
    n_valid = max(int(np.count_nonzero(data.mask)), 1)
    if method == METHOD_INTERVAL:
        pieces = []
        for start in range(0, len(models), 32):
            stop = min(start + 32, len(models))
            pieces.append(
                _interval_objective(
                    models[start:stop], model_masks[start:stop], data, alpha
                )
            )
        return np.concatenate(pieces) * n_valid
    if method == METHOD_CONDITIONAL:
        return _conditional_objective(models, model_masks, data, sigma) * n_valid
    reduced = _lillis_objective(models, model_masks, data)
    dof = max(n_valid - 3, 1)
    return reduced * dof


def _evaluate_threshold(
    curves: list[np.ndarray],
    threshold: float,
    truth_index: int,
) -> tuple[float, float, float, float]:
    covered = []
    absolute_error = []
    edge = []
    for curve in curves:
        delta = curve - np.nanmin(curve)
        keep = delta <= threshold
        covered.append(bool(keep[truth_index]))
        estimate = U_GRID[int(np.nanargmin(curve))]
        absolute_error.append(abs(estimate - TRUTH[0]))
        edge.append(bool(keep[0] or keep[-1]))
    error = np.asarray(absolute_error)
    return (
        float(np.mean(covered)),
        float(np.median(error)),
        float(np.quantile(error, 0.9)),
        float(np.mean(edge)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=(METHOD_INTERVAL, METHOD_CONDITIONAL, METHOD_LILLIS),
        required=True,
    )
    parser.add_argument("--n-calibration", type=int, default=30)
    parser.add_argument("--n-evaluation", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument(
        "--data", type=Path, default=Path("data/1999/091_120APR/3D990429.TAB")
    )
    parser.add_argument(
        "--telemetry",
        type=Path,
        default=Path("artifacts/experiments/experiment-002/source/M9911914.B"),
    )
    parser.add_argument(
        "--response",
        type=Path,
        default=Path("experiments/experiment-002/response_audit.csv"),
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("experiments/experiment-003/panel_manifest.csv"),
    )
    args = parser.parse_args()

    alpha = 0.13168487690063668
    sigma = 0.42094407555567925
    context = ExperimentContext(args.data, args.telemetry, args.response, args.panel)
    primary_row = next(row for row in context.panel if row.spec_no == 653)
    base = context.spectrum(primary_row)
    models, model_masks, truth_index = _profile_models(base)
    truth_model = models[truth_index]
    fitted_lillis = models[int(np.argmin(_lillis_objective(models, model_masks, base)))]
    lillis_residual = base.ratio2.norm2d - fitted_lillis
    lillis_sigma = float(np.sqrt(np.mean(np.square(lillis_residual[base.mask]))))
    rng = np.random.default_rng(args.seed)

    curves = []
    total = args.n_calibration + args.n_evaluation
    for _ in range(total):
        simulated = _simulated_data(
            args.method,
            base,
            truth_model,
            alpha,
            sigma,
            lillis_sigma,
            rng,
        )
        curves.append(_curve(args.method, simulated, models, model_masks, alpha, sigma))
        gc.collect()

    calibration = curves[: args.n_calibration]
    evaluation = curves[args.n_calibration :]
    truth_delta = np.array(
        [curve[truth_index] - np.nanmin(curve) for curve in calibration]
    )
    rows = []
    for level in (0.68, 0.95):
        finite_sample_level = min(
            np.ceil((args.n_calibration + 1) * level) / args.n_calibration,
            1.0,
        )
        calibrated_threshold = float(
            np.quantile(truth_delta, finite_sample_level, method="higher")
        )
        coverage, median_error, q90_error, edge_fraction = _evaluate_threshold(
            evaluation, calibrated_threshold, truth_index
        )
        rows.append(
            {
                "method": args.method,
                "threshold_source": "empirical_calibration",
                "nominal_level": level,
                "threshold": calibrated_threshold,
                "evaluation_coverage": coverage,
                "median_abs_error_v": median_error,
                "q90_abs_error_v": q90_error,
                "profile_edge_fraction": edge_fraction,
                "n_calibration": args.n_calibration,
                "n_evaluation": args.n_evaluation,
                "nuisance_parameters_fixed": True,
            }
        )
    if args.method in (METHOD_INTERVAL, METHOD_CONDITIONAL):
        for level, threshold in ((0.68, 0.5), (0.95, 1.920729)):
            coverage, median_error, q90_error, edge_fraction = _evaluate_threshold(
                evaluation, threshold, truth_index
            )
            rows.append(
                {
                    "method": args.method,
                    "threshold_source": "nominal_likelihood",
                    "nominal_level": level,
                    "threshold": threshold,
                    "evaluation_coverage": coverage,
                    "median_abs_error_v": median_error,
                    "q90_abs_error_v": q90_error,
                    "profile_edge_fraction": edge_fraction,
                    "n_calibration": args.n_calibration,
                    "n_evaluation": args.n_evaluation,
                    "nuisance_parameters_fixed": True,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"coverage_{args.method}.csv"
    with output.open("w", encoding="ascii", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"method={args.method} lillis_sigma={lillis_sigma:.8g}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
