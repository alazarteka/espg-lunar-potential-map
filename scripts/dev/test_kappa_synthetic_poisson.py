#!/usr/bin/env python3
"""
Compare weighted vs unweighted Kappa fits on Poisson-noisy synthetic ER data.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from src.kappa import Kappa
from src.utils.synthetic import prepare_synthetic_er_poisson


@dataclass
class FitSummary:
    kappa: float
    theta: float
    chi2: float


def _fit_one(
    seed: int,
    density: float,
    kappa_true: float,
    theta_true: float,
    background_count: float,
    n_starts: int,
    use_convolution: bool,
) -> tuple[FitSummary | None, FitSummary | None, tuple[float, float, float]]:
    er = prepare_synthetic_er_poisson(
        density=density,
        kappa=kappa_true,
        theta=theta_true,
        seed=seed,
        background_count=background_count,
    )

    kappa_w = Kappa(er, 1)
    if not kappa_w.is_data_valid:
        return None, None, (np.nan, np.nan, np.nan)

    sigma_range = (
        float(np.min(kappa_w.sigma_log_flux)),
        float(np.max(kappa_w.sigma_log_flux)),
        float(np.median(kappa_w.sigma_log_flux)),
    )

    fit_w = kappa_w.fit(
        n_starts=n_starts,
        use_fast=True,
        use_weights=True,
        use_convolution=use_convolution,
    )
    fit_uw = Kappa(er, 1).fit(
        n_starts=n_starts,
        use_fast=True,
        use_weights=False,
        use_convolution=use_convolution,
    )

    def pack(fit) -> FitSummary | None:
        if fit is None:
            return None
        params = fit.params
        return FitSummary(
            kappa=float(params.kappa),
            theta=float(params.theta.magnitude),
            chi2=float(fit.error),
        )

    return pack(fit_w), pack(fit_uw), sigma_range


def _summarize(
    label: str,
    fits: list[FitSummary],
    kappa_true: float,
    theta_true: float,
) -> None:
    kappas = np.array([f.kappa for f in fits], dtype=float)
    thetas = np.array([f.theta for f in fits], dtype=float)
    chi2 = np.array([f.chi2 for f in fits], dtype=float)

    kappa_err = kappas - kappa_true
    theta_ratio = thetas / theta_true

    print(f"\n{label}")
    print(f"  kappa: mean={kappas.mean():.3f}, std={kappas.std():.3f}")
    print(f"  kappa error: mean={kappa_err.mean():.3f}, std={kappa_err.std():.3f}")
    print(
        f"  theta ratio: mean={theta_ratio.mean():.3f}, std={theta_ratio.std():.3f}"
    )
    print(f"  chi2 median: {np.median(chi2):.3e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare weighted vs unweighted Kappa fits on Poisson-noisy synthetic ER data.",
    )
    parser.add_argument("--density", type=float, default=1e10, help="Density (m^-3)")
    parser.add_argument("--kappa", type=float, default=4.5, help="True kappa value")
    parser.add_argument("--theta", type=float, default=1e6, help="True theta (m/s)")
    parser.add_argument(
        "--background-count",
        type=float,
        default=0.0,
        help="Background counts per channel added before Poisson sampling",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials")
    parser.add_argument(
        "--n-starts",
        type=int,
        default=10,
        help="Random starts per Kappa fit",
    )
    parser.add_argument(
        "--use-convolution",
        action="store_true",
        help="Enable energy response convolution",
    )
    args = parser.parse_args()

    weighted: list[FitSummary] = []
    unweighted: list[FitSummary] = []
    sigma_ranges: list[tuple[float, float, float]] = []

    for i in range(args.trials):
        seed = args.seed + i
        fit_w, fit_uw, sigma_range = _fit_one(
            seed=seed,
            density=args.density,
            kappa_true=args.kappa,
            theta_true=args.theta,
            background_count=args.background_count,
            n_starts=args.n_starts,
            use_convolution=args.use_convolution,
        )
        sigma_ranges.append(sigma_range)
        if fit_w is not None and fit_uw is not None:
            weighted.append(fit_w)
            unweighted.append(fit_uw)

    print(f"Trials requested: {args.trials}")
    print(f"Trials completed: {len(weighted)}")
    print(
        "sigma_log_flux range (min, max, median) for first trial:",
        sigma_ranges[0],
    )

    if weighted:
        _summarize("Weighted fits", weighted, args.kappa, args.theta)
    if unweighted:
        _summarize("Unweighted fits", unweighted, args.kappa, args.theta)


if __name__ == "__main__":
    main()
