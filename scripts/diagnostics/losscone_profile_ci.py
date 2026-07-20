#!/usr/bin/env python3
"""Profile-likelihood confidence sets under the calibrated-flux quasi-likelihood.

This graduates the diagnostic role of ``losscone_u_profile.py`` onto the D2
objective. Legacy Lillis χ²(U) profiles and LHS ``u_width`` metrics remain
available but are **not** confidence intervals.

Examples
--------
Synthetic demo (no archive data required)::

  uv run python scripts/diagnostics/losscone_profile_ci.py --demo

Real sweep (requires ER day tables + optional batch polarity)::

  uv run python scripts/diagnostics/losscone_profile_ci.py \\
    --year 1999 --month 4 --day 29 --spec-no 10 \\
    --batch-npz artifacts/potential_cache/.../potential_batch_....npz \\
    --skip-bootstrap
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.losscone.profile_ci import ProfileCIConfig, fit_profile_confidence_set
from src.losscone.quasi_likelihood import (
    QuasiLikelihoodConfig,
    mean_model_for_params,
    simulate_calibrated_flux,
)
from src.losscone.response_folded import build_calibration_mask


def _demo(args: argparse.Namespace) -> dict:
    energies = np.geomspace(40, 3000, 12)
    pitches = np.linspace(0, 180, 24)
    pitch_2d = np.broadcast_to(pitches, (len(energies), len(pitches)))
    ql = QuasiLikelihoodConfig(
        n_energy_quad=3,
        sigma_rel=0.08,
        use_response_folding=not args.no_response_folding,
    )
    true_u, true_r = -90.0, 0.6
    mean = mean_model_for_params(
        energies,
        pitch_2d,
        u_surface=true_u,
        bs_over_bm=true_r,
        beam_amp=0.15,
        cfg=ql,
    )
    mask = build_calibration_mask(mean)
    rng = np.random.default_rng(args.seed)
    data = simulate_calibrated_flux(mean, mask, cfg=ql, rng=rng)

    cfg = ProfileCIConfig(
        coarse_u_count=args.u_count,
        n_r_grid=args.r_count,
        n_beam_grid=args.beam_count,
        bootstrap_n=0 if args.skip_bootstrap else args.bootstrap_n,
        u_min=args.u_min,
        u_max=args.u_max,
        r_min=args.r_min,
        r_max=args.r_max,
        ql=ql,
        bootstrap_seed=args.seed,
    )
    result = fit_profile_confidence_set(
        data,
        energies,
        pitch_2d,
        polarity=-1,
        cfg=cfg,
        skip_bootstrap=args.skip_bootstrap,
        c_alpha_override=args.c_alpha,
    )
    cs = result.confidence_set
    payload = {
        "mode": "demo",
        "true_u": true_u,
        "true_r": true_r,
        "confidence_set": cs.to_dict(),
        "observation_level": cs.observation_level,
        "note": (
            "This is a calibrated-flux quasi-likelihood confidence set, "
            "not a Poisson/multinomial count CI."
        ),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Run synthetic demo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--u-count", type=int, default=21)
    parser.add_argument("--r-count", type=int, default=9)
    parser.add_argument("--beam-count", type=int, default=5)
    parser.add_argument("--u-min", type=float, default=-400.0)
    parser.add_argument("--u-max", type=float, default=20.0)
    parser.add_argument("--r-min", type=float, default=0.02)
    parser.add_argument("--r-max", type=float, default=1.0)
    parser.add_argument("--bootstrap-n", type=int, default=20)
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument(
        "--c-alpha",
        type=float,
        default=None,
        help="Override bootstrap c_alpha (e.g. 1.0 for asymptotic χ²_1 ~68%)",
    )
    parser.add_argument("--no-response-folding", action="store_true")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("artifacts/diagnostics/profile_ci_demo.json"),
    )
    parser.add_argument(
        "--out-npz",
        type=Path,
        default=Path("artifacts/diagnostics/profile_ci_demo.npz"),
    )
    args = parser.parse_args()

    if not args.demo:
        parser.error(
            "Archive single-sweep loading is not wired in this slice; "
            "pass --demo for the synthetic confidence-set path."
        )

    payload = _demo(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))

    cs = payload["confidence_set"]
    np.savez(
        args.out_npz,
        observation_level=np.array(payload["observation_level"]),
        u_hat=np.array(cs["u_hat"]),
        r_hat=np.array(cs["r_hat"]),
        c_alpha=np.array(cs["c_alpha"]),
        profile_u=np.asarray(cs["profile_u"], dtype=np.float64),
        profile_lambda=np.asarray(cs["profile_lambda"], dtype=np.float64),
        component_lo=np.array([c["lo"] for c in cs["components"]], dtype=np.float64),
        component_hi=np.array([c["hi"] for c in cs["components"]], dtype=np.float64),
        is_full_domain=np.array(cs["is_full_domain"]),
        is_one_sided=np.array(cs["is_one_sided"]),
        gate_reason=np.array(cs["gate_reason"]),
        true_u=np.array(payload["true_u"]),
    )
    print(json.dumps(payload, indent=2))
    print(f"Wrote {args.out_json} and {args.out_npz}")


if __name__ == "__main__":
    main()
