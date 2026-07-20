#!/usr/bin/env python3
"""Priority-0 check: is the ~-100 V median partly an r>=0.3 bound artifact?

Fresh-look §6.1A - three parts, all using the existing forward model + fitter
bounds. Write nothing into the paper narrative until this resolves.

Parts
-----
(A1) Injection through an unmodified-style objective with the default
     ``LOSS_CONE_BS_OVER_BM_MIN = 0.3`` floor. Truth ``U ∈ {0,-10,-30}`` V at
     weak (r_t ∈ [0.02,0.3)) vs strong (r_t ∈ [0.3,1]) mirrors. The artifact
     hypothesis predicts the truth-near-zero *weak-mirror* population piles near
     ``-K_eff(1 - r_t/0.3) ~ -(80-120) V``.

(A2) Optional archive census: if a batch NPZ is supplied, tabulate fitted
     ``r`` for sweeps near the -108 V median. The mechanism operates only
     through the active bound → predicts ``r`` pinned at 0.3.

(A3) Refit the same injections with ``r ∈ (0,1]`` (floor removed). If the
     pile-up collapses, the floor is implicated.

Examples
--------
  uv run python scripts/diagnostics/r_bound_artifact_check.py
  uv run python scripts/diagnostics/r_bound_artifact_check.py \\
      --batch-npz artifacts/potential_cache/.../potential_batch_....npz \\
      --median-window-v 20
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from src import config
from src.model import synth_losscone


@dataclass
class InjectionTrial:
    true_u: float
    true_r: float
    fitted_u: float
    fitted_r: float
    chi2: float
    r_floor: float
    regime: str  # "weak" | "strong"


def _fit_halekas_style(
    energies: np.ndarray,
    pitches: np.ndarray,
    flux: np.ndarray,
    *,
    u_spacecraft: float,
    r_min: float,
    r_max: float,
    seed: int,
) -> tuple[float, float, float]:
    eps = 1e-6
    data_mask = flux > 0
    log_data = np.log(flux + eps)
    pitch_2d = np.broadcast_to(pitches, (len(energies), len(pitches)))

    def chi2_scalar(params: np.ndarray) -> float:
        u_surf, bs, beam_amp = params
        model = synth_losscone(
            energy_grid=energies,
            pitch_grid=pitch_2d,
            U_surface=float(u_surf),
            U_spacecraft=u_spacecraft,
            bs_over_bm=float(bs),
            beam_width_eV=config.LOSS_CONE_BEAM_WIDTH_EV,
            beam_amp=float(beam_amp),
            beam_pitch_sigma_deg=config.LOSS_CONE_BEAM_PITCH_SIGMA_DEG,
            background=config.LOSS_CONE_BACKGROUND,
        )
        if not np.all(np.isfinite(model)) or (model <= 0).all():
            return 1e30
        log_model = np.log(model + eps)
        diff = (log_data - log_model) * data_mask
        chi2 = float(np.sum(diff * diff))
        return chi2 if np.isfinite(chi2) else 1e30

    bounds = [
        (config.LOSS_CONE_U_SURFACE_MIN, config.LOSS_CONE_U_SURFACE_MAX),
        (float(r_min), float(r_max)),
        (config.LOSS_CONE_BEAM_AMP_MIN, config.LOSS_CONE_BEAM_AMP_MAX),
    ]
    result = differential_evolution(
        chi2_scalar,
        bounds,
        maxiter=80,
        tol=1e-3,
        seed=seed,
        workers=1,
        updating="deferred",
    )
    u, r, _a = result.x
    return float(u), float(r), float(result.fun)


def _make_spectrum(
    u: float, r: float, *, u_spacecraft: float, noise: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    energies = np.geomspace(20, 20000, config.SWEEP_ROWS)
    pitches = np.linspace(0, 180, config.CHANNELS)
    pitch_2d = np.broadcast_to(pitches, (len(energies), len(pitches)))
    model = synth_losscone(
        energy_grid=energies,
        pitch_grid=pitch_2d,
        U_surface=u,
        U_spacecraft=u_spacecraft,
        bs_over_bm=r,
        beam_width_eV=config.LOSS_CONE_BEAM_WIDTH_EV,
        beam_amp=0.0,
        beam_pitch_sigma_deg=config.LOSS_CONE_BEAM_PITCH_SIGMA_DEG,
        background=config.LOSS_CONE_BACKGROUND,
    )
    if noise > 0:
        model = model * np.exp(rng.normal(0.0, noise, size=model.shape))
        model = np.clip(model, config.LOSS_CONE_BACKGROUND, 1.0)
    return energies, pitches, model


def run_injection_panel(
    *,
    r_floor: float,
    n_per_cell: int = 8,
    noise: float = 0.1,
    u_spacecraft: float = 0.0,
    seed0: int = 0,
) -> list[InjectionTrial]:
    true_us = (0.0, -10.0, -30.0)
    weak_rs = np.linspace(0.05, 0.25, 4)
    strong_rs = np.linspace(0.35, 0.95, 4)
    trials: list[InjectionTrial] = []
    k = 0
    for regime, rs in (("weak", weak_rs), ("strong", strong_rs)):
        for u in true_us:
            for r in rs:
                for _rep in range(n_per_cell):
                    seed = seed0 + k
                    k += 1
                    energies, pitches, flux = _make_spectrum(
                        u, float(r), u_spacecraft=u_spacecraft, noise=noise, seed=seed
                    )
                    fu, fr, chi2 = _fit_halekas_style(
                        energies,
                        pitches,
                        flux,
                        u_spacecraft=u_spacecraft,
                        r_min=r_floor,
                        r_max=config.LOSS_CONE_BS_OVER_BM_MAX,
                        seed=seed,
                    )
                    trials.append(
                        InjectionTrial(
                            true_u=float(u),
                            true_r=float(r),
                            fitted_u=fu,
                            fitted_r=fr,
                            chi2=chi2,
                            r_floor=float(r_floor),
                            regime=regime,
                        )
                    )
    return trials


def summarize_trials(trials: list[InjectionTrial]) -> dict:
    out: dict = {}
    for regime in ("weak", "strong"):
        for u in (0.0, -10.0, -30.0):
            sel = [t for t in trials if t.regime == regime and t.true_u == u]
            if not sel:
                continue
            fu = np.array([t.fitted_u for t in sel])
            fr = np.array([t.fitted_r for t in sel])
            key = f"{regime}_U{int(u)}"
            out[key] = {
                n: float(v)
                for n, v in {
                    "n": len(sel),
                    "fitted_u_median": float(np.median(fu)),
                    "fitted_u_p16": float(np.percentile(fu, 16)),
                    "fitted_u_p84": float(np.percentile(fu, 84)),
                    "fitted_r_median": float(np.median(fr)),
                    "frac_r_at_floor": float(
                        np.mean(np.abs(fr - sel[0].r_floor) < 0.02)
                    ),
                    "frac_u_in_80_120": float(np.mean((fu <= -80.0) & (fu >= -120.0))),
                }.items()
            }
    return out


def archive_r_census(
    batch_npz: Path, *, median_target: float = -108.0, window_v: float = 20.0
) -> dict:
    data = np.load(batch_npz, allow_pickle=False)
    # Prefer spec-level arrays; fall back to rows.
    if "spec_potential" in data.files:
        u = np.asarray(data["spec_potential"], dtype=np.float64)
        r = np.asarray(data["spec_bs_over_bm"], dtype=np.float64)
    elif "rows_potential" in data.files:
        u = np.asarray(data["rows_potential"], dtype=np.float64)
        r = np.asarray(data["rows_bs_over_bm"], dtype=np.float64)
    else:
        # Common batch schema variants.
        keys = set(data.files)
        u_key = next(
            (
                k
                for k in ("projected_potential", "u_surface", "spec_u_surface")
                if k in keys
            ),
            None,
        )
        r_key = next(
            (
                k
                for k in ("bs_over_bm", "spec_bs_over_bm", "rows_bs_over_bm")
                if k in keys
            ),
            None,
        )
        if u_key is None or r_key is None:
            raise KeyError(f"NPZ missing potential/r fields; available: {sorted(keys)}")
        u = np.asarray(data[u_key], dtype=np.float64)
        r = np.asarray(data[r_key], dtype=np.float64)

    m = np.isfinite(u) & np.isfinite(r)
    near = m & (np.abs(u - median_target) <= window_v)
    r_near = r[near]
    floor = float(config.LOSS_CONE_BS_OVER_BM_MIN)
    return {
        "n_total_finite": int(np.count_nonzero(m)),
        "n_near_median": int(np.count_nonzero(near)),
        "median_u_near": float(np.median(u[near])) if np.any(near) else float("nan"),
        "median_r_near": float(np.median(r_near)) if r_near.size else float("nan"),
        "frac_r_at_floor": float(np.mean(np.abs(r_near - floor) < 0.02))
        if r_near.size
        else float("nan"),
        "r_floor": floor,
        "window_v": window_v,
        "median_target": median_target,
    }


def _verdict(with_floor: dict, no_floor: dict) -> str:
    """Heuristic decision text for the artifact hypothesis."""
    weak0 = with_floor.get("weak_U0", {})
    weak0_free = no_floor.get("weak_U0", {})
    pile = float(weak0.get("frac_u_in_80_120", 0.0))
    pile_free = float(weak0_free.get("frac_u_in_80_120", 0.0))
    med = float(weak0.get("fitted_u_median", 0.0))
    med_free = float(weak0_free.get("fitted_u_median", 0.0))
    if pile >= 0.3 and med <= -60 and pile_free < pile * 0.5:
        return (
            "SUPPORTS artifact hypothesis: weak-mirror truth-zero fits pile near "
            f"-80...-120 V with r floor (median={med:.1f} V, frac={pile:.2f}) and "
            f"the pile-up collapses when the floor is removed "
            f"(median={med_free:.1f} V, frac={pile_free:.2f})."
        )
    if abs(med) < 40 and pile < 0.15:
        return (
            "REFUTES artifact hypothesis for this forward model: weak-mirror "
            f"truth-zero median stays near zero ({med:.1f} V)."
        )
    return (
        "INCONCLUSIVE with current sample: "
        f"with-floor weak_U0 median={med:.1f} V frac_80_120={pile:.2f}; "
        f"no-floor median={med_free:.1f} V frac={pile_free:.2f}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-cell", type=int, default=4)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-npz", type=Path, default=None)
    parser.add_argument("--median-window-v", type=float, default=20.0)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("artifacts/diagnostics/r_bound_artifact_check.json"),
    )
    args = parser.parse_args()

    print(
        "=== A1: injections with default r floor "
        f"({config.LOSS_CONE_BS_OVER_BM_MIN}) ==="
    )
    with_floor = run_injection_panel(
        r_floor=float(config.LOSS_CONE_BS_OVER_BM_MIN),
        n_per_cell=args.n_per_cell,
        noise=args.noise,
        seed0=args.seed,
    )
    sum_floor = summarize_trials(with_floor)
    print(json.dumps(sum_floor, indent=2))

    print("\n=== A3: injections with r floor removed (r_min=0.02) ===")
    no_floor = run_injection_panel(
        r_floor=0.02,
        n_per_cell=args.n_per_cell,
        noise=args.noise,
        seed0=args.seed + 10_000,
    )
    sum_free = summarize_trials(no_floor)
    print(json.dumps(sum_free, indent=2))

    census = None
    if args.batch_npz is not None:
        print("\n=== A2: archive r census near -108 V ===")
        census = archive_r_census(args.batch_npz, window_v=args.median_window_v)
        print(json.dumps(census, indent=2))
    else:
        print("\n=== A2: skipped (pass --batch-npz for archive census) ===")

    verdict = _verdict(sum_floor, sum_free)
    print("\n=== VERDICT ===")
    print(verdict)

    payload = {
        "with_r_floor": sum_floor,
        "without_r_floor": sum_free,
        "archive_census": census,
        "verdict": verdict,
        "config_r_floor": float(config.LOSS_CONE_BS_OVER_BM_MIN),
        "n_per_cell": args.n_per_cell,
        "noise": args.noise,
        "seed": args.seed,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))
    # Also keep a human-readable copy under docs for the decision record.
    note = Path("docs/archive/analysis/r_bound_artifact_check.md")
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        "# r>=0.3 bound artifact check (Priority 0)\n\n"
        f"**Verdict:** {verdict}\n\n"
        "Raw JSON summary written to "
        f"`{args.out_json.as_posix()}`.\n\n"
        "Re-run:\n\n"
        "```bash\n"
        "uv run python scripts/diagnostics/r_bound_artifact_check.py\n"
        "```\n"
    )
    print(f"\nWrote {args.out_json} and {note}")


if __name__ == "__main__":
    main()
