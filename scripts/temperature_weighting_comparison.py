import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple, List

import matplotlib.pyplot as plt
import numpy as np

import src.config as config
from src.flux import ERData
from src.kappa import Kappa, FitResults
from src.spacecraft_potential import theta_to_temperature_ev


def extract_temperature_eV(fit: Optional[FitResults]) -> Optional[float]:
    """Return Te [eV] from a FitResults, or None if unavailable/invalid."""
    if fit is None or not fit.is_good_fit:
        return None
    d_mag, kappa, theta = fit.params.to_tuple()
    try:
        return float(theta_to_temperature_ev(theta, kappa))
    except Exception:
        return None


def _process_one_file(path: str) -> Tuple[List[float], List[float]]:
    te_w_list: List[float] = []
    te_uw_list: List[float] = []
    try:
        er_data = ERData(path)
        spec_nos = er_data.data[config.SPEC_NO_COLUMN].unique()
        for spec_no in spec_nos:
            try:
                k = Kappa(er_data, spec_no=int(spec_no))
                if not k.is_data_valid:
                    continue
                fit_w = k.fit(n_starts=10, use_fast=True, use_weights=True)
                te_w = extract_temperature_eV(fit_w)
                if te_w is None or not np.isfinite(te_w) or te_w <= 0:
                    continue

                # Rebuild Kappa to avoid internal state side-effects
                k2 = Kappa(er_data, spec_no=int(spec_no))
                if not k2.is_data_valid:
                    continue
                fit_uw = k2.fit(n_starts=10, use_fast=True, use_weights=False)
                te_uw = extract_temperature_eV(fit_uw)
                if te_uw is None or not np.isfinite(te_uw) or te_uw <= 0:
                    continue

                te_w_list.append(te_w)
                te_uw_list.append(te_uw)
            except Exception:
                # Skip this spectrum on any error
                continue
    except Exception:
        # Skip this file on any error
        pass
    return te_w_list, te_uw_list


def main() -> None:
    # Mirror dataset sweep from error_distribution_analysis.py
    data_files = glob.glob(str(config.DATA_DIR / "199*" / "*" / "*.TAB"))

    if not data_files:
        print("No data files found matching pattern data/199*/*/*.TAB")
        return

    # Parallel per-file processing
    n_cores = max(1, os.cpu_count() or 1)
    n_workers = min(n_cores, len(data_files))
    print(f"Detected CPU cores: {n_cores}; using {n_workers} workers")

    te_weighted: list[float] = []
    te_unweighted: list[float] = []

    from tqdm import tqdm

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_process_one_file, f): f for f in data_files}
        with tqdm(total=len(futures), desc="Files", unit="file") as pbar:
            for fut in as_completed(futures):
                w, uw = fut.result()
                if w and uw:
                    te_weighted.extend(w)
                    te_unweighted.extend(uw)
                pbar.update(1)

    if not te_weighted:
        print("No valid temperature pairs were computed.")
        return

    te_weighted_arr = np.array(te_weighted, dtype=float)
    te_unweighted_arr = np.array(te_unweighted, dtype=float)
    delta_te = te_weighted_arr - te_unweighted_arr
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_delta_te = delta_te / te_unweighted_arr

    # Plots ---------------------------------------------------------------------
    # Ensure output directory exists
    Path("temp").mkdir(parents=True, exist_ok=True)

    # 1) Overlay histogram of Te (weighted vs unweighted)
    plt.figure(figsize=(10, 6))
    bins = np.geomspace(
        max(1e-2, np.nanpercentile(np.concatenate([te_weighted_arr, te_unweighted_arr]), 0.5)),
        max(2.0, np.nanpercentile(np.concatenate([te_weighted_arr, te_unweighted_arr]), 99.5)),
        80,
    )
    plt.hist(te_weighted_arr, bins=bins, alpha=0.5, label="Te weighted", histtype="stepfilled")
    plt.hist(te_unweighted_arr, bins=bins, alpha=0.5, label="Te unweighted", histtype="stepfilled")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Electron temperature Te (eV)")
    plt.ylabel("Count (log)")
    plt.title("Weighted vs Unweighted κ-fit Temperatures (original fits)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("temp/te_weighted_unweighted_hist.png", dpi=150)

    # 2) Histogram of absolute and relative differences
    plt.figure(figsize=(10, 6))
    plt.hist(delta_te[np.isfinite(delta_te)], bins=120, alpha=0.8)
    plt.xlabel("ΔTe = Te_w − Te_uw (eV)")
    plt.ylabel("Count")
    plt.title("Difference in Temperatures (weighted − unweighted)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("temp/te_diff_hist.png", dpi=150)

    plt.figure(figsize=(10, 6))
    finite_rel = np.isfinite(rel_delta_te)
    plt.hist(100 * rel_delta_te[finite_rel], bins=120, alpha=0.8)
    plt.xlabel("Relative difference ΔTe/Te_uw (%)")
    plt.ylabel("Count")
    plt.title("Relative Difference in Temperatures (weighted vs unweighted)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("temp/te_rel_diff_hist.png", dpi=150)

    # 3) Scatter Te_w vs Te_uw with 1:1 line
    plt.figure(figsize=(7, 7))
    lim_lo = np.nanpercentile(te_unweighted_arr, 1)
    lim_hi = np.nanpercentile(te_unweighted_arr, 99)
    lim_lo = max(1e-2, lim_lo)
    lim_hi = max(lim_hi, lim_lo * 10)
    plt.loglog(te_unweighted_arr, te_weighted_arr, ".", alpha=0.3)
    grid = np.geomspace(lim_lo, lim_hi, 200)
    plt.plot(grid, grid, "k--", lw=1, label="1:1")
    plt.xlabel("Te unweighted (eV)")
    plt.ylabel("Te weighted (eV)")
    plt.title("Te comparison: weighted vs unweighted (original fits)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("temp/te_weighted_vs_unweighted_scatter.png", dpi=150)

    # Console stats -------------------------------------------------------------
    print("\n--- Te Comparison (original fits) ---")
    print(f"Pairs: {len(te_weighted_arr)}")
    for name, arr in [("weighted", te_weighted_arr), ("unweighted", te_unweighted_arr)]:
        print(
            f"Te {name:10s}: mean={np.mean(arr):.3g} eV, median={np.median(arr):.3g} eV, "
            f"P5={np.percentile(arr,5):.3g}, P95={np.percentile(arr,95):.3g}"
        )
    print(
        f"ΔTe: mean={np.mean(delta_te):.3g} eV, median={np.median(delta_te):.3g} eV, "
        f"P5={np.percentile(delta_te,5):.3g}, P95={np.percentile(delta_te,95):.3g}"
    )
    finite_rel = np.isfinite(rel_delta_te)
    if np.any(finite_rel):
        rel = 100 * rel_delta_te[finite_rel]
        print(
            f"ΔTe/Te_uw: mean={np.mean(rel):.3g} %, median={np.median(rel):.3g} %, "
            f"P5={np.percentile(rel,5):.3g} %, P95={np.percentile(rel,95):.3g} %"
        )


if __name__ == "__main__":
    main()
