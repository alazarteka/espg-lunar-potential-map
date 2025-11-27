"""Compare κ-fit temperatures with and without count weighting."""

import glob
import multiprocessing as mp
import os
import sys
import time
from multiprocessing import Process
from multiprocessing.queues import Queue as MPQueue
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Prevent BLAS/OpenMP oversubscription in workers and avoid after-fork hangs
# Set thread caps before importing numpy/scipy/matplotlib
_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for _v in _THREAD_VARS:
    os.environ.setdefault(_v, "1")

import matplotlib

matplotlib.use("Agg")
import queue as queue_mod

import matplotlib.pyplot as plt
import numpy as np

import src.config as config
from src.flux import ERData
from src.kappa import FitResults, Kappa
from src.spacecraft_potential import theta_to_temperature_ev

# Optional: per-task timeout (seconds) to prevent a stuck file from stalling all work
FILE_TIMEOUT_S = float(os.environ.get("TEMP_WEIGHTING_FILE_TIMEOUT_S", "600"))
# Hard wall-time budget inside worker for one file (seconds); returns partial results
FILE_WALLTIME_S = float(os.environ.get("TEMP_WEIGHTING_FILE_WALLTIME_S", "900"))
# Parent-side kill timeout for a single file task (seconds)
PARENT_KILL_TIMEOUT_S = float(
    os.environ.get("TEMP_WEIGHTING_PARENT_KILL_TIMEOUT_S", str(FILE_WALLTIME_S + 300))
)
# Number of random starts per fit (override via env for speed/testing)
FIT_STARTS = int(os.environ.get("TEMP_WEIGHTING_FIT_STARTS", "10"))


def _init_worker() -> None:
    """Initializer for worker processes to enforce 1 thread for BLAS/OMP libs.

    Called in each spawned worker before executing tasks.
    """
    for v in _THREAD_VARS:
        os.environ.setdefault(v, "1")
    # Ensure stdout/err are unbuffered for timely logs (if any)
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass


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
    start_t = time.monotonic()
    try:
        print(f"[worker] start {path}")
        er_data = ERData(path)
        spec_nos = er_data.data[config.SPEC_NO_COLUMN].unique()
        for spec_no in spec_nos:
            if time.monotonic() - start_t > FILE_WALLTIME_S:
                print(f"[worker] timeout (wall) {path}; returning partial")
                break
            try:
                k = Kappa(er_data, spec_no=int(spec_no))
                if not k.is_data_valid:
                    continue
                fit_w = k.fit(n_starts=FIT_STARTS, use_fast=True, use_weights=True)
                te_w = extract_temperature_eV(fit_w)
                if te_w is None or not np.isfinite(te_w) or te_w <= 0:
                    continue

                # Rebuild Kappa to avoid internal state side-effects
                k2 = Kappa(er_data, spec_no=int(spec_no))
                if not k2.is_data_valid:
                    continue
                fit_uw = k2.fit(n_starts=FIT_STARTS, use_fast=True, use_weights=False)
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
    print(f"[worker] done  {path} -> pairs={min(len(te_w_list), len(te_uw_list))}")
    return te_w_list, te_uw_list


def _worker_entry(file_path: str, result_q: MPQueue) -> None:
    """Spawned worker entrypoint for a single file with controlled env."""
    _init_worker()
    try:
        res = _process_one_file(file_path)
        try:
            result_q.put((file_path, res[0], res[1]))
        except Exception:
            pass
    except Exception:
        # Ensure we at least signal completion
        try:
            result_q.put((file_path, [], []))
        except Exception:
            pass


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

    # Custom spawn-based pool with per-task kill
    mp_ctx = mp.get_context("spawn")
    result_q: MPQueue = mp_ctx.Queue()
    active: Dict[str, tuple[Process, float]] = {}
    pending = list(data_files)

    with tqdm(total=len(data_files), desc="Files", unit="file") as pbar:
        # Launch initial batch
        while pending or active:
            # Start new tasks if capacity
            while pending and len(active) < n_workers:
                f = pending.pop(0)
                p = mp_ctx.Process(
                    target=_worker_entry, args=(f, result_q), daemon=True
                )
                p.start()
                active[f] = (p, time.monotonic())

            # Collect any finished results without blocking too long
            drained = False
            while True:
                try:
                    fpath, w, uw = result_q.get_nowait()
                except queue_mod.Empty:
                    break
                drained = True
                if fpath in active:
                    proc, _ = active.pop(fpath)
                    proc.join(timeout=0.1)
                if w and uw:
                    te_weighted.extend(w)
                    te_unweighted.extend(uw)
                pbar.update(1)

            # Reap processes that exited without posting
            for f, (proc, start_time) in list(active.items()):
                if not proc.is_alive():
                    proc.join(timeout=0.1)
                    active.pop(f)
                    # No result posted; count as empty
                    pbar.update(1)

            # Kill long-runners
            now = time.monotonic()
            for f, (proc, start_time) in list(active.items()):
                if now - start_time > PARENT_KILL_TIMEOUT_S:
                    print(f"[parent] killing long-running task: {f}")
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    proc.join(timeout=1.0)
                    active.pop(f, None)
                    # Count as timeout
                    pbar.update(1)

            # Sleep a bit to avoid busy spin if nothing drained and no state change
            if not drained:
                time.sleep(0.2)

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
    Path("scratch").mkdir(parents=True, exist_ok=True)

    # 1) Overlay histogram of Te (weighted vs unweighted)
    plt.figure(figsize=(10, 6))
    bins = np.geomspace(
        max(
            1e-2,
            np.nanpercentile(np.concatenate([te_weighted_arr, te_unweighted_arr]), 0.5),
        ),
        max(
            2.0,
            np.nanpercentile(
                np.concatenate([te_weighted_arr, te_unweighted_arr]), 99.5
            ),
        ),
        80,
    )
    plt.hist(
        te_weighted_arr,
        bins=bins,
        alpha=0.5,
        label="Te weighted",
        histtype="stepfilled",
    )
    plt.hist(
        te_unweighted_arr,
        bins=bins,
        alpha=0.5,
        label="Te unweighted",
        histtype="stepfilled",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Electron temperature Te (eV)")
    plt.ylabel("Count (log)")
    plt.title("Weighted vs Unweighted κ-fit Temperatures (original fits)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("scratch/te_weighted_unweighted_hist.png", dpi=150)

    # 2) Histogram of absolute and relative differences
    plt.figure(figsize=(10, 6))
    plt.hist(delta_te[np.isfinite(delta_te)], bins=120, alpha=0.8)
    plt.xlabel("ΔTe = Te_w − Te_uw (eV)")
    plt.ylabel("Count")
    plt.title("Difference in Temperatures (weighted − unweighted)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("scratch/te_diff_hist.png", dpi=150)

    plt.figure(figsize=(10, 6))
    finite_rel = np.isfinite(rel_delta_te)
    plt.hist(100 * rel_delta_te[finite_rel], bins=120, alpha=0.8)
    plt.xlabel("Relative difference ΔTe/Te_uw (%)")
    plt.ylabel("Count")
    plt.title("Relative Difference in Temperatures (weighted vs unweighted)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("scratch/te_rel_diff_hist.png", dpi=150)

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
    plt.savefig("scratch/te_weighted_vs_unweighted_scatter.png", dpi=150)

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
