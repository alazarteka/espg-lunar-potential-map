"""
L-curve analysis for temporal regularization parameter selection.

Sweeps temporal_lambda values to find optimal trade-off between
temporal smoothness and data fit quality.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def run_temporal_harmonics(
    temporal_lambda: float,
    start: str,
    end: str,
    lmax: int,
) -> tuple[float, float]:
    """
    Run temporal harmonic fitting with given lambda and return metrics.
    
    Returns:
        (temporal_roughness, data_misfit) tuple
    """
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    try:
        cmd = [
            "uv", "run", "python",
            "scripts/dev/temporal_harmonic_coefficients.py",
            "--start", start,
            "--end", end,
            "--lmax", str(lmax),
            "--temporal-lambda", str(temporal_lambda),
            "--output", str(tmp_path),
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        
        # Load results
        with np.load(tmp_path) as data:
            coeffs = data["coeffs"]
            rms = data["rms_residuals"]
        
        # Compute metrics
        diffs = np.diff(coeffs, axis=0)
        temporal_roughness = np.mean(np.linalg.norm(diffs, axis=1))
        data_misfit = np.median(rms)
        
        return temporal_roughness, data_misfit
    
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def plot_l_curve(
    lambdas: np.ndarray,
    roughness: np.ndarray,
    misfit: np.ndarray,
    output_path: Path | None = None,
) -> None:
    """Plot L-curve showing roughness vs. misfit trade-off."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # L-curve in log-log space
    axes[0].loglog(misfit, roughness, 'o-', markersize=8, linewidth=2)
    
    # Annotate lambda values
    for i, lam in enumerate(lambdas):
        if i % max(1, len(lambdas) // 8) == 0:  # Label every Nth point
            axes[0].annotate(
                f'λ={lam:.1e}',
                (misfit[i], roughness[i]),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=9,
                alpha=0.8,
            )
    
    axes[0].set_xlabel('Data Misfit (RMS residual, V)', fontsize=12)
    axes[0].set_ylabel('Temporal Roughness (mean ||Δa||, V)', fontsize=12)
    axes[0].set_title('L-Curve: Temporal Regularization Trade-off', fontsize=13)
    axes[0].grid(True, alpha=0.3, which='both')
    
    # Individual metric vs. lambda
    ax2 = axes[1]
    ax2.semilogx(lambdas, misfit, 'o-', label='Data Misfit (RMS)', color='C0')
    ax2.set_xlabel('Temporal λ', fontsize=12)
    ax2.set_ylabel('Data Misfit (V)', color='C0', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='C0')
    ax2.grid(True, alpha=0.3)
    
    ax3 = ax2.twinx()
    ax3.semilogx(lambdas, roughness, 's-', label='Temporal Roughness', color='C1')
    ax3.set_ylabel('Temporal Roughness (V)', color='C1', fontsize=12)
    ax3.tick_params(axis='y', labelcolor='C1')
    
    axes[1].set_title('Metrics vs. Regularization Strength', fontsize=13)
    
    fig.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved L-curve plot to {output_path}")
    else:
        plt.show()
    
    plt.close(fig)


def find_elbow(lambdas: np.ndarray, roughness: np.ndarray, misfit: np.ndarray) -> int:
    """
    Find elbow point in L-curve using curvature heuristic.
    
    Returns index of optimal lambda.
    """
    # Normalize to [0, 1] for balanced curvature calculation
    rough_norm = (roughness - roughness.min()) / (roughness.max() - roughness.min())
    misfit_norm = (misfit - misfit.min()) / (misfit.max() - misfit.min())
    
    # Compute distance from line connecting endpoints
    p1 = np.array([misfit_norm[0], rough_norm[0]])
    p2 = np.array([misfit_norm[-1], rough_norm[-1]])
    
    distances = []
    for i in range(len(lambdas)):
        p = np.array([misfit_norm[i], rough_norm[i]])
        # Distance from point to line
        d = np.abs(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)
        distances.append(d)
    
    return int(np.argmax(distances))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="L-curve analysis for temporal regularization parameter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--start",
        default="1998-04-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        default="1998-04-30",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--lmax",
        type=int,
        default=5,
        help="Maximum spherical harmonic degree",
    )
    parser.add_argument(
        "--lambda-min",
        type=float,
        default=1e-4,
        help="Minimum temporal_lambda to test",
    )
    parser.add_argument(
        "--lambda-max",
        type=float,
        default=1e0,
        help="Maximum temporal_lambda to test",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=10,
        help="Number of lambda values to test",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/l_curve"),
        help="Directory for output plots",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    # Generate lambda values to test
    lambdas = np.logspace(
        np.log10(args.lambda_min),
        np.log10(args.lambda_max),
        args.n_points,
    )
    
    print(f"Testing {len(lambdas)} lambda values from {args.lambda_min:.1e} to {args.lambda_max:.1e}")
    print(f"Date range: {args.start} to {args.end}")
    print(f"Using lmax={args.lmax}")
    print()
    
    roughness = np.zeros(len(lambdas))
    misfit = np.zeros(len(lambdas))
    
    for i, lam in enumerate(lambdas):
        print(f"[{i+1}/{len(lambdas)}] Testing λ = {lam:.4e}...", end=" ", flush=True)
        
        try:
            rough, mis = run_temporal_harmonics(
                lam,
                args.start,
                args.end,
                args.lmax,
            )
            roughness[i] = rough
            misfit[i] = mis
            print(f"roughness={rough:.1f} V, misfit={mis:.1f} V")
        
        except subprocess.CalledProcessError as e:
            print(f"FAILED: {e}")
            return 1
    
    # Find optimal lambda
    optimal_idx = find_elbow(lambdas, roughness, misfit)
    optimal_lambda = lambdas[optimal_idx]
    
    print()
    print("="*60)
    print("L-Curve Analysis Results")
    print("="*60)
    print(f"Optimal temporal_lambda: {optimal_lambda:.4e}")
    print(f"  Data misfit:           {misfit[optimal_idx]:.1f} V")
    print(f"  Temporal roughness:    {roughness[optimal_idx]:.1f} V")
    print("="*60)
    print()
    
    # Save results
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "l_curve_data.npz"
    np.savez(
        results_file,
        lambdas=lambdas,
        roughness=roughness,
        misfit=misfit,
        optimal_idx=optimal_idx,
        optimal_lambda=optimal_lambda,
    )
    print(f"Saved L-curve data to {results_file}")
    
    # Plot
    plot_l_curve(
        lambdas,
        roughness,
        misfit,
        output_dir / "l_curve.png",
    )
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
