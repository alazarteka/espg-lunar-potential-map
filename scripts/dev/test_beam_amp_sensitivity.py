#!/usr/bin/env python3
"""
Test sensitivity of U_surface and Bs/Bm fits to beam amplitude parameter.

Compares several strategies:
1. beam_amp as free parameter (current)
2. beam_amp fixed to median (~25)
3. beam_amp = 0 (no beam)
4. Higher upper bound (100)
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import config
from src.flux import ERData, PitchAngle, LossConeFitter


def fit_with_different_strategies(er_file: Path, n_chunks: int = 100):
    """
    Fit chunks with different beam_amp strategies.

    Returns:
        dict with keys 'free', 'fixed_25', 'no_beam', 'high_bound'
        Each value is array of [U_surface, bs_bm, beam_amp, chi2, chunk_idx]
    """
    print(f"Loading {er_file.name}...")
    er_data = ERData(str(er_file))
    pitch_angle = PitchAngle(er_data, str(config.DATA_DIR / config.THETA_FILE))

    results = {
        'free': [],
        'fixed_25': [],
        'no_beam': [],
        'high_bound': []
    }

    max_chunks = min(n_chunks, len(er_data.data) // config.SWEEP_ROWS)

    print(f"Fitting {max_chunks} chunks with 4 different strategies...")

    for i in range(max_chunks):
        if i % 10 == 0:
            print(f"  Progress: {i}/{max_chunks}")

        try:
            # Strategy 1: Free beam_amp (current default, bounds 0-50)
            fitter = LossConeFitter(er_data, str(config.DATA_DIR / config.THETA_FILE), pitch_angle)
            U_surface, bs_bm, beam_amp, chi2 = fitter._fit_surface_potential(i)
            results['free'].append([U_surface, bs_bm, beam_amp, chi2, i])

            # Strategy 2: Fixed beam_amp = 25
            fitter_fixed = LossConeFitter(er_data, str(config.DATA_DIR / config.THETA_FILE), pitch_angle)
            fitter_fixed.beam_amp_min = 25.0
            fitter_fixed.beam_amp_max = 25.0
            U_surface, bs_bm, beam_amp, chi2 = fitter_fixed._fit_surface_potential(i)
            results['fixed_25'].append([U_surface, bs_bm, beam_amp, chi2, i])

            # Strategy 3: No beam (beam_amp = 0)
            fitter_nobeam = LossConeFitter(er_data, str(config.DATA_DIR / config.THETA_FILE), pitch_angle)
            fitter_nobeam.beam_amp_min = 0.0
            fitter_nobeam.beam_amp_max = 0.0
            U_surface, bs_bm, beam_amp, chi2 = fitter_nobeam._fit_surface_potential(i)
            results['no_beam'].append([U_surface, bs_bm, beam_amp, chi2, i])

            # Strategy 4: High upper bound (100)
            fitter_high = LossConeFitter(er_data, str(config.DATA_DIR / config.THETA_FILE), pitch_angle)
            fitter_high.beam_amp_max = 100.0
            U_surface, bs_bm, beam_amp, chi2 = fitter_high._fit_surface_potential(i)
            results['high_bound'].append([U_surface, bs_bm, beam_amp, chi2, i])

        except Exception as e:
            # Skip failed fits
            continue

    # Convert to arrays
    for key in results:
        results[key] = np.array(results[key])

    return results


def plot_comparison(results: dict, output_path: Path):
    """Plot comparison of different strategies."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Extract baseline (free parameter)
    free = results['free']

    strategies = ['fixed_25', 'no_beam', 'high_bound']
    labels = ['Fixed beam_amp=25', 'No beam (beam_amp=0)', 'High bound (0-100)']

    for idx, (strategy, label) in enumerate(zip(strategies, labels)):
        ax = axes.flat[idx]

        data = results[strategy]

        # Compare U_surface
        U_surface_free = free[:, 0]
        U_surface_strat = data[:, 0]

        ax.scatter(U_surface_free, U_surface_strat, alpha=0.5, s=20)
        ax.plot([-1000, 1000], [-1000, 1000], 'r--', alpha=0.5, label='1:1')
        ax.set_xlabel('U_surface (free beam_amp) [V]')
        ax.set_ylabel(f'U_surface ({label}) [V]')
        ax.set_title(f'{label}')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Calculate statistics
        diff = U_surface_strat - U_surface_free
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        max_diff = np.max(np.abs(diff))

        ax.text(0.05, 0.95, f'Mean diff: {mean_diff:.1f} V\nStd diff: {std_diff:.1f} V\nMax diff: {max_diff:.1f} V',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Fourth panel: beam_amp distribution comparison
    ax = axes.flat[3]
    ax.hist(free[:, 2], bins=20, alpha=0.5, label='Free (0-50)')
    ax.hist(results['high_bound'][:, 2], bins=20, alpha=0.5, label='High bound (0-100)')
    ax.axvline(50, color='r', linestyle='--', alpha=0.5, label='Original upper bound')
    ax.set_xlabel('beam_amp')
    ax.set_ylabel('Count')
    ax.set_title('Beam amplitude distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved comparison plot to: {output_path}")


def print_statistics(results: dict):
    """Print detailed statistics."""
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS RESULTS")
    print("="*70)

    free = results['free']

    print(f"\nBaseline (free beam_amp, bounds 0-50):")
    print(f"  Samples: {len(free)}")
    print(f"  beam_amp hitting upper bound (≥49.9): {np.sum(free[:, 2] >= 49.9)} ({100*np.sum(free[:, 2] >= 49.9)/len(free):.1f}%)")
    print(f"  beam_amp mean: {np.mean(free[:, 2]):.1f}")
    print(f"  beam_amp median: {np.median(free[:, 2]):.1f}")

    strategies = ['fixed_25', 'no_beam', 'high_bound']
    labels = ['Fixed beam_amp=25', 'No beam (beam_amp=0)', 'High bound (0-100)']

    for strategy, label in zip(strategies, labels):
        data = results[strategy]

        print(f"\n{label}:")
        print(f"  Samples: {len(data)}")

        # U_surface comparison
        U_surface_diff = data[:, 0] - free[:, 0]
        print(f"  U_surface difference:")
        print(f"    Mean: {np.mean(U_surface_diff):.2f} V")
        print(f"    Std: {np.std(U_surface_diff):.2f} V")
        print(f"    Max: {np.max(np.abs(U_surface_diff)):.2f} V")
        print(f"    RMS: {np.sqrt(np.mean(U_surface_diff**2)):.2f} V")

        # Bs/Bm comparison
        bs_bm_diff = data[:, 1] - free[:, 1]
        print(f"  Bs/Bm difference:")
        print(f"    Mean: {np.mean(bs_bm_diff):.4f}")
        print(f"    Std: {np.std(bs_bm_diff):.4f}")
        print(f"    Max: {np.max(np.abs(bs_bm_diff)):.4f}")

        # Chi2 comparison
        chi2_ratio = data[:, 3] / free[:, 3]
        print(f"  χ² ratio (strategy/baseline):")
        print(f"    Median: {np.median(chi2_ratio):.2f}")

        if strategy == 'high_bound':
            print(f"  beam_amp hitting new upper bound (≥99.9): {np.sum(data[:, 2] >= 99.9)} ({100*np.sum(data[:, 2] >= 99.9)/len(data):.1f}%)")
            print(f"  beam_amp > 50: {np.sum(data[:, 2] > 50)} ({100*np.sum(data[:, 2] > 50)/len(data):.1f}%)")


def find_er_files(data_dir: Path) -> list:
    """Find all ER .TAB files in data directory."""
    er_files = []
    for year_dir in sorted(data_dir.glob("????/")):
        for month_dir in sorted(year_dir.glob("*/")):
            for tab_file in sorted(month_dir.glob("3D*.TAB")):
                er_files.append(tab_file)
    return er_files


def aggregate_results(all_results: list) -> dict:
    """Aggregate results from multiple dates."""
    aggregated = {
        'free': [],
        'fixed_25': [],
        'no_beam': [],
        'high_bound': []
    }

    for results in all_results:
        for key in aggregated:
            if len(results[key]) > 0:
                aggregated[key].append(results[key])

    # Concatenate arrays
    for key in aggregated:
        if aggregated[key]:
            aggregated[key] = np.vstack(aggregated[key])
        else:
            aggregated[key] = np.array([])

    return aggregated


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test beam amplitude sensitivity")
    parser.add_argument("--er-file", type=Path, help="Path to single ER .TAB file")
    parser.add_argument("--n-dates", type=int, default=15, help="Number of random dates to test")
    parser.add_argument("--chunks-per-date", type=int, default=50, help="Number of chunks per date")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Data directory")
    parser.add_argument("--output", type=Path, default=Path("scratch/beam_amp_sensitivity.png"),
                        help="Output plot path")

    args = parser.parse_args()

    if args.er_file:
        # Single file mode
        print("Running sensitivity test on single file...")
        results = fit_with_different_strategies(args.er_file, args.chunks_per_date)
        print_statistics(results)
        plot_comparison(results, args.output)
    else:
        # Multi-date mode
        print(f"Running sensitivity test on {args.n_dates} random dates...")

        # Find all ER files
        all_er_files = find_er_files(args.data_dir)
        print(f"Found {len(all_er_files)} total ER files")

        if len(all_er_files) == 0:
            print("No ER files found!")
            sys.exit(1)

        # Sample random dates
        np.random.seed(42)  # Reproducible
        if len(all_er_files) < args.n_dates:
            print(f"Warning: Only {len(all_er_files)} files available")
            selected_files = all_er_files
        else:
            indices = np.random.choice(len(all_er_files), args.n_dates, replace=False)
            selected_files = [all_er_files[i] for i in sorted(indices)]

        print(f"\nSelected {len(selected_files)} files:")
        for f in selected_files:
            print(f"  {f}")

        # Run test on each date
        all_results = []
        for i, er_file in enumerate(selected_files):
            print(f"\n[{i+1}/{len(selected_files)}] Processing {er_file.name}...")
            try:
                results = fit_with_different_strategies(er_file, args.chunks_per_date)
                all_results.append(results)
            except Exception as e:
                print(f"  Error: {e}")
                continue

        # Aggregate results
        print(f"\nAggregating results from {len(all_results)} successful dates...")
        combined_results = aggregate_results(all_results)

        # Print statistics
        print_statistics(combined_results)

        # Plot comparison
        plot_comparison(combined_results, args.output)

    print("\n" + "="*70)
    print("RECOMMENDATION:")
    print("="*70)

    # Analyze results and provide recommendation
    if args.er_file:
        free = results['free']
        fixed = results['fixed_25']
    else:
        free = combined_results['free']
        fixed = combined_results['fixed_25']

    U_surface_diff = fixed[:, 0] - free[:, 0]
    rms_diff = np.sqrt(np.mean(U_surface_diff**2))

    if rms_diff < 10:
        print("✓ U_surface is very robust to beam_amp choice (RMS < 10 V)")
        print("  → Consider fixing beam_amp to simplify fitting")
    elif rms_diff < 50:
        print("⚠ U_surface shows moderate sensitivity to beam_amp (RMS < 50 V)")
        print("  → Document this uncertainty but current approach is reasonable")
    else:
        print("✗ U_surface is highly sensitive to beam_amp (RMS > 50 V)")
        print("  → Need to investigate further or reconsider model")
