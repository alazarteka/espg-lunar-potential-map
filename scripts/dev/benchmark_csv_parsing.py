#!/usr/bin/env python3
"""
Benchmark different pandas CSV parsing engines for ER data files.
"""
import time
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src import config
from src.potential_mapper.pipeline import DataLoader

def benchmark_engine(file_path, engine, sep=r"\s+", **kwargs):
    """Benchmark a specific pandas engine."""
    start = time.time()
    try:
        df = pd.read_csv(
            file_path,
            sep=sep,
            engine=engine,
            header=None,
            names=config.ALL_COLS,
            **kwargs
        )
        elapsed = time.time() - start
        return elapsed, len(df), None
    except Exception as e:
        elapsed = time.time() - start
        return elapsed, 0, str(e)

def main():
    # Get a test file
    files = DataLoader.discover_flux_files()
    if not files:
        print("No files found!")
        return
    
    test_file = files[0]
    print(f"Benchmarking with: {test_file}")
    print("=" * 80)
    
    # Test configurations
    configs = [
        ("python", r"\s+", {}, "Python engine with regex sep"),
        ("c", r"\s+", {}, "C engine with regex sep (may fail)"),
        ("c", " ", {"skipinitialspace": True}, "C engine with space + skipinitialspace"),
    ]
    
    # Try pyarrow if available
    try:
        import pyarrow
        configs.append(("pyarrow", r"\s+", {}, "PyArrow engine"))
    except ImportError:
        print("PyArrow not installed, skipping...")
    
    results = []
    for engine, sep, kwargs, desc in configs:
        print(f"\nTesting: {desc}")
        print(f"  Engine: {engine}, Sep: {repr(sep)}, Kwargs: {kwargs}")
        
        elapsed, rows, error = benchmark_engine(test_file, engine, sep, **kwargs)
        
        if error:
            print(f"  ❌ FAILED: {error}")
        else:
            print(f"  ✓ Success: {elapsed:.3f}s ({rows} rows)")
            results.append((desc, elapsed, rows))
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    if results:
        # Sort by time
        results.sort(key=lambda x: x[1])
        baseline = results[-1][1]  # slowest
        
        for desc, elapsed, rows in results:
            speedup = baseline / elapsed
            print(f"{desc:50s} {elapsed:6.3f}s  ({speedup:4.1f}x speedup)")
    else:
        print("No successful runs!")

if __name__ == "__main__":
    main()
