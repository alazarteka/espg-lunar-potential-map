
import cProfile
import pstats
import io
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.potential_mapper.pipeline import DataLoader, process_lp_file
from src import config

def profile_run(limit=1):
    """
    Run the potential mapper pipeline on a few files and profile it.
    """
    # Setup logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    # Load SPICE kernels
    from src.potential_mapper.spice import load_spice_files
    load_spice_files()
    
    # Discover files (no date filter to get any available files)
    files = DataLoader.discover_flux_files()
    
    if not files:
        print("No files found!")
        return

    files_to_process = files[:limit]
    print(f"Profiling processing of {len(files_to_process)} files...")
    
    # Profile
    pr = cProfile.Profile()
    pr.enable()
    
    for file_path in files_to_process:
        try:
            print(f"Processing {file_path}...")
            result = process_lp_file(file_path)
            print(f"  Processed {len(result.spacecraft_latitude)} rows")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
            
    pr.disable()
    
    # Print stats - top 50 by cumulative time
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats(50)
    
    print("\n" + "="*80 + "\n")
    print("TOP 50 BY CUMULATIVE TIME")
    print("="*80)
    print(s.getvalue())
    
    # Look specifically for fitting-related functions
    print("\n" + "="*80 + "\n")
    print("FITTING-RELATED FUNCTIONS")
    print("="*80)
    ps.print_stats("fit|synth|minimize|chi2")

if __name__ == "__main__":
    profile_run(limit=1)
