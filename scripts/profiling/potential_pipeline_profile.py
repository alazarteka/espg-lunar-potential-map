"""Profile the potential mapping pipeline CLI end-to-end."""

import cProfile
import pstats
import time
from io import StringIO

import src.config as config
from src.potential_mapper.cli import parse_arguments
from src.potential_mapper.pipeline import run
from src.potential_mapper.spice import load_spice_files


def main() -> None:
    """
    Profile the potential mapping pipeline end-to-end.

    Mirrors the profiling style of existing scripts, dumping a .prof file to scratch
    and printing the top cumulative-time functions.
    """
    args = parse_arguments()

    # Ensure SPICE is loaded similarly to the CLI entry
    load_spice_files()

    pr = cProfile.Profile()
    pr.enable()
    t0 = time.time()

    exit_code = run(args)

    dt = time.time() - t0
    pr.disable()

    out_path = config.PROJECT_ROOT / "scratch" / "profiles" / "potential_mapping_profile.prof"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pr.dump_stats(out_path)

    s = StringIO()
    ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumulative")
    ps.print_stats(40)

    print(f"Pipeline exit code: {exit_code}")
    print(f"Elapsed: {dt:.2f} s")
    print("\nTop functions by cumulative time:")
    print(s.getvalue())


if __name__ == "__main__":
    main()
