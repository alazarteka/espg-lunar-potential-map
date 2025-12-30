import argparse
import logging
from pathlib import Path

from src.potential_mapper import plot as plot_mod
from src.potential_mapper.pipeline import (
    DataLoader,
    _spacecraft_potential_per_row,
    process_lp_file,
)
from src.potential_mapper.results import PotentialResults, _concat_results

__all__ = [
    "DataLoader",
    "_concat_results",
    "_spacecraft_potential_per_row",
    "process_lp_file",
    "run",
]


def run(args: argparse.Namespace) -> int:
    """Entry point for CLI to orchestrate processing and optional plotting."""
    flux_files = DataLoader.discover_flux_files(
        year=args.year,
        month=args.month,
        day=args.day,
    )

    if not flux_files:
        logging.info("No ER flux files discovered. Exiting.")
        return 0

    results: list[PotentialResults] = []
    for fp in flux_files:
        try:
            logging.debug(f"Processing {fp}")
            res = process_lp_file(fp)
            results.append(res)
        except Exception as e:
            logging.exception(f"Failed to process {fp}: {e}")

    if not results:
        logging.warning("All files failed to process; nothing to plot or save.")
        return 1

    # Aggregate results across files
    agg = _concat_results(results) if len(results) > 1 else results[0]

    if args.output or args.display:
        fig, _ax = plot_mod.plot_map(agg, illumination=args.illumination)
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=200)
            logging.info(f"Saved plot to {out_path}")
        if args.display:
            try:
                import matplotlib.pyplot as plt

                plt.show()
            except Exception:
                logging.warning("Display requested but matplotlib backend failed.")

    return 0
