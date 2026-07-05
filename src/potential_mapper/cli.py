import argparse

from src.potential_mapper.cli_args import add_common_batch_args
from src.potential_mapper.logging_utils import setup_logging
from src.potential_mapper.pipeline import run
from src.potential_mapper.spice import load_spice_files


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the potential mapping CLI.

    Flags cover basic date filters (single year/month/day), plotting output,
    verbosity, and an optional illumination filter (day/night) for display.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.potential_mapper",
        description=(
            "This tool maps the surface potential of the Moon using data from "
            "the Lunar Prospector mission."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_common_batch_args(parser, include_overwrite=False)

    parser.add_argument(
        "--output", type=str, default=None, help="Path to the output file"
    )

    parser.add_argument(
        "-d", "--display", action="store_true", default=False, help="Show the plot"
    )

    parser.add_argument(
        "--illumination",
        choices=["day", "night"],
        default=None,
        help="Filter plotted points by illumination: 'day', 'night', or omit for all",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    setup_logging(args.verbose)

    # Load SPICE kernels once up front
    load_spice_files()

    # Run the pipeline with provided arguments and propagate its exit code
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
