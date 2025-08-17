import argparse
import logging

import src.config as config
from src.potential_mapper.spice import load_spice_files
from src.potential_mapper.pipeline import run

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="This tool maps the surface potential of the Moon using data from the Lunar Prospector mission.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Year to process (one of 1998, 1999)"
    )

    parser.add_argument(
        "--month",
        type=int,
        default=None,
        help="Month to process (one of 1-12)"
    )

    parser.add_argument(
        "--day",
        type=int,
        default=None,
        help="Day to process (one of 1-31)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the output file"
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output"
    )

    parser.add_argument(
        '-d',
        '--display',
        action='store_true',
        default=False,
        help="Show the plot"
    )

    return parser.parse_args()

def main():
    args = parse_arguments()

    LOG_LEVEL = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Load SPICE kernels once up front
    load_spice_files()

    # Run the pipeline with provided arguments
    run(args)


if __name__ == "__main__":
    main()
