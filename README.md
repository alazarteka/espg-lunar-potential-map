# Lunar Prospector Plasma Analysis

## Description

This project analyzes data from the Lunar Prospector mission to model plasma flux and map lunar surface potential. It utilizes data from the NASA Planetary Data System and SPICE kernels for trajectory and orientation information.

Currently in active development—major revisions ongoing.

## Dependencies

This project uses modern Python scientific computing libraries:

- **Core Scientific:** `numpy`, `pandas`, `scipy`, `matplotlib`
- **Space Science:** `spiceypy` (NASA SPICE toolkit)
- **Data Processing:** `requests`, `beautifulsoup4`, `tqdm`
- **Interactive Analysis:** `jupyter`, `plotly`, `ipython`
- **Development:** `pytest`, `black`, `ruff`, `mypy`

All dependencies are managed through UV and specified in `pyproject.toml`.

## Project Structure

```
.
├── README.md               # This file
├── pyproject.toml          # UV project configuration and dependencies
├── uv.lock                 # Dependency lock file
├── .gitignore              # Git ignore file
├── artifacts/              # Generated outputs (plots, reports, caches)
│   ├── plots/              # Figures, animations, HTML exports
│   ├── reports/            # Generated analysis summaries
│   └── potential_cache/    # Default output for NPZ caches
├── scratch/                # Ad-hoc scratch space for experiments (gitignored)
├── data/                   # Processed data (created/populated by src/data_acquisition.py)
│   └── spice_kernels/      # SPICE kernel files
│       └── kernels.lock    # SHA-1 for SPICE kernels
└── src/                    # Source code
    ├── config.py           # Configuration file
    ├── data_acquisition.py # Handles data and SPICE kernel acquisition
    ├── flux.py             # ER flux + loss-cone fitting
    ├── kappa.py            # Kappa distribution fitting utilities
    ├── model.py            # Core modeling components
    ├── spacecraft_potential.py # Spacecraft charging estimation
    ├── physics/            # Physics calculations
    │   ├── charging.py     # Current density calculations
    │   ├── jucurve.py      # J-U curve logic
    │   └── kappa.py        # Kappa distribution math
    ├── potential_mapper/   # Modular potential mapping package
    │   ├── __init__.py
    │   ├── __main__.py     # Enables `python -m src.potential_mapper`
    │   ├── cli.py          # CLI argument parsing and entrypoint
    │   ├── pipeline.py     # File discovery, processing, orchestration
    │   ├── batch.py        # Batch processing
    │   ├── coordinates.py  # Frame transforms, projections, intersections
    │   ├── plot.py         # Plotting utilities
    │   ├── results.py      # Typed results container
    │   └── spice.py        # SPICE kernel loading
    ├── temporal/           # Time-dependent spherical harmonic analysis
    │   ├── basis.py        # Temporal basis functions
    │   ├── coefficients.py # Harmonic coefficient fitting
    │   ├── dataset.py      # Data loading/saving
    │   └── reconstruction.py # Global map reconstruction
    └── utils/              # Utility functions module
        ├── __init__.py
        ├── attitude.py     # Attitude data handling
        ├── coordinates.py  # Coordinate transformations
        ├── energy.py       # Energy binning
        ├── file_ops.py     # File system operations
        ├── flux_files.py   # Flux file selection
        ├── geometry.py     # Ray-sphere intersections
        ├── spice_lock.py   # SPICE kernel locking
        ├── spice_ops.py    # SPICE operations
        ├── synthetic.py    # Synthetic data generation
        └── units.py        # Unit definitions
```

Generated plots and reports now live under `artifacts/`; use `scratch/` for ad-hoc runs
and exploratory outputs that should stay out of version control.

## Installation

### Prerequisites
- Python 3.12 or later
- [UV package manager](https://docs.astral.sh/uv/)

### Setup

1.  **Install UV** (if not already installed):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Set up the project environment:**
    ```bash
    # Clone the repository (if not already done)
    git clone <repository-url>
    cd espg-lunar-potential-map
    
    # Install dependencies and create virtual environment
    uv sync
    ```

3.  **Download necessary data:**
    ```bash
    uv run python -m src.data_acquisition
    ```

## Documentation

The codebase is documented using Google Style Python Docstrings. Each public function, class, and method includes a docstring describing its purpose, arguments, and return values.

To explore the documentation, you can use Python's built-in `help()` function or read the source code directly.

## Data

*   Data required by the core modules are expected in `data/`.
*   SPICE kernels are expected in `data/spice_kernels/`.
*   Running `src/data_acquisition.py` will download the files in the appropriate directory if they are not locally available.
*   SHA-1 for SPICE kernels used in `data/spice_kernels/kernels.lock`, generated by running ```shasum -a 1 data/spice_kernels/*.* > data/spice_kernels/kernels.lock``` in the home directory.

## Usage

After setting up and downloading data, run the desired module using UV.

### Potential Mapper

Run the potential mapper to process ER data and generate surface potential maps:

```bash
uv run python -m src.potential_mapper
```

### Temporal Analysis

Compute time-dependent spherical harmonic coefficients:

```bash
uv run python -m src.temporal --start 1998-01-01 --end 1999-01-01 --output artifacts/temporal_coeffs.npz
```

### Interactive Analysis

Start Jupyter Lab or Notebook for interactive analysis:

```bash
uv run jupyter lab
# or
uv run jupyter notebook
```

### Testing

Run the test suite to ensure everything is working correctly:

```bash
uv run pytest
```

### Batch potential cache generation

The batch runner in `src/potential_mapper/batch.py` processes ER files and
writes per-file NPZ caches under `artifacts/potential_cache/`. Each cache contains
row-aligned spacecraft and surface potentials along with spectrum rollups.

```bash
# Compute caches for a single day (writes to artifacts/potential_cache by default)
uv run python -m src.potential_mapper.batch \
  --year 1998 --month 9 --day 16 --overwrite
```

Key options:
- `--year/--month/--day`: filter ER files by date.
- `--output-dir`: target directory for NPZ artifacts.
- `--parallel`: enable parallel processing.

### Visualising cached potentials

Scripts in `scripts/analysis/` can be used to visualize the output.

## Development

```bash
# Add new dependencies
uv add package-name

# Add development dependencies
uv add --dev pytest black ruff

# Update dependencies
uv sync --upgrade

# Run with specific Python version
uv run --python 3.12 python script.py
```
