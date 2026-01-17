# Lunar Prospector Plasma Analysis

## Description

This project analyzes data from the Lunar Prospector mission to model plasma flux and map lunar surface potential. It utilizes data from the NASA Planetary Data System and SPICE kernels for trajectory and orientation information.

Currently in active development—major revisions ongoing.

## Dependencies

This project uses modern Python scientific computing libraries:

- **Core Scientific:** `numpy`, `pandas`, `scipy`, `matplotlib`, `numba`
- **Space Science:** `spiceypy` (NASA SPICE toolkit)
- **Data Processing:** `requests`, `beautifulsoup4`, `tqdm`, `pint`
- **Visualization:** `plotly`, `tabulate`

Optional dependencies:
- **Notebooks:** `jupyter`, `ipykernel`, `ipython` (install with `--extra notebook`)
- **Export:** `imageio[ffmpeg]`, `kaleido` (install with `--extra export`)
- **GPU Acceleration:** `torch` (install with `--extra gpu` or `--extra gpu-legacy`)

Development tools:
- **Testing:** `pytest`, `pytest-cov`
- **Linting/Formatting:** `ruff`
- **Type Checking:** `mypy`

All dependencies are managed through [UV](https://docs.astral.sh/uv/) and specified in `pyproject.toml`.

## Project Structure

```
.
├── README.md               # This file
├── pyproject.toml          # UV project configuration and dependencies
├── locks/                  # Tracked lockfiles for GPU variants
│   ├── uv.lock.cuda11       # Legacy GPU (CUDA 11.8)
│   └── uv.lock.cuda12       # Modern GPU (CUDA 12.x)
├── uv.lock                 # Machine-local lockfile (gitignored; copied from locks/)
├── .gitignore              # Git ignore file
├── artifacts/              # Generated outputs (plots, reports, caches)
│   ├── plots/              # Figures, animations, HTML exports
│   ├── reports/            # Generated analysis summaries
│   └── potential_cache/    # Default output for NPZ caches (was data/potential_cache)
├── scratch/                # Ad-hoc scratch space for experiments (gitignored)
├── data/                   # Processed data (created/populated by src/data_acquisition.py)
│   └── spice_kernels/      # SPICE kernel files
│       └── kernels.lock    # SHA-1 for SPICE kernels
└── src/                    # Source code
    ├── config.py           # Configuration file
    ├── data_acquisition.py # Handles data and SPICE kernel acquisition
    ├── flux.py             # ER flux + loss-cone fitting
    ├── kappa.py            # Kappa distribution fitting utilities
    ├── kappa_torch.py      # PyTorch-accelerated Kappa fitter
    ├── model.py            # Core modeling components
    ├── model_torch.py      # PyTorch-accelerated loss-cone model
    ├── potential_mapper/   # Modular potential mapping package
    │   ├── __init__.py
    │   ├── __main__.py     # Enables `python -m src.potential_mapper`
    │   ├── cli.py          # CLI argument parsing and entrypoint
    │   ├── pipeline.py     # File discovery, processing, orchestration
    │   ├── coordinates.py  # Frame transforms, projections, intersections
    │   ├── plot.py         # Plotting utilities
    │   └── results.py      # Typed results container
    └── utils/              # Utility functions module
        ├── __init__.py
        ├── attitude.py
        ├── coordinates.py
        ├── file_ops.py
        ├── geometry.py
        ├── optimization.py # Batched differential evolution optimizer
        └── spice_ops.py
```

Generated plots and reports now live under `artifacts/`; use `scratch/` for ad-hoc runs
and exploratory outputs that should stay out of version control.

## Installation

### Prerequisites
- Python 3.12 (required; 3.13 is not supported yet)
- [UV package manager](https://docs.astral.sh/uv/)

### Setup

1.  **Install UV** (if not already installed):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Clone and set up the project:**
    ```bash
    git clone https://github.com/alazarteka/espg-lunar-potential-map.git
    cd espg-lunar-potential-map

    # Install core dependencies
    uv sync

    # Or install with development tools
    uv sync --group dev

    # Or install with notebook support
    uv sync --extra notebook

    # Or install everything (dev + notebooks + export)
    uv sync --group dev --extra notebook --extra export
    ```

3.  **GPU acceleration (optional, recommended workflow):**
    ```bash
    # Modern GPU (RTX 20xx+, CUDA 12.x)
    ./scripts/select-env.sh cuda12

    # Legacy GPU (GTX 10xx, CUDA 11.8)
    ./scripts/select-env.sh cuda11
    ```

    This copies the appropriate lockfile from `locks/` into `uv.lock` and runs
    `uv sync --frozen` with the matching extra.

4.  **Download necessary data:**
    ```bash
    uv run python -m src.data_acquisition
    ```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src

# Run tests excluding slow/CI-skipped tests
uv run pytest -m "not skip_ci and not slow"

# Run specific test file
uv run pytest tests/test_model_torch.py -v
```

### Linting and Formatting

```bash
# Check for lint errors
uv run ruff check src tests

# Auto-fix fixable issues
uv run ruff check src tests --fix

# Check formatting
uv run ruff format --check src tests

# Apply formatting
uv run ruff format src tests
```

### Type Checking

```bash
uv run mypy src
```

### Adding Dependencies

```bash
# Add a runtime dependency
uv add package-name

# Add to a dependency group (dev tools)
uv add --group dev package-name

# Add to an optional extra
uv add --optional notebook package-name

# Update all dependencies
uv sync --upgrade
```

## Data

*   Data required by the core modules are expected in `data/`.
*   SPICE kernels are expected in `data/spice_kernels/`.
*   Running `src/data_acquisition.py` will download the files in the appropriate directory if they are not locally available.
*   SHA-1 for SPICE kernels used in `data/spice_kernels/kernels.lock`, generated by running ```shasum -a 1 data/spice_kernels/*.* > data/spice_kernels/kernels.lock``` in the home directory.

## Usage

After setting up and downloading data, run the desired module using UV. For example:

```bash
# Run the potential mapper
uv run python -m src.potential_mapper

# Start Jupyter Lab for interactive analysis (requires --extra notebook)
uv run jupyter lab

# Start Jupyter Notebook
uv run jupyter notebook

# Run tests
uv run pytest
```

### Batch potential cache generation

The batch runner in `scripts/dev/potential_mapper_batch.py` processes ER files and
writes per-file NPZ caches under `artifacts/potential_cache/`. Each cache contains
row-aligned spacecraft and surface potentials along with spectrum rollups.

```bash
# Compute caches for a single day (writes to artifacts/potential_cache by default)
uv run python scripts/dev/potential_mapper_batch.py \
  --year 1998 --month 9 --day 16 --limit 1 --workers 1
```

Key options:
- `--year/--month/--day`: filter ER files by tokenised date.
- `--limit`: stop after processing the first *N* files (useful for quick checks).
- `--output-dir`: target directory for NPZ artefacts.

The command loads SPICE kernels as needed and is CPU-heavy; consider reducing the
sample size (via `--limit`) when iterating locally.

### Visualising cached potentials

Two analysis CLIs load the cached NPZ artefacts for plotting:

```bash
# Static Matplotlib sphere projection
uv run python scripts/analysis/potential_map_matplotlib_sphere.py \
  --start 1998-09-16 --end 1998-09-16 \
  --cache-dir artifacts/potential_cache --sample 5000 --output artifacts/plots/1998-09-16.png

# Interactive Plotly globe with optional animation/MP4 export
uv run python scripts/analysis/potential_map_plotly_static.py \
  --start 1998-09-16 --end 1998-09-16 \
  --cache-dir artifacts/potential_cache --sample 5000 \
  --output artifacts/plots/1998-09-16.html --animate
```

For large animations, prefer modest sampling (`--sample`) and frame counts. MP4
export requires Kaleido plus `imageio[ffmpeg]`; ensure Chrome/Chromium is
available for Kaleido on headless systems.

### Validating spacecraft potential outputs

Cached files expose `rows_spacecraft_potential` and
`rows_projected_potential`. Inspect a representative NPZ to confirm the pipeline
is debiasing the loss-cone fits with the spacecraft charging term:

```bash
uv run python - <<'PY'
import numpy as np
from pathlib import Path
path = Path('artifacts/potential_cache/1998/244_273SEP/3D980916.npz')
with np.load(path) as data:
    sc = data['rows_spacecraft_potential']
    proj = data['rows_projected_potential']
    print(f"rows={sc.size}")
    print(f"spacecraft phi range: {np.nanmin(sc):.2f} -> {np.nanmax(sc):.2f} V")
    finite_proj = np.isfinite(proj)
    print(f"surface phi samples: {finite_proj.sum()} / {proj.size}")
PY
```

The example above reports approximately -74 V to +39 V spacecraft potentials and
strong negative surface potentials for the September 16, 1998 pass, indicating
the median spacecraft bias is being removed before fitting ΔU.
