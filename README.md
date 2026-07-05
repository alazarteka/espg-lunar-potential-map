# Lunar Prospector Plasma Analysis

## Manuscript provenance release

Tag `lp-er-derived-data-v1.0.0` identifies the research-code state used as
provenance for generating the derived LP ER spacecraft-relative potential cache
archived with the manuscript "Identifiability and Sampling Limits on
Spacecraft-Relative Lunar Surface Potential from Lunar Prospector Electron
Reflectometer Observations."

This repository is research/provenance code, not the polished manuscript
reproduction package. The manuscript source, figure-regeneration scripts, and
final figure inputs are archived separately in
[`lp-er-lunar-potential`](https://github.com/alazarteka/lp-er-lunar-potential).

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
└── src/                    # Core Python package (see AGENTS.md for the full map)
    ├── config.py           # Central configuration constants
    ├── data_acquisition.py # NASA/PDS data + SPICE kernel download
    ├── flux.py, kappa.py, model.py, spacecraft_potential.py  # core fitting
    ├── losscone/           # Loss-cone fitting core (cpu, model, chi2, masks, torch/)
    ├── potential_mapper/   # Mapping pipeline + GPU batch runner
    ├── temporal/           # Spherical-harmonic temporal reconstruction
    ├── engineering/        # GlobalStats, SiteStats, site analysis
    ├── diagnostics/        # Loss-cone session management
    ├── physics/            # Kappa physics, spacecraft charging, J–U curves
    ├── utils/              # SPICE, geometry, attitude, units helpers
    └── visualization/      # Plot styling and loaders
```

Generated plots and reports now live under `artifacts/`; use `scratch/` for ad-hoc runs
and exploratory outputs that should stay out of version control. See
[docs/README.md](docs/README.md) for the full documentation index.

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
    # Detects your GPU and syncs the right PyTorch build automatically:
    #   sm_70+ (RTX 20xx and newer) -> modern CUDA 12.8 stack (default)
    #   sm_61  (GTX 10xx / TITAN Xp) -> legacy CUDA 11.8 stack
    ./scripts/bootstrap.sh

    # Force a specific stack (e.g. on a GPU-less build node):
    ./scripts/bootstrap.sh modern
    ./scripts/bootstrap.sh legacy
    ```

    This detects the GPU architecture, copies the matching lockfile from
    `locks/` into `uv.lock`, and runs `uv sync --frozen` with the right extra.

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

The batch runner (`python -m src.potential_mapper.batch`) processes ER files and
writes a single NPZ cache per run under `artifacts/potential_cache/`, containing
row-aligned spacecraft and surface potentials plus per-spectrum rollups.

```bash
# Compute a day of caches on the GPU (writes potential_batch_1998_09_16.npz)
uv run python -m src.potential_mapper.batch --fast --year 1998 --month 9 --day 16
```

Key options:
- `--year/--month/--day`: filter ER files by date (month and day optional).
- `--fast`: use the GPU (torch) fitter; `--parallel` selects the CPU multi-process path.
- `--losscone-fit-method {halekas,lillis}`, `--u-spacecraft`: fitting overrides.
- `--output-dir`: target directory for the NPZ artefact; `--overwrite` to recompute.

See [docs/cli/potential_mapper_batch.md](docs/cli/potential_mapper_batch.md) for the
full option reference.

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
path = Path('artifacts/potential_cache/potential_batch_1998_09_16.npz')
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
