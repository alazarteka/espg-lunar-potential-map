# Lunar Prospector Surface-Potential Mapping

Invert the magnetic loss cone in Lunar Prospector electron-reflectometer (ER)
spectra to estimate the Moon's surface electrostatic potential — and map the
limits of what that data can actually constrain. The central result is a
**negative one**: a global spatiotemporal potential map is *not* recoverable from
these observations.

[![CI](https://github.com/alazarteka/espg-lunar-potential-map/actions/workflows/ci.yml/badge.svg)](https://github.com/alazarteka/espg-lunar-potential-map/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20623399-blue.svg)](https://doi.org/10.5281/zenodo.20623399)

> **Research / provenance code.** This is the analysis code behind the manuscript
> *"Identifiability and Sampling Limits on Spacecraft-Relative Lunar Surface
> Potential from Lunar Prospector Electron Reflectometer Observations."* It is
> not a polished reproduction package — see [Citation](#citation) for the
> archived manuscript inputs and DOIs.

## What it does

The lunar surface charges to an electrostatic potential set by its local plasma
and photoelectron environment. That potential leaves a fingerprint in the
electrons Lunar Prospector measured: a magnetic **loss cone** whose shape depends
on how strongly returning electrons were reflected versus absorbed or accelerated
at the surface (Halekas et al., 2008). This project inverts that signal per
spectrum. The manuscript's central conclusion is that those per-spectrum
estimates **cannot be assembled into a map**: the surface potential changes over
time, while each epoch illuminates only a small patch of the Moon — so recovering
it as a field means reconstructing joint variation in *both* space and time from
data that cannot constrain that joint variation. That reconstruction is not
identifiable. What the data *do* support is statistics over the measurements
themselves (aggregate and per-site distributions), not a coherent global map.

## How it works

The pipeline runs in stages, each backed by a `src/` subpackage:

| Stage | What happens | Module |
|-------|--------------|--------|
| **1. Acquire** | Download LP ER 3D electron flux, SPICE kernels, and attitude tables from NASA PDS/NAIF | `src.data_acquisition` |
| **2. Fit** | Per-spectrum loss-cone inversion → surface potential `U_surface` (Halekas log-χ² or Lillis masked-χ²) | `src.losscone` |
| **3. Correct** | Estimate the spacecraft's own floating potential from a κ-distribution charging balance and debias the fits | `src.spacecraft_potential`, `src.physics` |
| **4. Project** | Trace each measurement's magnetic footprint to lunar-surface coordinates via SPICE geometry | `src.potential_mapper` |
| **5. Aggregate** | Batch the fits into per-run NPZ caches (row- and spectrum-level) | `src.potential_mapper.batch` |
| **6. Assess** | Statistics over the measurements (aggregate + per-site) plus the joint spatiotemporal reconstruction that the paper shows is *not* identifiable | `src.engineering`, `src.temporal` |

See [docs/architecture/pipeline_overview.md](docs/architecture/pipeline_overview.md)
for the end-to-end detail and [coordinate_frames.md](docs/architecture/coordinate_frames.md)
for the SPICE/frame conventions.

## Getting started

**Prerequisites:** Python 3.12 (3.13 is not yet supported) and the
[uv](https://docs.astral.sh/uv/) package manager
(`curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
# 1. Clone and install
git clone https://github.com/alazarteka/espg-lunar-potential-map.git
cd espg-lunar-potential-map
uv sync                              # core deps (add --group dev --extra notebook as needed)

# 2. GPU acceleration (optional but recommended) — auto-detects your GPU
./scripts/bootstrap.sh               # sm_70+ -> CUDA 12.8; sm_61 (GTX 10xx) -> CUDA 11.8

# 3. Download the data (~13 GB; parallel, resumable)
uv run python -m src.data_acquisition

# 4. Map a day of observations
uv run python -m src.potential_mapper.batch --fast --year 1998 --month 9 --day 16
```

That writes `artifacts/potential_cache/potential_batch_1998_09_16.npz`, ready to
visualise (below).

`bootstrap.sh` picks the matching lockfile from `locks/` and runs
`uv sync --frozen`; pass `modern` / `legacy` to override detection. See
[docs/dev/lockfiles.md](docs/dev/lockfiles.md) for the GPU-stack details and
[docs/cli/data_acquisition.md](docs/cli/data_acquisition.md) for download options.

## Usage

### Batch potential-cache generation

`python -m src.potential_mapper.batch` fits ER files and writes one NPZ cache per
run under `artifacts/potential_cache/`, with row-aligned spacecraft and surface
potentials plus per-spectrum rollups.

```bash
uv run python -m src.potential_mapper.batch --fast --year 1998 --month 9 --day 16
```

Key options: `--year/--month/--day` (month/day optional) · `--fast` (GPU fitter;
`--parallel` for the CPU path) · `--losscone-fit-method {halekas,lillis}` ·
`--u-spacecraft` · `--output-dir` · `--overwrite`. Full reference:
[docs/cli/potential_mapper_batch.md](docs/cli/potential_mapper_batch.md).

### Visualising cached potentials

```bash
# Static Matplotlib sphere projection
uv run python scripts/analysis/potential_map_matplotlib_sphere.py \
  --start 1998-09-16 --end 1998-09-16 \
  --cache-dir artifacts/potential_cache --sample 5000 \
  --output artifacts/plots/1998-09-16.png

# Interactive Plotly globe, with optional animation / MP4 export
uv run python scripts/analysis/potential_map_plotly_static.py \
  --start 1998-09-16 --end 1998-09-16 \
  --cache-dir artifacts/potential_cache --sample 5000 \
  --output artifacts/plots/1998-09-16.html --animate
```

MP4 export needs Kaleido plus `imageio[ffmpeg]` (the `export` extra) and a
Chrome/Chromium binary on headless systems. More plotting and analysis scripts
are indexed in [docs/cli/analysis.md](docs/cli/analysis.md).

### Inspecting a cache

```bash
uv run python - <<'PY'
import numpy as np
from pathlib import Path
with np.load(Path("artifacts/potential_cache/potential_batch_1998_09_16.npz")) as d:
    sc, proj = d["rows_spacecraft_potential"], d["rows_projected_potential"]
    print(f"rows={sc.size}")
    print(f"spacecraft phi: {np.nanmin(sc):.1f} -> {np.nanmax(sc):.1f} V")
    print(f"surface phi samples: {np.isfinite(proj).sum()} / {proj.size}")
PY
```

## Documentation

Full CLI and design docs live under [`docs/`](docs/README.md):

| Topic | Doc |
|-------|-----|
| Data acquisition | [cli/data_acquisition.md](docs/cli/data_acquisition.md) |
| Potential mapper (single day / batch) | [cli/potential_mapper.md](docs/cli/potential_mapper.md) · [cli/potential_mapper_batch.md](docs/cli/potential_mapper_batch.md) |
| Temporal harmonics | [cli/temporal.md](docs/cli/temporal.md) |
| Engineering products | [cli/engineering.md](docs/cli/engineering.md) |
| Diagnostics (beam / loss-cone) | [cli/diagnostics.md](docs/cli/diagnostics.md) |
| Analysis & plotting scripts | [cli/analysis.md](docs/cli/analysis.md) |
| Architecture, frames, GPU | [architecture/](docs/architecture/) |

## Project layout

```
src/               Core package (see AGENTS.md for the full module map)
  losscone/          Loss-cone inversion (CPU + torch backends)
  potential_mapper/  Geometry projection, batch runner, NPZ output
  temporal/          Spherical-harmonic temporal reconstruction
  engineering/       Site statistics and engineering maps
  physics/           κ-distribution physics, spacecraft charging
scripts/           CLI tools: diagnostics/, analysis/, plots/, profiling/, dev/
locks/             Tracked CUDA 11/12 lockfiles (copied to uv.lock by bootstrap.sh)
docs/              CLI and architecture documentation
data/              Downloaded ER data + SPICE kernels (gitignored)
artifacts/         Generated caches, plots, reports (gitignored)
```

## Development

```bash
uv sync --group dev
uv run pytest -q                       # tests
uv run ruff check src tests            # lint
uv run ruff format --check src tests   # format check (a separate CI gate)
```

Install the git hooks so these run automatically on commit:

```bash
uv tool install pre-commit && pre-commit install
```

See [docs/dev/development.md](docs/dev/development.md) and [AGENTS.md](AGENTS.md)
for the full contributor workflow and coding standards. (`mypy` is advisory —
there is a known type-hygiene backlog it is not yet a hard gate.)

## Citation

If you use this code, please cite the manuscript and the archived software/data:

- **Manuscript:** Gemechu, A. T., & Kim, H. J. *Identifiability and Sampling
  Limits on Spacecraft-Relative Lunar Surface Potential from Lunar Prospector
  Electron Reflectometer Observations.*
- **Software:** [10.5281/zenodo.20623399](https://doi.org/10.5281/zenodo.20623399)
- **Derived dataset:** [10.5281/zenodo.20623229](https://doi.org/10.5281/zenodo.20623229)

The manuscript source, figure-regeneration scripts, and final figure inputs are
archived separately in
[`lp-er-lunar-potential`](https://github.com/alazarteka/lp-er-lunar-potential).
The tag `lp-er-derived-data-v1.0.0` marks the code state used to generate the
archived derived-potential cache.

## License

[MIT](LICENSE) © Alazar Teka Gemechu, Hyun Jung Kim (KAIST).
