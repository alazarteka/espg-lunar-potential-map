# ESPG Lunar Potential Map

> **Note:** `CLAUDE.md` is a symlink to this file. When modifying or committing, target `AGENTS.md` directly.

## Project Overview

This project maps the lunar surface electrostatic potential using physics-based models. It processes SPICE kernel data, applies temporal reconstruction techniques, and generates potential maps accounting for plasma interactions and surface properties.

**Key capabilities:**
- SPICE kernel data acquisition and processing
- Flux calculations for plasma-surface interactions
- Kappa distribution modeling for particle populations
- Temporal coefficient reconstruction for time-varying phenomena
- Electrostatic potential mapping with configurable physics models
- Loss cone analysis and secondary electron beam detection

## Project Structure & Module Organization

- `src/`: Core Python package (e.g., `config.py`, `data_acquisition.py`, `flux.py`, `kappa.py`, `potential_mapper/`, plus `physics/` and `utils/`).
- `src/diagnostics/`: Loss cone session management and analysis tools.
- `tests/`: Pytest suite (`test_*.py`, with scopes like `physics/`, `utils/`).
- `data/`: Runtime data; `data/spice_kernels/` populated by acquisition; commit only small metadata (e.g., `kernels.lock`).
- `artifacts/`: Generated outputs (plots, reports, potential caches in `artifacts/potential_cache`); keep runtime inputs in `data/`.
- `scratch/`: Ad-hoc experiments, perf runs, and attachments; gitignored by default.
- `scripts/`: One-off helpers
  - `analysis/`: plotting & exploration CLIs intended for sharing
  - `profiling/`: performance measurement scripts
  - `diagnostics/`: data quality and beam detection tools
  - `dev/`: ad-hoc diagnostics, quick experiments
- `notebooks/`, `docs/`: Research notes and outputs (avoid committing large binaries).

## Build, Test, and Development Commands

Use UV with Python 3.12.

- Install/upgrade deps: `uv sync` / `uv sync --upgrade`
- Install dev / test deps: `uv sync --locked --all-extras --dev
- Run modules: `uv run python -m src.potential_mapper`
- Acquire data: `uv run python -m src.data_acquisition`
- Tests: `uv run pytest -q` (coverage: `uv run pytest --cov=src`)
- Lint: `uv run ruff check .`
- Format: `uv run ruff format src tests`
- Type check: `uv run mypy src`
- Pre-commit: `pre-commit run --all-files`

## GPU Acceleration

This project has GPU-accelerated paths for compute-intensive operations. **Always check script flags and use GPU acceleration when available.**

Key flags:
- `--fast`: Enable PyTorch GPU acceleration (auto-detects dtype and batch size)
- Requires GPU extra: `uv sync --extra gpu`

Examples:
```bash
# Batch potential mapping (ALWAYS use --fast on GPU machines)
uv run python -m src.potential_mapper.batch --fast --year 1998 --month 4

# GPU batch size sweep for tuning
uv run python scripts/profiling/gpu_batch_sweep.py
```

The GPU path auto-detects:
- **dtype**: float16 on modern GPUs (Volta+), float32 on older GPUs/CPU
- **batch_size**: Calculated from available VRAM

## Coding Style & Naming Conventions

- Python 3.12, 4-space indent, Ruff line length 88 (`pyproject.toml`).
- Ruff rules: `E,F,W,B,I,UP,SIM`; keep imports sorted; prefer explicit exports.
- Type hints required where practical; `mypy` configured with `pint` plugin.
- Names: modules/functions `snake_case`; classes `PascalCase`; constants `UPPER_SNAKE_CASE` (see `src/config.py").
- Units: use `src.utils.units` quantities for physical values.

## Testing Guidelines

- Framework: Pytest; discovery set to `tests`, files `test_*.py`, classes `Test*`, functions `test_*`.
- Mark slow/data-dependent tests with `@pytest.mark.skip_ci`.
- Prefer synthetic fixtures (see `tests/test_kappa_fitter.py") to avoid downloads.

## Commit & Pull Request Guidelines

- Commit messages: Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `style:`, `test:`, `chore:`) with optional scope, e.g., `feat(physics): ...`.
- PRs must include: clear description, linked issues, rationale, before/after plots if applicable, and reproduction steps.
- Pre-merge checklist: `ruff`, `mypy`, `pytest` all pass; update README/docs if behavior or interfaces change; do not commit large data or notebook outputs.

## Security & Configuration Tips (Optional)

- No secrets needed; configuration lives in `src/config.py` (`DATA_DIR`, `KERNELS_DIR`).
- Use `data_acquisition` to populate SPICE kernels; track hashes with `data/spice_kernels/kernels.lock`.

## Diagnostics Tools

The `scripts/diagnostics/` directory contains tools for data quality analysis:

### Beam Detection

Secondary electron beams appear as excess flux at high pitch angles (150-180°) in loss-cone normalized data. The beam energy corresponds to the potential difference ΔU = U_spacecraft - U_surface.

**Key scripts:**
- `losscone_peak_scan.py`: Scan ER files for beam signatures
- `view_norm2d.py`: Visualize normalized flux grids for specific sweeps
- `beam_detection_survey.py`: Sample files across the dataset to measure detection rates

**Detection thresholds** (tuned to filter noise while keeping real beams):
- `min_peak=2.0`: Peak normalized flux must exceed 2.0
- `min_neighbor=1.5`: At least one adjacent energy bin must exceed 1.5
- `min_band_points=5`: At least 5 valid pitch bins in the 150-180° band

**Survey results** (as of 2026-01):
- Overall beam detection rate: ~23% of valid sweeps
- Range by month: 14-35%
- Higher rates observed in May 1998, May 1999

### Usage Examples

```bash
# Scan a single file for beams
uv run python scripts/diagnostics/losscone_peak_scan.py data/1999/091_120APR/3D990429.TAB --list-peaks

# Visualize a specific sweep
uv run python scripts/diagnostics/view_norm2d.py data/1999/091_120APR/3D990429.TAB --spec-no 653 --show-band

# Survey detection rates across the dataset
uv run python scripts/diagnostics/beam_detection_survey.py --samples-per-month 4
```

## Agent Behavior Guidelines

When working on this project, AI agents should:

1. **Proactively update documentation**: After implementing new features, scripts, or significant changes, update this file (AGENTS.md) and relevant docs to reflect the changes. Don't wait to be asked.

2. **Document empirical findings**: When running analyses that produce useful statistics or insights (e.g., detection rates, parameter tuning results), add them to the relevant section of this file for future reference.

3. **Keep the project structure section current**: When adding new directories or reorganizing code, update the "Project Structure & Module Organization" section.

4. **Add usage examples**: When creating new scripts or tools, include example commands in the documentation.

5. **Record tuning decisions**: When parameters are tuned based on experimentation (like beam detection thresholds), document the reasoning and results.
