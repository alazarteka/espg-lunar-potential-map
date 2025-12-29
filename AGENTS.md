# ESPG Lunar Potential Map

## Project Overview

This project maps the lunar surface electrostatic potential using physics-based models. It processes SPICE kernel data, applies temporal reconstruction techniques, and generates potential maps accounting for plasma interactions and surface properties.

**Key capabilities:**
- SPICE kernel data acquisition and processing
- Flux calculations for plasma-surface interactions
- Kappa distribution modeling for particle populations
- Temporal coefficient reconstruction for time-varying phenomena
- Electrostatic potential mapping with configurable physics models

## Project Structure & Module Organization

- `src/`: Core Python package (e.g., `config.py`, `data_acquisition.py`, `flux.py`, `kappa.py`, `potential_mapper/`, plus `physics/` and `utils/`).
- `tests/`: Pytest suite (`test_*.py`, with scopes like `physics/`, `utils/").
- `data/`: Runtime data; `data/spice_kernels/` populated by acquisition; commit only small metadata (e.g., `kernels.lock").
- `artifacts/`: Generated outputs (plots, reports, potential caches in `artifacts/potential_cache`); keep runtime inputs in `data/`.
- `scratch/`: Ad-hoc experiments, perf runs, and attachments; gitignored by default.
- `scripts/`: One-off helpers
  - `analysis/`: plotting & exploration CLIs intended for sharing
  - `profiling/`: performance measurement scripts
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

- No secrets needed; configuration lives in `src/config.py` (`DATA_DIR`, `KERNELS_DIR").
- Use `data_acquisition` to populate SPICE kernels; track hashes with `data/spice_kernels/kernels.lock`.
