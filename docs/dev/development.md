# Development Workflow

Complete reference for building, testing, and contributing to the project.

## Prerequisites

- Python 3.12 (required, no other versions supported)
- [UV](https://docs.astral.sh/uv/) package manager
- Git

## Environment Setup

### Basic Installation

```bash
# Install core dependencies
uv sync

# Install with development tools (CPU-only)
uv sync --group dev
```

### Optional Extras

| Extra | Command | Description |
|-------|---------|-------------|
| `notebook` | `uv sync --extra notebook` | Jupyter notebook support |
| `export` | `uv sync --extra export` | Image/video export (Kaleido, imageio) |
| `diagnostics` | `uv sync --extra diagnostics` | Interactive browser tools (Panel, Bokeh) |
| `gpu` | `./scripts/select-env.sh cuda12` | PyTorch GPU acceleration (CUDA 12.8, RTX 20xx+) |
| `gpu-legacy` | `./scripts/select-env.sh cuda11` | PyTorch for older GPUs (CUDA 11.8, GTX 10xx) |

### GPU Lockfiles

This repo tracks CUDA-specific lockfiles under `locks/`. `uv.lock` is
machine-local (gitignored) and copied from `locks/` via the selector script.

```bash
# Modern GPUs (RTX 20xx, 30xx, 40xx, 50xx)
./scripts/select-env.sh cuda12

# Legacy GPUs (GTX 10xx series)
./scripts/select-env.sh cuda11
```

### Upgrading Dependencies

```bash
# Upgrade to latest compatible versions
uv sync --upgrade

# Regenerate lock file (machine-local)
uv lock --upgrade

# Update the tracked lockfile for your GPU tier
cp uv.lock locks/uv.lock.cuda12  # or locks/uv.lock.cuda11
```

---

## Running Code

### Main Modules

```bash
# Interactive potential mapping
uv run python -m src.potential_mapper --year 1998 --month 1 --day 15 -d

# Batch processing (use --fast on GPU machines)
uv run python -m src.potential_mapper.batch --fast --year 1998 --month 4

# Temporal harmonic fitting
uv run python -m src.temporal

# Data acquisition (download SPICE kernels and ER data)
uv run python -m src.data_acquisition
```

### Scripts

```bash
# Run any script
uv run python scripts/diagnostics/losscone_peak_scan.py data/...

# Interactive tools (requires diagnostics extra)
uv run python scripts/diagnostics/losscone_studio.py data/...
```

---

## Testing

### Running Tests

```bash
# Quick test run
uv run pytest -q

# With coverage
uv run pytest --cov=src

# Verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_flux.py

# Run tests matching pattern
uv run pytest -k "kappa"
```

### Test Markers

```bash
# Skip slow/data-dependent tests (used in CI)
uv run pytest -m "not skip_ci"

# Run only slow tests
uv run pytest -m slow
```

### Test Guidelines

- Test files: `tests/test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`
- Use synthetic fixtures to avoid network/file dependencies (see `tests/test_kappa_fitter.py`)
- Mark data-dependent tests with `@pytest.mark.skip_ci`

---

## Linting & Formatting

### Ruff (Linting + Formatting)

```bash
# Check for issues
uv run ruff check .

# Auto-fix issues
uv run ruff check . --fix

# Format code
uv run ruff format src tests

# Check format without modifying
uv run ruff format --check src tests
```

**Ruff rules enabled:**
- `E, F, W`: pycodestyle errors/warnings, pyflakes
- `B`: flake8-bugbear
- `I`: isort (import sorting)
- `UP`: pyupgrade
- `SIM`: flake8-simplify
- `C4`: flake8-comprehensions
- `PT`: flake8-pytest-style
- `RUF`: Ruff-specific rules
- `TCH`: flake8-type-checking
- `PERF`: performance lint

### MyPy (Type Checking)

```bash
# Full type check
uv run mypy src

# Check specific module
uv run mypy src/potential_mapper
```

---

## Pre-commit Hooks

### Setup

```bash
# Install hooks (one-time)
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

### What Runs

Pre-commit runs:
1. Ruff check + auto-fix
2. Ruff format
3. Pytest (CPU-only; `CUDA_VISIBLE_DEVICES=""`)

Note: Hooks are configured as local `uv --no-cache run ...` commands (no network fetch). If
your environment can't write to `~/.cache/pre-commit`, set
`PRE_COMMIT_HOME=/tmp/pre-commit`.

---

## GPU Acceleration

### Setup

```bash
# Modern GPUs (RTX 20xx, 30xx, 40xx, 50xx)
./scripts/select-env.sh cuda12

# Legacy GPUs (GTX 10xx series)
./scripts/select-env.sh cuda11
```

### Usage

```bash
# Batch processing with GPU acceleration
uv run python -m src.potential_mapper.batch --fast --year 1998 --month 4
```

The `--fast` flag enables:
- **Auto dtype**: float16 on Volta+ GPUs, float32 on older/CPU
- **Auto batch size**: Calculated from available VRAM
- **~160-1000x speedup** over CPU for large batches

### Benchmarking

```bash
# GPU batch size sweep
uv run python scripts/profiling/gpu_batch_sweep.py

# Profile full pipeline
uv run python scripts/profiling/potential_pipeline_profile.py
```

---

## Commit Guidelines

### Message Format

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

[optional body]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `refactor`: Code restructuring (no behavior change)
- `style`: Formatting, whitespace
- `test`: Adding/updating tests
- `chore`: Build, CI, dependencies

**Examples:**
```
feat(physics): add kappa distribution temperature weighting
fix(temporal): correct harmonic coefficient sign convention
docs: update diagnostics usage examples
refactor(model): extract common loss cone calculation
```

### Pre-merge Checklist

Before merging:
- [ ] `uv run ruff check .` passes
- [ ] `uv run mypy src` passes
- [ ] `uv run pytest -q` passes
- [ ] Update docs if interfaces changed
- [ ] No large data files or notebook outputs committed

---

## Directory Conventions

| Directory | Purpose |
|-----------|---------|
| `scripts/diagnostics/` | Data quality and beam detection tools |
| `scripts/analysis/` | Plotting and exploration (shareable) |
| `scripts/profiling/` | Performance measurement |
| `scripts/dev/` | Quick experiments (may not be polished) |
| `scripts/plots/` | Paper-ready publication figures |
| `scratch/` | Ad-hoc experiments (gitignored) |

---

## Troubleshooting

### SPICE Kernel Issues

```bash
# Re-download all kernels
uv run python -m src.data_acquisition --force
```

### GPU Not Detected

```python
# Check PyTorch CUDA status
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
```

### Import Errors

```bash
# Ensure all dependencies installed
uv sync --locked --all-extras --dev
```
