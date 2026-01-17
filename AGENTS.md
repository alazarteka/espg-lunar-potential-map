# ESPG Lunar Potential Map

> **Note:** `CLAUDE.md` is a symlink to this file. When modifying, target `AGENTS.md` directly.

## Project Overview

This project maps the lunar surface electrostatic potential using physics-based models. It processes SPICE kernel data, applies temporal reconstruction techniques, and generates potential maps accounting for plasma interactions.

### Strategic Directions

1. **Diagnostics & Validation** - Interactive tools for loss cone analysis, beam detection, and cross-validation of fitting methods. Key: `src/diagnostics/`, `scripts/diagnostics/`

2. **Production Mapping** - Batch processing pipeline, temporal harmonics reconstruction, and engineering statistics. Key: `src/potential_mapper/`, `src/temporal/`, `src/engineering/`

3. **Physics Refinement** - Improving loss cone fitting accuracy, kappa distribution modeling, and spacecraft potential corrections. Key: `src/model.py`, `src/kappa.py`, `src/physics/`

## Quick Reference

```bash
./scripts/select-env.sh cuda12         # Sync modern GPU env
./scripts/select-env.sh cuda11         # Sync legacy GPU env
uv run pytest -q                        # Run tests
uv run ruff check . && uv run mypy src  # Lint + type check
uv run python -m src.data_acquisition   # Download data
uv run python -m src.potential_mapper.batch --fast --year 1998 --month 4  # GPU batch
```

See [docs/development.md](docs/development.md) for complete workflow reference.

## Project Structure

- `src/`: Core Python package
  - `config.py`, `flux.py`, `kappa.py`, `model.py` - Core fitting
  - `spacecraft_potential.py` - Spacecraft potential calculations
  - `potential_mapper/` - Mapping pipeline
  - `temporal/` - Spherical harmonic reconstruction
  - `engineering/` - GlobalStats, SiteStats, site analysis
  - `diagnostics/` - Loss cone session management
  - `physics/`, `utils/`, `visualization/` - Supporting modules
- `scripts/`: CLI tools
  - `diagnostics/` - Beam detection and loss cone tools
  - `analysis/` - Plotting and exploration
  - `profiling/` - Performance measurement
  - `dev/` - Quick experiments
- `locks/` - Tracked lockfiles for CUDA 11/12 GPU variants
- `uv.lock` - Machine-local lockfile (gitignored; copied from `locks/`)
- `tests/`, `data/`, `artifacts/`, `docs/`, `notebooks/`

## Documentation

| Topic | Link | Description |
|-------|------|-------------|
| Architecture | [WALKTHROUGH.md](docs/WALKTHROUGH.md) | End-to-end code walkthrough |
| Development | [development.md](docs/development.md) | Build, test, lint workflow |
| Lockfiles | [lockfiles.md](docs/lockfiles.md) | GPU environment configurations |
| Batch Processing | [potential_mapper_batch.md](docs/potential_mapper_batch.md) | GPU-accelerated batch mode |
| Diagnostics | [diagnostics.md](docs/diagnostics.md) | Beam detection tools |
| Temporal | [temporal.md](docs/temporal.md) | Spherical harmonics fitting |
| Visualization | [visualization.md](docs/visualization.md) | Plot styling |

## Coding Standards

- Python 3.12, 4-space indent, Ruff line length 88
- Type hints required; `mypy` configured with `pint` plugin
- Use `src.utils.units` for physical quantities
- Conventional Commits: `feat(scope):`, `fix(scope):`, `docs:`, etc.

See [docs/development.md](docs/development.md) for full guidelines.

---

## Agent Behavior Guidelines

When working on this project, AI agents should follow these principles:

### 1. Documentation Responsibilities

**Proactively update documentation** after implementing features or making significant changes:

- Update this file (AGENTS.md) when adding new modules, scripts, or capabilities
- Document empirical findings immediately (detection rates, parameter tuning results, performance benchmarks)
- Add usage examples for new scripts in `docs/diagnostics.md` or relevant doc files
- Record parameter tuning decisions with the reasoning and results that led to them

**Don't wait to be asked.** If you add a script, document it. If you discover something useful, write it down.

### 2. Strategic Direction Awareness

Understand which workstream your current task belongs to:

**Diagnostics & Validation**
- Loss cone analysis, beam detection, cross-validation
- Interactive visualization (losscone_studio, losscone_orbit_studio)
- Data quality assessment
- Work primarily in: `src/diagnostics/`, `scripts/diagnostics/`

**Production Mapping**
- Batch processing for full dataset
- Temporal reconstruction with spherical harmonics
- Engineering statistics (GlobalStats, SiteStats)
- Work primarily in: `src/potential_mapper/`, `src/temporal/`, `src/engineering/`

**Physics Refinement**
- Loss cone model improvements
- Kappa distribution fitting accuracy
- Spacecraft potential corrections
- Work primarily in: `src/model.py`, `src/kappa.py`, `src/physics/`

When working in an area, reference existing code patterns and tests.

### 3. Code Quality Expectations

Before declaring work complete:

- Run `uv run ruff check .` and fix issues
- Run `uv run mypy src` and address type errors
- Run `uv run pytest -q` to verify nothing broke
- Add type hints to new functions
- Follow existing patterns in the codebase (look at similar files)
- Use `src.utils.units` for physical quantities with units

**Prefer simple, focused changes.** Don't over-engineer. Fix what was asked, no more.

### 4. Script Development Guidelines

Place new scripts in the appropriate directory:

| Directory | Purpose | Polish Level |
|-----------|---------|--------------|
| `scripts/diagnostics/` | Data quality, beam detection | Production |
| `scripts/analysis/` | Plotting, exploration (shareable) | Production |
| `scripts/profiling/` | Performance measurement | Internal |
| `scripts/dev/` | Quick experiments | Rough |
| `scratch/` | Ad-hoc experiments | Gitignored |

For production scripts:
- Include a docstring with usage examples
- Add to `docs/diagnostics.md` or relevant documentation
- Follow CLI conventions (argparse, `--help` support)

For dev scripts:
- Can be rough, but should still be runnable
- Graduate to `analysis/` or `diagnostics/` when mature

### 5. Research & Experimentation

When exploring new approaches:

- Use `scratch/` for throwaway experiments (gitignored)
- Document experimental plans before starting major explorations
- Record findings even if the experiment fails
- Reference prior work (check `scripts/dev/` and git history)

When an experiment succeeds:
- Graduate the code to the appropriate location
- Add tests for the new functionality
- Document the approach and any tuning decisions

### 6. Autonomy vs Asking

**Proceed autonomously when:**
- The task is well-defined with clear success criteria
- You're following established patterns in the codebase
- Making targeted fixes or additions within a single module
- Running diagnostics, tests, or analyses

**Ask the user when:**
- The task involves architectural decisions affecting multiple modules
- You're uncertain about the intended behavior or scope
- Multiple valid approaches exist with different tradeoffs
- You need to make assumptions about physics or scientific interpretation
- Something unexpected comes up that changes the plan

**Default to asking if unsure.** It's better to clarify than to implement the wrong thing.

### 7. Git Workflow

- Use Conventional Commits with scope: `feat(physics): ...`, `fix(temporal): ...`
- Pre-merge: ruff, mypy, pytest must all pass
- Don't commit large data files or notebook outputs
- Update docs if interfaces or behaviors change

### 8. Common Patterns

**Adding a new diagnostic script:**
1. Create in `scripts/diagnostics/`
2. Add docstring with usage example
3. Document in `docs/diagnostics.md`
4. Update AGENTS.md if it's a significant new capability

**Improving a physics model:**
1. Add tests first (or alongside)
2. Verify with existing data
3. Document any parameter changes
4. Update relevant docs if behavior changes

**Running batch analyses:**
1. Use `--fast` flag for GPU acceleration
2. Check `artifacts/` for existing cached results
3. Document findings in relevant doc files
