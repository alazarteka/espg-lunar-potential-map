# Environment Lockfiles

This project uses uv for dependency management with multiple GPU environment configurations.

## Lockfile Structure

| Lockfile | Environment | Python | CUDA | Purpose |
|----------|-------------|--------|------|---------|
| `uv.lock` | Default (CPU) | 3.12 | N/A | Development without GPU |
| `locks/uv.lock.cuda12` | GPU (Modern) | 3.12 | 12.8 | RTX 20xx+, including 50xx |
| `locks/uv.lock.cuda11` | GPU (Legacy) | 3.12 | 11.8 | GTX 10xx (sm_61) |

## Usage

```bash
# Default environment (CPU)
uv sync

# Modern GPU (CUDA 12.8)
./scripts/select-env.sh cuda12

# Legacy GPU (CUDA 11.8)
./scripts/select-env.sh cuda11
```

## Lockfile Selection

The `select-env.sh` script copies the appropriate lockfile to `uv.lock` before running `uv sync`.

## Adding Dependencies

### Core Dependencies

Add to `[project.dependencies]` in `pyproject.toml`, then regenerate lockfiles:

```bash
uv lock --lockfile locks/uv.lock.cuda12
uv lock --lockfile locks/uv.lock.cuda11
```

### GPU-Specific Dependencies

Add to `[project.optional-dependencies]`:

```toml
[project.optional-dependencies]
gpu = ["torch>=2.7.0"]
gpu-legacy = ["torch>=2.2,<2.3"]
```

### Diagnostics Dependencies

Add to `[project.optional-dependencies]`:

```toml
diagnostics = ["panel>=1.4.0", "bokeh>=3.4.0"]
```

**Note:** The diagnostics extra (`panel`, `bokeh`) is not included in GPU lockfiles by default. To add it permanently:

```bash
# After adding diagnostics to pyproject.toml
uv lock --extra diagnostics --lockfile locks/uv.lock.cuda12
uv lock --extra diagnostics --lockfile locks/uv.lock.cuda11
```

---

## Known Issues

### CUDA 11 Lockfile and Diagnostics

**Problem:** The CUDA 11 lockfile (`locks/uv.lock.cuda11`) does not include the diagnostics dependencies (`panel`, `bokeh`).

**Symptom:**
```
ModuleNotFoundError: No module named 'panel'
```

**Workaround:** Install diagnostics packages on top of the CUDA 11 environment:

```bash
./scripts/select-env.sh cuda11
uv pip install panel bokeh
```

**Permanent Fix:** Regenerate the lockfile with the diagnostics extra:

```bash
# Make sure diagnostics is in pyproject.toml [project.optional-dependencies]
uv lock --extra diagnostics --lockfile locks/uv.lock.cuda11
```

### Torch Version Conflict

**Problem:** The `pyproject.toml` specifies conflicting torch versions:
- `gpu` extra requires `torch>=2.7.0`
- `gpu-legacy` extra requires `torch>=2.2,<2.3`

This prevents generating a single lockfile with both extras.

**Solution:** Lockfiles are generated separately for each GPU variant to avoid this conflict. Never run `uv lock` without specifying a lockfile when using GPU extras.

### NVIDIA Package Conflicts

**Symptom:** During `uv sync`, warnings like:
```
warning: The module `nvidia` is provided by more than one package, which causes
an install race condition and can result in a broken module.
```

**Cause:** Multiple NVIDIA packages (cublas, cufft, curand, etc.) all provide a `nvidia` module.

**Impact:** These are harmless warnings. The packages install correctly despite the warning.

### Panel/Bokeh Not in Default Lockfile

**Problem:** Interactive diagnostics tools require `panel` and `bokeh`, but these are not in the default `uv.lock`.

**Solution:** Use the `diagnostics` extra:
```bash
uv sync --extra diagnostics
```

---

## Regenerating Lockfiles

When dependencies change, regenerate all lockfiles:

```bash
# Default (CPU)
uv lock

# CUDA 12 GPU
uv lock --lockfile locks/uv.lock.cuda12

# CUDA 11 GPU
uv lock --lockfile locks/uv.lock.cuda11

# With diagnostics
uv lock --extra diagnostics --lockfile locks/uv.lock.cuda12
uv lock --extra diagnostics --lockfile locks/uv.lock.cuda11
```

## Dependency Groups

| Group | Contents |
|-------|----------|
| `dev` | pytest, ruff, mypy, snakeviz |
| `test` | pytest, pytest-cov |
| `lint` | ruff, mypy |
