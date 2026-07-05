# Environment Lockfiles

This project uses uv for dependency management with multiple GPU environment configurations.

## Lockfile Structure

| Lockfile | Environment | Python | CUDA | torch | Purpose |
|----------|-------------|--------|------|-------|---------|
| `uv.lock` | Default (CPU) | 3.12 | N/A | — | Development without GPU (machine-local, gitignored) |
| `locks/uv.lock.cuda12` | GPU (Modern) | 3.12 | 12.8 | 2.7.1+cu128 | sm_70+ (RTX 20xx and newer, including 50xx) |
| `locks/uv.lock.cuda11` | GPU (Legacy) | 3.12 | 11.8 | 2.6.0+cu118 | Pascal sm_61 (GTX 10xx / TITAN Xp) |

`uv.lock` is machine-local (gitignored) and written by `bootstrap.sh`, which
copies the matching `locks/*` file into place before syncing.

## Usage

```bash
# Default environment (CPU)
uv sync

# GPU: auto-detect the architecture and sync the matching stack
./scripts/bootstrap.sh

# Force a stack explicitly
./scripts/bootstrap.sh modern   # CUDA 12.8, sm_70+
./scripts/bootstrap.sh legacy   # CUDA 11.8, Pascal sm_61
```

## Lockfile Selection

`bootstrap.sh` detects the GPU compute capability (via
`nvidia-smi --query-gpu=compute_cap`) and copies the appropriate lockfile to
`uv.lock` before running `uv sync --frozen`. It picks by the *oldest* visible
GPU so a mixed-generation host still gets a build that runs on every card
(`sm_70+` → modern CUDA 12.8; `sm_61` → legacy CUDA 11.8). Pass `modern` or
`legacy` to override detection.

## Regenerating Lockfiles

> `uv lock` has **no** `--lockfile` flag — it only ever writes `./uv.lock`. Each
> `locks/*` file is produced by locking, then copying `uv.lock` into place. The
> two GPU locks cannot be produced by one `uv lock` run because the `gpu` and
> `gpu-legacy` extras pin mutually exclusive torch builds (see Known Issues).

### Modern (`locks/uv.lock.cuda12`)

The default `pyproject.toml` resolves the `gpu` (cu128) fork directly:

```bash
uv lock
cp uv.lock locks/uv.lock.cuda12
```

### Legacy (`locks/uv.lock.cuda11`)

The cu118 torch will **not** materialize while the `gpu` extra,
`required-environments`, and the `extra != 'gpu'` source marker are all present
(uv resolves only the cu128 fork). Generate it from a temporary legacy-only
configuration, then restore `pyproject.toml`:

1. In `[project.optional-dependencies]`, remove the `gpu` line (keep `gpu-legacy`).
2. In `[tool.uv]`, remove the `required-environments` line.
3. In `[tool.uv.sources]`, replace the torch entry with an unconditional
   `torch = [{ index = "pytorch-cu118" }]`.
4. Run and capture:
   ```bash
   uv lock
   cp uv.lock locks/uv.lock.cuda11
   git checkout pyproject.toml   # restore the full config
   ```

Verify each lock installs the expected torch without downloading everything:

```bash
cp locks/uv.lock.cuda12 uv.lock && uv sync --frozen --extra gpu --dev --dry-run
cp locks/uv.lock.cuda11 uv.lock && uv sync --frozen --extra gpu-legacy --dev --dry-run
```

---

## Known Issues

### Modern (cu128) torch is capped at 2.7.1 on linux-x86_64

**Problem:** `torch>=2.8` (cu128) fails to resolve for the required
`x86_64-linux` environment:

```
nvidia-cublas-cu12==12.8.4.1 has no `platform_machine == 'x86_64' ...`-compatible wheels
```

**Cause:** torch 2.8 introduced a hard pin on the `cuda-toolkit` metapackage
(`nvidia-cublas-cu12==12.8.4.1`). That wheel *does* exist for x86_64
(`manylinux_2_27_x86_64`), but uv rejects it for the abstract required
environment (no glibc floor is declared), so only `<2.8` resolves. This is a
resolver-level gap, not a missing wheel; revisit when uv or the pytorch index
changes. The modern lock therefore stays at **2.7.1+cu128** on x86_64.

### Legacy (cu118) runs Pascal sm_61 via PTX JIT, not native SASS

torch 2.6.0+cu118 ships SASS for `sm_50, sm_60, sm_70, sm_75, sm_80, sm_86,
sm_90` — **`sm_61` is not in the list**. Consumer Pascal (GTX 10xx / TITAN Xp,
capability 6.1) runs anyway because the driver JIT-compiles the wheel's PTX to
sm_61 on first kernel launch (then caches it in `~/.nv/ComputeCache`). Verified
end-to-end: matmul, softmax, argsort/topk, gather, `linalg.solve`/`lstsq`, erf,
and autocast all execute correctly on the TITAN Xp. If a future op ever lacks
PTX coverage (`no kernel image is available for execution on the device`), drop
the legacy pin to a version whose cu118 build still lists `sm_61` natively
(e.g. `torch>=2.4,<2.5`).

### CUDA 11 lockfile and diagnostics

**Problem:** `locks/uv.lock.cuda11` does not include the diagnostics
dependencies (`panel`, `bokeh`).

**Symptom:** `ModuleNotFoundError: No module named 'panel'`

**Workaround:** Install them on top of the legacy environment:

```bash
./scripts/bootstrap.sh legacy
uv pip install panel bokeh
```

### The two GPU extras pin conflicting torch versions

`gpu` requires `torch>=2.7.0` (cu128) and `gpu-legacy` requires `torch>=2.6,<2.7`
(cu118). uv cannot hold both in one lock, which is why they are generated and
committed as separate files (see Regenerating Lockfiles). Never sync both extras
together.

### NVIDIA package conflict warnings

During `uv sync` you may see:

```
warning: The module `nvidia` is provided by more than one package ...
```

These are harmless — multiple NVIDIA packages (cublas, cufft, curand, …) all
ship a top-level `nvidia` module. The packages install correctly regardless.

## Dependency Groups

| Group | Contents |
|-------|----------|
| `dev` | pytest, ruff, mypy, snakeviz |
| `test` | pytest, pytest-cov |
| `lint` | ruff, mypy |
