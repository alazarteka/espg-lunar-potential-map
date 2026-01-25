# GPU Acceleration

GPU acceleration is available for loss-cone fitting in batch mode.

## How to Use

```bash
uv run python -m src.potential_mapper.batch --fast --year 1998 --month 4
```

## Prerequisites

- GPU-capable environment: `uv sync --extra gpu` (or `--extra gpu-legacy`)
- PyTorch installed with CUDA support

## What It Accelerates

- Loss-cone forward model: `src/model_torch.py`
- Kappa fitting: `src/kappa_torch.py`
- Differential evolution optimizer: `src/utils/optimization.py`

## Execution Path (What Actually Runs on GPU)

When `--fast` is enabled in batch mode:

1. The merged dataset is split into spectra/chunks.
2. Loss-cone models are evaluated in batches on the selected torch device.
3. A batched Differential Evolution optimizer searches parameter space in GPU
   memory (no CPU↔GPU transfer inside the inner loop).
4. Chi² is computed in log-space with masks for invalid bins (same logic as CPU).

The intent is functional parity with CPU fitting while trading memory for speed.

## Auto-Detection

- **dtype**: float16 on Volta+ GPUs, float32 on older GPUs/CPU
- **batch size**: derived from available VRAM

## Batching & Memory Scaling

Memory use grows with:

- number of spectra processed in a batch
- population size in Differential Evolution
- energy bins × pitch bins per spectrum

If you hit OOM, reduce scope (month → day) or rerun without `--fast` for a
smaller slice.

## Parallel vs Fast

- `--parallel` is deprecated (legacy CPU multiprocessing)
- `--fast` uses PyTorch (GPU if available, CPU if not)
- Use `--fast` for performance; CPU parallelism now falls back to sequential

## Performance Notes

- GPU speedup depends on spectrum count and batch size
- Float16 can be faster but may slightly change numerical behavior
- VRAM pressure is the common failure mode; reduce input scope if OOM

## Precision & Reproducibility

- GPU uses the same loss-cone model math as CPU, but float16/float32 can shift
  minima slightly in flat chi² landscapes.
- Random seeds are set inside the DE implementation, but GPU nondeterminism can
  still lead to small differences run-to-run.
- If you need maximum reproducibility, run on CPU or reduce parallelism.

## Fallbacks

If PyTorch is not installed or CUDA is unavailable, the pipeline falls back to
CPU implementations without changing CLI usage (but `--fast` will be ignored).

## Troubleshooting

- **Torch installed but no GPU used**: confirm CUDA-enabled torch build and
  that a GPU is visible to the process.
- **Slow on GPU for small runs**: kernel launch overhead can dominate; CPU may
  be faster for a few spectra.
- **Model diverges with float16**: rerun on CPU or with a smaller date slice.
