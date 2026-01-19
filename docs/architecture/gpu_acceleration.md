# GPU Acceleration

GPU acceleration is available for loss-cone fitting in batch mode.

## How to Use

```bash
uv run python -m src.potential_mapper.batch --fast --year 1998 --month 4
```

## What It Accelerates

- Loss-cone forward model: `src/model_torch.py`
- Kappa fitting: `src/kappa_torch.py`
- Differential evolution optimizer: `src/utils/optimization.py`

## Auto-Detection

- **dtype**: float16 on Volta+ GPUs, float32 on older GPUs/CPU
- **batch size**: derived from available VRAM

## Fallbacks

If PyTorch is not installed or CUDA is unavailable, the pipeline falls back to
CPU implementations without changing CLI usage (but `--fast` will be ignored).
