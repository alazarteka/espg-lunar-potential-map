# D1 Forward-Parity Result

## Decision

**The CPU/GPU forward-model parity gate passes. Optimizer parity remains
unopened.**

The D1 implementation defines one canonical NumPy float64 model and a padded,
batched PyTorch equivalent. No GPU optimizer, calibration case, confirmation
seed, confirmation support flux, or real pilot output was used.

## Hardware And Environment

- Four NVIDIA TITAN Xp GPUs, 12 GB each.
- Compute capability 6.1.
- PyTorch 2.6.0 with CUDA 11.8.
- Both float32 and float64 CUDA execution were tested directly.

## Implemented Equality Surface

The parity suite covers:

- geometric channel-edge construction;
- linear-energy Gaussian response averaged over each channel;
- normalized log-energy quadratic basis;
- all twelve smooth-continuum coefficients;
- the complete 15-parameter continuum-plus-beam model;
- canonical logarithmic residuals and objective;
- variable row and sample counts through finite padding and masks;
- batch-order invariance;
- finite gradients through padded batches; and
- CPU/CUDA agreement in float32 and float64.

The direct CUDA command

```bash
uv run pytest -q experiments/experiment-006/test_beam_model.py \
  experiments/experiment-006/test_beam_torch.py
```

completed with `12 passed`. In the restricted no-device sandbox, the same suite
returns `10 passed, 2 skipped`, with only the explicit CUDA cases skipped.

## Interpretation

This result establishes that the GPUs can evaluate the frozen scientific
forward model in large padded batches without changing its values or poisoning
gradients at padding locations. It does not yet show that a GPU optimizer finds
the same basin as twelve SciPy starts, produces the same covariance, or preserves
QC classifications.

The next implementation gate is a benchmark of forward and Jacobian throughput,
followed by batched projected Levenberg--Marquardt candidate search. GPU minima
will remain provisional until CPU float64 polishing reproduces the reference
winning basin under the frozen D1 tolerances.
