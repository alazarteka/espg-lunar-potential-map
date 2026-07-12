# D1 Protocol: CPU Reference And GPU Candidate Backend

**Status:** Frozen before GPU parity or benchmark outputs

**Configuration:** [`configuration-d1.json`](configuration-d1.json)

## Purpose

D1 may accelerate Experiment 006 implementation tests and simulations, but it
must not change the scientific objective, deterministic starts, selected
minimum, uncertainty scale, or QC classification. The CPU float64 implementation
is the numerical reference. GPU arithmetic initially generates candidate
minima; CPU float64 certifies every reported result.

The available system has four NVIDIA TITAN Xp GPUs, each with 12 GB memory and
compute capability 6.1. The verified environment is PyTorch 2.6.0 with CUDA
11.8. These Pascal GPUs favor float32 throughput; final covariance and
threshold-adjacent decisions remain on CPU float64.

## Canonical Forward Model

[`beam_model.py`](beam_model.py) defines the reference residual:

\[
r_i(\theta)=\log R_i(\theta)-\log y_i,
\]

where

\[
R(E,\alpha)=C(E,\alpha)+A\,G_E(E;E_b)
G_\alpha(\alpha;180^\circ,\sigma_\alpha).
\]

The four continuum quantities remain bounded quadratic functions of normalized
log energy. The energy response is a linear-energy Gaussian with nominal
`FWHM/E = 0.48`, averaged over channel intervals inferred from geometric
midpoints of the native energy centers. This replaces center-only response
evaluation without introducing a voltage equation into the continuum.

The parameter order is amplitude, center, pitch sigma, then twelve smooth
continuum coefficients. NumPy float64 output is canonical.

## Backend Roles

### CPU reference

- Constructs the twelve deterministic starts.
- Fits every start with bounded SciPy nonlinear least squares.
- Defines the reference winning basin and objective.
- Computes the final numerical Jacobian, rank, covariance, standard errors,
  bound contacts, stability diagnostics, and QC.
- Remains available as a complete fallback.

### GPU candidate search

- Evaluates padded batches shaped by case, deterministic start, and sample.
- Maintains an explicit validity mask; padded values are made finite before any
  logarithm or gradient operation.
- Uses independently converging, bounded trajectories in scaled parameter
  coordinates.
- Returns candidate parameters and canonical objective values, never final
  scientific classifications.

### Hybrid certification

The planned optimizer is batched projected Levenberg--Marquardt/Gauss--Newton.
For each trajectory it solves

\[
(J^T J+\lambda D)\Delta=-J^T r
\]

with independent damping and stopping state. The best GPU candidates are
polished by SciPy float64. The initial number of polished candidates is three,
but that value is not frozen until implementation-only basin-recall tests pass.
Ambiguous objectives, separated center basins, bound contacts, nonfinite
Jacobians, or CPU/GPU objective disagreement trigger expansion or complete CPU
fallback.

Adam, batch-wide stopping, and the existing three-parameter differential-
evolution backend are not eligible substitutes. They do not provide the same
per-trajectory least-squares semantics in this 15-parameter problem.

## Information Barrier

Backend development, tolerance selection, and performance tuning may use:

- analytic and noiseless implementation cases;
- generic synthetic supports; and
- the twelve frozen calibration supports.

They may not use confirmation seeds, confirmation residual templates,
confirmation support flux morphology, or partial V1 output. A backend change
after C1 begins requires a fresh calibration namespace. A change after V1 begins
invalidates Experiment 006.

## Required Parity Gates

Before optimizer benchmarking:

- NumPy/Torch float64 forward agreement: `rtol <= 1e-9`.
- Torch float32 forward agreement: `rtol <= 2e-5`.
- Float32 objective relative difference: `<= 1e-4`.
- Finite autograd Jacobians on matched, edge, broad, and near-singular cases.
- Padding and batch-order invariance.
- Repeatability across reruns, batch sizes, and available GPUs.

Before C1:

- GPU top-K contains the CPU winning basin in every verification case.
- CPU-polished center difference is no greater than `1e-4` local response FWHM.
- CPU-polished objective relative difference is no greater than `1e-8`.
- Final QC classification agreement is 100%.
- No material difference in bound contacts or fallback classifications.

The gates may be tightened after implementation-only tests. They may not be
relaxed after calibration or confirmation outcomes are inspected.

## Benchmark Matrix

Benchmark batch sizes `16, 32, 64, 128, 256` in float32 and float64. Record:

- forward, Jacobian, LM-iteration, and complete-fit throughput;
- CPU start-construction and polish time;
- leave-one-energy-out time;
- peak VRAM and GPU utilization;
- convergence, fallback, and top-K basin-recall rates; and
- one-GPU versus four-GPU throughput.

Eager mode is primary on Pascal. `torch.compile` is a labeled comparison only.
A 512-case implementation dry run will estimate total C1/V1 runtime. Static
multi-GPU sharding uses `case_id % device_count`, one process per GPU, atomic
per-device outputs, and a canonical ordered merge with duplicate/missing-case
checks.

## Decision

A GPU speedup is accepted only if every numerical parity gate passes. If forward
parity passes but basin recall does not, GPU evaluation may still accelerate
simulation generation and diagnostics while the complete estimator remains
CPU-only. Runtime is never a reason to weaken the frozen scientific matrix.
