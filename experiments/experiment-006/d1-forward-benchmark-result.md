# D1 Forward Benchmark Result

## Decision

**Use float64 for GPU candidate search. Float32 is not accepted as the D1
scientific search precision.**

The benchmark ran after the forward implementation was frozen in commit
`6482735`. It used deterministic implementation-only cases on the native LP
energy grid. No confirmation support flux, residual template, seed, calibration
output, or real-data fit was opened.

## Single-GPU Matrix

GPU 0 evaluated batch sizes 16, 32, 64, 128, and 256 in float32 and float64.
The complete machine-readable result is
[`d1-forward-benchmark.json`](d1-forward-benchmark.json).

| Dtype | Batch | Case evaluations/s | Peak allocated memory | Maximum relative error |
| --- | ---: | ---: | ---: | ---: |
| float32 | 16 | 21,774 | 8.95 MiB | `4.49e-5` |
| float32 | 64 | 86,546 | 11.40 MiB | `1.60e-5` |
| float32 | 256 | 343,696 | 21.20 MiB | `2.11e-6` |
| float64 | 16 | 21,694 | 9.56 MiB | `6.22e-16` |
| float64 | 64 | 83,783 | 13.85 MiB | `9.17e-13` |
| float64 | 256 | 303,062 | 31.31 MiB | `7.16e-15` |

One float32 case at batch 16 exceeded the frozen pure-relative forward gate of
`2e-5`, reaching `4.49e-5`. The earlier focused parity suite passed because it
used a different, smaller implementation corpus and an explicit absolute
tolerance. The gate is not relaxed after observing this result.

Float64 passes the frozen `1e-9` relative gate by several orders of magnitude.
At batch 256 it is only 11.8% slower than float32 for this overhead-dominated
forward calculation. The modest speed difference does not justify a separate
float32 basin-recall risk.

## Four-GPU Consistency

Each TITAN Xp was independently run for 200 iterations at batch sizes 128 and
256 in float64.

| GPU | Batch 128 evaluations/s | Batch 256 evaluations/s |
| ---: | ---: | ---: |
| 0 | 167,058 | 305,264 |
| 1 | 163,280 | 296,999 |
| 2 | 163,023 | 301,882 |
| 3 | 165,611 | 297,804 |

All devices returned identical maximum relative errors (`2.57e-14` at batch
128 and `7.16e-15` at batch 256). Aggregate forward capacity is approximately
1.20 million case-evaluations/s at four simultaneous batch-256 shards, before
optimizer Jacobian and linear-solve costs.

## Interpretation

The benchmark establishes that these GPUs are not memory-limited for the D1
model and that float64 forward evaluation is fast enough to remain canonical.
It does not predict complete-fit speed: Jacobian construction, 15-by-15 solves,
independent damping, CPU start construction, SciPy polishing, leave-one-row-out
fits, and fallback rates still require measurement.

The next backend therefore uses:

- GPU float64 model, residual, Jacobian, and candidate iteration;
- batch size 256 as the initial throughput target;
- one static process per GPU;
- CPU float64 polishing and final covariance/QC; and
- no float32 scientific search path unless a future separately preregistered
  experiment demonstrates full basin and classification parity.
