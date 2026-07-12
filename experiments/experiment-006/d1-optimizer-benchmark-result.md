# D1 Batched-LM Benchmark Result

## Decision

**The float64 GPU candidate optimizer passes the implementation recovery and
throughput gate. Proceed to CPU/GPU basin-recall certification.**

The optimizer was frozen in commit `e5558dc`; the benchmark runner was frozen
in `7ca74bc`. All cases are deterministic, noiseless, implementation-only
problems. No calibration output, confirmation support flux, confirmation seed,
residual template, or real-data fit was opened.

## GPU 0 Batch Matrix

Each case has 12 starts. The complete result is
[`d1-optimizer-benchmark.json`](d1-optimizer-benchmark.json).

| Cases | Trajectories | Time | Trajectories/s | Converged | Median iterations | Peak memory |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12 | 0.917 s | 13 | 100% | 6 | 53 MiB |
| 4 | 48 | 0.133 s | 362 | 100% | 7 | 166 MiB |
| 8 | 96 | 0.148 s | 648 | 100% | 7 | 314 MiB |
| 16 | 192 | 0.176 s | **1,088** | 100% | 7 | 612 MiB |
| 21 | 252 | 0.237 s | 1,062 | 100% | 7 | 796 MiB |

The one-case row includes first-use autograd/CUDA setup and is not a steady-
state throughput estimate. A fresh-process 192-trajectory run takes about 1.08
seconds on every card; once the process is warm, the same workload takes about
0.18 seconds.

All 588 trajectories in the batch matrix converged within nine iterations.
Across the largest batch, median center error was `2.79e-10` response FWHM and
the maximum was `1.89e-8` response FWHM. These are matched noiseless local
problems and measure numerical implementation, not scientific recovery under
model discrepancy.

## CPU Comparison

The identical 192-trajectory float64 PyTorch computation on CPU reaches 73.2
trajectories/s after setup, compared with 1,087.9 trajectories/s on one warm
GPU. The measured single-GPU speedup is therefore approximately 14.9 times for
this corpus. Four warm static shards have a nominal aggregate rate near 4,300
trajectories/s, before process startup, I/O, CPU polishing, and fallback costs.

The planned 13,600 simulation cases contain 163,200 twelve-start trajectories.
At the measured local-problem rate, raw four-GPU candidate search would take
under one minute. Realistic full-support observations, difficult basins,
leave-one-row-out fits, and CPU certification will increase that time; the
result supports a minutes-scale GPU search budget, not a final end-to-end
runtime promise.

## Device Consistency

A fresh process on each of the four TITAN Xp cards produced identical
parameters, errors, objectives, iteration counts, and convergence decisions for
the 192-trajectory corpus. Cold-start throughput was 176--178 trajectories/s
per card and peak allocation was 612 MiB. The narrow speed range and identical
numerics support deterministic static sharding.

## Limitations And Next Gate

The benchmark starts close to known matched-model truth. It does not establish:

- recall of the best basin among the frozen twelve practical starts;
- behavior under nulls, weak beams, model discrepancy, or support padding;
- equivalence of GPU candidate ranking and complete SciPy multistart fitting;
- adequacy of provisional CPU polish `K=3`;
- parity of covariance, leave-one-energy-out checks, or final QC; or
- end-to-end multi-GPU output merging and restart safety.

The next D1 corpus must compare every GPU candidate basin with all twelve CPU
SciPy starts on implementation-only and calibration-support cases. Top-K is
frozen only after 100% reference-basin recall and 100% final QC agreement.
