# D1 Basin-Certification Result

**Decision:** Fail the GPU-candidate gate; use complete CPU multistart for C1.

**Execution commit:** `66d826f8d0e48fb7b5a254c2f25fb0d972b733c8`

**Information barrier:** Calibration geometry only. Confirmation flux was not
opened.

## Design

The frozen corpus contained 512 deterministic cases and 12 truth-blind starts
per case: 96 matched analytic cases, 160 generic model-discrepancy cases, 64
null/adversarial cases, and 192 cases on the 12 calibration supports. Every
available CPU and GPU trajectory was retained. GPU candidates were ranked by
the canonical NumPy objective and CPU-polished before comparison with the
complete SciPy multistart reference.

The preregistered decision required 100% reference-basin recall with some
`K <= 6`, finite candidates in every case, and final objective, model, and
identifiable-center parity. Failure required complete CPU fitting before C1.

## Results

The strict four-shard merge recovered every case ID exactly once and verified
one repository commit and configuration hash. The gate failed:

| Arm | Cases | Rank-1 recall | Rank-12 recall | Fallback fraction |
| --- | ---: | ---: | ---: | ---: |
| Matched analytic | 96 | 82.3% | 82.3% | 52.1% |
| Generic discrepancy | 160 | 92.5% | 92.5% | 27.5% |
| Calibration support | 192 | 88.0% | 88.0% | 38.5% |
| Null/adversarial | 64 | 62.5% | 70.3% | 90.6% |
| **All cases** | **512** | **85.2%** | **86.1%** | **44.1%** |

No `K` achieved full recall. Increasing `K` barely changed the result outside
the null/adversarial arm, showing that the dominant problem was candidate
eligibility or basin absence rather than ranking among several good basins.
Only 337 of 512 cases supplied at least six converged GPU candidates. Complete
CPU fallback was invoked in 226 cases: 177 for too few eligible candidates, 36
for bound contact, six for ambiguous polished basins, and seven because no CPU
reference trajectory was constructable.

The seven unavailable CPU references were all deliberately non-identifiable
controls: one flat-pitch case, one single-energy case, and five
zero-dynamic-range cases. They remain failed observations in the denominator;
they were not deleted or repaired after seeing the result.

## Interpretation

The float64 channel-integrated CUDA forward model remains numerically valid.
This result rejects the current projected-LM implementation as a generally
reliable candidate generator under the frozen practical starts. It does not
test, and therefore does not reject, the restricted-domain beam-proxy
hypothesis.

The decisive warning is that misses also occurred in matched, identifiable
cases. It would be invalid to waive the adversarial controls and declare the
optimizer certified. Nor would raising `K` solve the observed failure, because
rank-12 recall remained only 86.1% overall and 82.3% in matched analytic cases.

## Consequence

C1 will use the complete deterministic 12-start SciPy reference fitter. GPU
forward evaluation may still be used where it is algebraically identical, but
GPU candidate search cannot replace or prune the complete CPU starts. Any
future redesigned GPU optimizer requires a separately frozen certification;
it cannot be tuned against these 512 revealed outcomes and then reused within
Experiment 006.

The large trajectory CSV and per-shard checkpoints remain local execution
artifacts. Their SHA-256 digest is recorded in
`d1-certification-provenance.json`; the compact merged summary and provenance
are tracked.
