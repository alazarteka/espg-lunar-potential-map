# Experiment 002: LP ER Count Reconstruction Audit

**Status:** Complete

**Decision:** Outcome C, partial recovery

**Branch:** `experiments`

**Depends on:** [Experiment 001](../experiment-001/README.md)

## Scientific Question

Can the calibrated Lunar Prospector Electron Reflectometer 3-D flux product be
mapped to statistically meaningful independent counts closely enough to support
a Poisson likelihood, or must inference use an archive-level uncertainty and
model-discrepancy description?

Experiment 001 showed that the current count reconstruction gives the published
benchmark a reduced Cash deviance near 1,421 and shifts its relative voltage from
about -162 V to -253 V. Background, floor, mask-side, pairwise-normalization, and
row-intensity ablations do not explain the failure. The unresolved question is
how much of the asserted information scale is justified by the telemetry and
calibration chain.

## Competing Outcomes

### Outcome A: Recoverable Count Likelihood

The calibrated values can be mapped to original counters or well-defined count
intervals with known exposure, prescaling, decompression, dead-time treatment,
and background. A likelihood can then be formulated at that raw or interval-
censored level.

### Outcome B: Derived-Product Likelihood Only

The calibrated flux cannot be inverted into independent Poisson observations
because compression, prescaling, dead-time correction, averaging, calibration,
or unavailable metadata destroy the required count-level information. The
current pure-Poisson Cash method must then be retired as a formal likelihood.

### Outcome C: Partial Recovery

Counts are recoverable only for specific telemetry modes, count-rate ranges, or
energy/angular samples. A count likelihood may be usable on a restricted subset,
with explicit censoring and exclusions, but not as an archive-wide default.

## Frozen Inputs

### Primary Benchmark

- File: `data/1999/091_120APR/3D990429.TAB`
- Spectrum: `spec_no = 653`, internal chunk 650
- UTC rows: 1999-04-29 14:44:08 through 14:45:18
- Published target: `U_M = -160 V`, `U_sc = +11 V`, `B_S/B_M = 0.975`

### Timestamp Controls

- Spectrum 652: immediately before the benchmark.
- Spectrum 654: immediately after the benchmark.

### Count-Level Controls

Before inspecting reconstruction alternatives, select a small fixed set of
energy/channel samples spanning:

- Zero or minimum positive flux.
- Linear and logarithmic compression regimes.
- Median and high reconstructed count levels.
- Prescaled and unprescaled telemetry, if metadata permit identification.
- Low-energy and high-energy channels.
- Multiple latitude bands and inferred dwell times.

The final sample list and selection code must be committed before evaluating
agreement. This gate is now frozen in `sample_manifest.csv`: 16 bins selected
only from calibrated spectrum 653, spanning zero flux, positive-count proxy
quantiles, both energy extremes, and all angular exposure multiplicities. The
selection script does not read Level-0 telemetry. Its exposure proxy records the
pre-comparison angular-only hypothesis; that hypothesis was subsequently
rejected after the energy-sweep duty cycle was included.

## Authoritative Sources

- LP MAG/ER telemetry format and counter compression:
  <https://pds-ppi.igpp.ucla.edu/volume/LP_1201/DOCUMENT/PROJECT/LVL0SIS.PDF>
- Governing Format-3 frame table for the 1999 product:
  <https://pds.nasa.gov/data/lp-l-6-position-v1.0/lp_0016/document/magerfmt.pdf>
- Calibrated 3-D electron flux collection:
  <https://pds-ppi.igpp.ucla.edu/data/lp-er-calibrated/data-3deleflux/>
- PDS3 source data set:
  <https://pds.nasa.gov/ds-view/pds/viewProfile.jsp?dsid=LP-L-ER-3-RDR-3DELEFLUX-80SEC-V1.1>
- Derived electron-product description:
  <https://pds.nasa.gov/ds-view/pds/viewProfile.jsp?dsid=LP-L-ER-4-ELECTRON-DATA-V1.0>

Repository comments and previous analysis are hypotheses, not authoritative
instrument documentation.

## Work Packages

### 1. Product Lineage

Document the complete transformation for a 3-D flux bin:

1. Detector counter accumulation.
2. Energy and angular acquisition mode.
3. Accumulation duration and number of contributing spins/half-spins.
4. Counter prescaling.
5. Eight-bit logarithmic compression.
6. Ground decompression convention and represented count interval.
7. Dead-time correction and its validity limits.
8. Background handling.
9. Geometric factor, energy passband, and angular acceptance.
10. Conversion from corrected count rate to archived differential flux.

Every factor must have an instrument-document citation or be labeled unknown.

### 2. Exposure Audit

Test the current assumption

```text
dt_channel = 2.5 seconds / N_azimuth(latitude band)
```

against the telemetry mode used by each of the 15 energy rows. Determine whether
the 3-D product accumulates one half-spin, several half-spins, or mode-dependent
samples for each energy/channel bin.

### 3. Counter-Code Audit

Implement the documented compression/decompression table and determine whether
the current reconstructed values:

- Match a valid decompressed counter code.
- Fall inside the count interval represented by a compressed byte.
- Differ only by documented dead-time/calibration corrections.
- Require an unexplained scale factor.

Do not round calibrated values and call them observed counts unless this mapping
has been demonstrated.

### 4. Uncertainty Budget

Quantify separately, where possible:

- Poisson counting variance.
- Compression/digitization interval.
- Prescaling uncertainty or saturation.
- Dead-time correction uncertainty.
- Background uncertainty.
- Geometric-factor and calibration uncertainty.
- Correlation introduced by aggregation or normalization.

Report uncertainty as a function of energy, angular band, and count level rather
than as one global multiplier.

### 5. Information-Scale Check

Experiment 001 found that a common Poisson exposure multiplier near `7e-4` would
be required to reduce the benchmark deviance from about 1,421 to one. Compare
the documented telemetry/calibration uncertainty with this scale. This is a
diagnostic, not a proposed correction factor.

## Planned Artifacts

| Artifact | Purpose |
| --- | --- |
| `README.md` | Protocol, results, and decision record |
| `audit_counts.py` | Reproducible telemetry/flux/count comparison |
| `compression.py` | Experiment-local implementation of documented counter coding, if needed |
| `sample_manifest.csv` | Frozen benchmark/control bins and metadata |
| `count_audit.csv` | Per-bin reconstruction and uncertainty comparison |
| `count_audit.png` | Compact visual comparison across count regimes |
| `sources.md` | Exact instrument-document passages, locations, and unresolved ambiguities |

`response_audit.csv`, `infer_response.py`, and `fragment_layout_audit.py` were
added when the preregistered counter audit exposed a telemetry-version question.
The planned figure was not produced because the decisive checks are scalar and
fully represented in the tracked CSV tables.

## Execution Record

### Frozen Provenance And Sample Gate

- Exact PDS products, byte sizes, hashes, and documentation locations are in
  `sources.md`.
- `compression.py` implements lower-endpoint decoding and represented count
  intervals, including an open upper bound for undocumented `0xff` saturation.
- `raw_telemetry.py` reconstructs 179 complete Format-3 blocks from the 7,285
  records in `M9911914.B`; every accepted block contains exactly 1,320 bytes.
- `freeze_manifest.py` produced the committed 16-bin `sample_manifest.csv`
  without reading raw telemetry.

### Telemetry Version And Crosswalk

April 1999 uses telemetry Format 3, introduced in May 1998. Format 1 and Format
3 are identical only through 3-D byte 278. The initial Format-1 transcription
therefore failed sharply at frame type 9. After applying the governing Format-3
table, every documented fragment offset is the empirical optimum in a blinded
temporal-layout check; median fragment correlations are generally 0.90-1.00.

The first complete raw block aligns with Level-1 spectrum 630. The streams then
advance at the same 80-second cadence, placing block 23 at spectrum 653. Against
adjacent blocks 20-26, block 23 has the strongest agreement with spectrum 653:

| Metric | Block 23 / spectrum 653 |
| --- | ---: |
| Log-count/flux correlation | 0.9742 |
| Rank correlation | 0.9997 |
| First VCDU | 3849472 |

A 30-spectrum per-bin lag test independently peaks at zero lag. This establishes
the benchmark crosswalk without choosing the block that gives the most favorable
count interval matches.

### Exposure And Information Scale

The angular-only hypothesis frozen before raw inspection used the basic
`spin_period / 256` accumulation and angular multiplicity, giving
`spin_period / (16 N_azimuth)`. It omitted that each energy receives only about
one sixteenth of the sweep. The ideal per-energy/per-direction scale is instead

```text
dt_bin ~= spin_period / (256 * N_azimuth).
```

The archive states that the exact integration time also depends on energy. Raw
block 23 empirically confirms the scale: the production reconstruction is a
median 120.9 times the decoded raw lower endpoint across positive bins, close to
the ideal factor 128 between `2.5/N_azimuth` and the duty-cycle expression.

This is a much larger error than the initial factor-of-eight diagnosis. As a
first-order information-scale check, dividing Experiment 001's reduced deviance
of about 1,421 by 120.9 gives about 11.8. Correct counts should therefore improve
Cash dramatically, but count scale alone still cannot make the current forward
model statistically adequate.

### Calibration And Dead Time

A bin-specific effective response learned on 30 neighboring spectra, excluding
653, is highly stable and available for 99.85% of bins. Two dead-time inversions
were compared:

| Flux inversion | Median response relative MAD | Spectrum-653 interval recovery |
| --- | ---: | ---: |
| Per-bin `tau = 0.3 microsecond` | 0.3386% | 22.53% |
| Dead time retained in effective response | 0.0396% | 67.91% |

The per-bin correction is empirically rejected. Detector dead time is driven by
the total event stream, so the abbreviated PDS formula does not justify applying
the nonlinear inverse independently to each archived flux bin. Remaining misses
are concentrated where integer codes, ASCII flux precision, energy-dependent
exposure, spin timing, and compression boundaries matter most.

### Uncertainty Budget

- **Counting:** raw integer counts are available only as compression intervals,
  not exact calibrated-flux inverses.
- **Compression:** codes below `0x20` are exact; larger codes represent widening
  truncated intervals; `0xff` has no documented finite upper bound.
- **Prescaling:** the real-time prescale header governs pitch-angle products, not
  the 88-by-15 3-D counters.
- **Exposure/response:** strongly bin dependent; empirically stable over the
  local control interval but not fully documented by the archive.
- **Dead time:** approximately 0.3 microsecond and global-event-rate dependent;
  a per-bin inverse is not supported.
- **Background:** approximately 10 counts/s is retained in the archive and must
  enter the expected count model rather than be silently removed.
- **Model discrepancy:** remains substantial after the roughly 121-fold count
  correction and must be estimated separately from instrumental uncertainty.

### Final Decision

**Outcome C: Partial Recovery.** Original compressed count intervals are
recoverable for Format-3 telemetry when Level 0 is paired with Level 1. A local
effective response is also recoverable and stable enough to justify testing an
interval-censored count model in Experiment 003. The calibrated flux archive by
itself does not define exact independent Poisson observations, and the current
production Cash implementation is invalid because it uses the wrong exposure,
ignores compression intervals, and applies unsupported precision.

Experiment 003 must carry forward raw-code intervals, response uncertainty,
global dead-time uncertainty, background, and an explicit model-discrepancy
term. It must compare that candidate against conditional-shape and smooth-Lillis
alternatives rather than assuming corrected counts alone solve the benchmark.

## Reproduction

```bash
.venv/bin/python experiments/experiment-002/raw_telemetry.py \
  artifacts/experiments/experiment-002/source/M9911914.B --format 3
.venv/bin/python experiments/experiment-002/fragment_layout_audit.py \
  --telemetry artifacts/experiments/experiment-002/source/M9911914.B \
  --data data/1999/091_120APR/3D990429.TAB --format 3
.venv/bin/python experiments/experiment-002/audit_counts.py \
  --telemetry artifacts/experiments/experiment-002/source/M9911914.B \
  --data data/1999/091_120APR/3D990429.TAB --theta data/theta.tab \
  --manifest experiments/experiment-002/sample_manifest.csv \
  --output experiments/experiment-002/count_audit.csv
.venv/bin/python experiments/experiment-002/infer_response.py \
  --telemetry artifacts/experiments/experiment-002/source/M9911914.B \
  --data data/1999/091_120APR/3D990429.TAB \
  --output experiments/experiment-002/response_audit.csv \
  --dead-time-seconds 0
```

## Protocol Deviations

1. The first parser used Format 1 from the general SIS. The empirical layout
   discontinuity at byte 279 led to the governing 1999 Format-3 table; the
   corrected decoder was then rerun from the frozen manifest.
2. The initial exposure audit stopped at angular multiplicity and reported an
   eight-fold production error. The counter comparison exposed the omitted
   energy-sweep duty cycle, revising the empirical factor to 120.9.
3. A response-stability holdout was added to determine whether undocumented
   calibration prevented all count recovery. Spectrum 653 remained excluded
   from response estimation.

## Decision Gate Answers

1. Each archived value is dead-time-corrected differential flux derived from a
   compressed 3-D counter, with background retained.
2. The measurement is a half-spin 3-D snapshot sampled every 16 spins. Per-bin
   exposure includes angular multiplicity and energy-sweep duty cycle; the exact
   energy-dependent table is not archived.
3. The original counter interval is exactly recoverable from paired Format-3
   Level-0 telemetry. Flux-only recovery is approximate, reaching 67.9% exact
   held-out interval agreement with a locally learned response.
4. Log compression, saturation, global dead time, response calibration, ASCII
   quantization, retained background, and normalization all prevent treating
   calibrated flux as an exact independent Poisson draw.
5. A raw interval-count likelihood is scientifically testable for paired,
   version-identified telemetry with response uncertainty. It is not justified
   as an archive-wide calibrated-flux default.
6. Experiment 003 must propagate count intervals, response and global-dead-time
   uncertainty, background, and explicit forward-model discrepancy.

The experiment downloaded the minimum Level-0 and sun-pulse products from PDS,
froze 16 benchmark bins before raw comparison, and avoided mission-scale
reprocessing. Instrument-team or PDS-curator confirmation would still be useful
before promoting a response model beyond the local Format-3 interval.

## Non-Goals

- Selecting the final loss-cone likelihood.
- Tuning an overdispersion parameter.
- Recomputing a full day or mission archive.
- Claiming that a large reduced deviance proves the geometric factor is wrong.
- Reimplementing the complete PDS calibration pipeline unless necessary to
  answer the count-validity question.
