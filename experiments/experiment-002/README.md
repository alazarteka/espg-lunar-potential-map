# Experiment 002: LP ER Count Reconstruction Audit

**Status:** Running

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
selection script does not read Level-0 telemetry.

## Authoritative Sources

- LP MAG/ER telemetry format and counter compression:
  <https://pds-ppi.igpp.ucla.edu/volume/LP_1201/DOCUMENT/PROJECT/LVL0SIS.PDF>
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

## Execution Record

### Frozen Provenance And Sample Gate

- Exact PDS products, byte sizes, hashes, and documentation locations are in
  `sources.md`.
- `compression.py` implements lower-endpoint decoding and represented count
  intervals, including an open upper bound for undocumented `0xff` saturation.
- `raw_telemetry.py` reconstructs 179 complete Format-1 blocks from the 7,285
  records in `M9911914.B`; every accepted block contains exactly 1,320 bytes.
- `freeze_manifest.py` produced the committed 16-bin `sample_manifest.csv`
  without reading raw telemetry.

### Exposure Finding

The documented nominal exposure is `spin_period / 256` per basic accumulation,
with multiplicity one, two, or four for the archive angular cells. At a
five-second spin, the resulting 0.01953125, 0.0390625, and 0.078125 second
exposures are uniformly eight times below the current production values.

This invalidates the current reconstructed-count scale. It does not by itself
explain Experiment 001: a coherent factor-of-eight reduction would lower a
reduced deviance near 1,421 only to roughly 178.

### Provisional Outcome

Evidence currently favors **Outcome C: Partial Recovery**. Raw compressed
counter intervals are recoverable from Level-0 telemetry, but a formal pairing
to Level-1 spectrum 653 still requires the instrument timing crosswalk. Flux
alone cannot currently recover exact independent Poisson observations because
the archived processing includes dead-time correction, compression intervals,
uncertain energy-dependent exposure, and incompletely documented response
calibration.

Files will be added only when they have real content; these names define the
expected experiment record rather than empty placeholders.

## Decision Gate

Experiment 002 is complete only when all of the following are answered:

1. What physical quantity does each archived flux number represent?
2. What accumulation/exposure produced it?
3. Can an original counter value or interval be recovered?
4. Which corrections make it non-Poisson or correlated?
5. Is a count likelihood valid for all bins, a restricted subset, or none?
6. What uncertainty information must Experiment 003 carry forward?

The result must select Outcome A, B, or C with documented evidence. It must not
tune a new loss-cone objective or acceptance threshold.

## Discussion Decisions Before Execution

Before starting, decide:

- Whether Level 0/1 telemetry and header metadata are locally available or must
  be downloaded from PDS.
- Whether exact reconstruction of spectrum 653 is feasible from available raw
  products, or whether the primary deliverable is a defensible uncertainty
  bound on the calibrated product.
- How many frozen control bins are sufficient to cover telemetry modes without
  turning this into an instrument reprocessing project.
- Whether assistance from a Lunar Prospector ER instrument expert or PDS
  curator should be sought for undocumented exposure/prescaling details.

## Non-Goals

- Selecting the final loss-cone likelihood.
- Tuning an overdispersion parameter.
- Recomputing a full day or mission archive.
- Claiming that a large reduced deviance proves the geometric factor is wrong.
- Reimplementing the complete PDS calibration pipeline unless necessary to
  answer the count-validity question.
