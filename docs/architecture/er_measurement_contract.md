# ER Measurement Contract

**Status:** Phase 1 evidence record — 2026-07-20.  This document defines what
the project currently observes and what must be established before a new
spacecraft-relative potential (`ΔU`) likelihood is implemented.

## Scope and primary source

The current pipeline reads the PDS-calibrated Lunar Prospector ER 3-D electron
flux collection, `LP-L-ER-3-RDR-3DELEFLUX-80SEC-V1.1`, not Level-0 telemetry.
The archived collection spans 1998-01-16 through 1999-07-29.  The local checkout
contains 560 daily tables, one per calendar day in that span.

The authoritative product description is the [PDS ER 3-D data description]
(https://pds-ppi.igpp.ucla.edu/data/lp-er-calibrated/document/data-3deleflux-desc.txt).
The PDS bundle is [Lunar Prospector ER Calibrated]
(https://pds.nasa.gov/ds-view/pds/viewBundle.jsp?identifier=urn%3Anasa%3Apds%3Alp-er-calibrated&version=1.0).

### Evidence hierarchy and recovery result

This contract was expanded by an archive-wide documentation search on 2026-07-20.
Its evidence is deliberately ranked:

1. The calibrated-product descriptions and errata define the analysis input and
   its published processing history.
2. The [LP Science Interface Specification]
   (https://pds.nasa.gov/data/lp-l-6-position-v1.0/lp_0019/document/lvl0sis.pdf)
   and [MAG/ER telemetry format]
   (https://pds.nasa.gov/data/lp-l-6-position-v1.0/lp_0019/document/magerfmt.pdf)
   define the Level-0 packet, timing, compression, and nominal response
   behavior.
3. A peer-reviewed LP ER analysis confirms operational use of relative
   anode-sensitivity correction and the effective energy resolution; it does
   not publish the needed calibration table: [Halekas et al. (2012)]
   (https://ntrs.nasa.gov/api/citations/20140010167/downloads/20140010167.pdf).
4. The [derived electron-reflection product description]
   (https://pds-ppi.igpp.ucla.edu/data/lp-er-derived/document/data-electron-desc.txt)
   independently records that flux uncertainty includes compression digitization
   noise, while its omitted absolute geometric-factor uncertainty was estimated
   from electrostatic-optics simulations with grid-transmission and MCP-efficiency
   corrections.

The public calibrated-document directory contains the three data descriptions,
errata, and collection metadata, but no response/efficiency table or processing
software. The Level-0 SIS explicitly marks the per-anode efficiency table and
the detailed on-board phase-space arithmetic as `TBS`. Therefore they must be
treated as missing source inputs, rather than inferred from a nominal geometric
factor or from a rounded calibrated flux.

The [low-resolution]
(https://pds-ppi.igpp.ucla.edu/data/lp-er-calibrated/document/data-lowresflux-desc.txt)
and [high-resolution]
(https://pds-ppi.igpp.ucla.edu/data/lp-er-calibrated/document/data-highresflux-desc.txt)
descriptions add two public calibration targets: 15-channel, 80-second
omnidirectional flux and two-channel, half-spin flux. They also identify
spacecraft command logs as ancillary data. These products may validate a future
emulator's timing and aggregate flux behavior, but cannot identify a 256-anode
flat-field after their angular information has been summed away.

## What one calibrated sweep contains

One PDS table row has 183 fields:

| Field | Shape | Present in local schema | Meaning |
|---|---:|---|---|
| UTC and epoch time | scalar | yes | Time of one energy-row measurement |
| Energy | scalar | yes | Center energy for the row |
| Spectrum number | scalar | yes | Groups 15 consecutive energy rows into one ER 3-D sweep |
| Magnetic field | 3 | yes | SCD-frame average during the accumulation |
| Electron flux | 88 | yes | Calibrated differential flux per angular cell |
| Cell longitude | 88 | yes | Energy-row-specific SCD azimuths |
| Cell latitude | 88 | ancillary `theta.tab` | Fixed SCD inclinations |

The detector samples 88 solid-angle cells spanning approximately `4π` and 15
logarithmically spaced energy channels. A full 3-D sweep is sampled every 16
spins (about 80 s); each energy row is accumulated for roughly half a spin
(about 2.5 s).  The archive states that the effective exposure varies with both
energy and solid angle.  Therefore the current approximation `2.5 s / N_azimuth`
is not an evidence-backed cell-exposure model.

The mission used both a roughly 40 eV–20 keV sweep and a roughly 7 eV–20 keV
sweep. A new inference path must retain each sweep's recorded energy centers;
it must not assume a universal energy grid.

The nominal intrinsic energy resolution is `ΔE/E ≈ 0.25`. The on-board
processor combines adjacent bins, so the operational 15-channel data used in
the potential literature have effective `ΔE/E ≈ 0.5`. A response-folded model
must integrate over the stated, sweep-specific energy bands; a point evaluation
at a universal energy centre is an approximation to be disclosed and tested.

## Processing history and its statistical consequence

The calibrated flux is downstream of the following chain:

1. Onboard raw counts are logarithmically compressed.
2. Ground processing decompresses them and divides by integration time.
3. A non-extendable dead-time correction is applied; values with correction
   factor above 1.25 are masked by the processing.
4. The product retains an approximately 10 count/s instrumental/cosmic-ray
   background; it is not subtracted.
5. The result is divided by geometric factor and energy to yield differential
   flux.

Consequences for this project:

- `src/losscone/er_data.py` reconstructs and rounds *pseudo-counts* from
  calibrated flux, then sums the 88 cells. These columns are derived helpers,
  not archived detector counts and not valid inputs to a per-cell count
  likelihood.
- The calibrated 183-column table has no raw compression code, raw count,
  integration-time, or explicit dead-time-mask column. A zero flux value cannot
  be interpreted automatically as an observed zero count.
- The current per-energy normalization in `src/losscone/normalization.py`
  creates shared-denominator dependence. It is an optimization transform, not a
  likelihood observation model.
- Spacecraft-potential effects are not corrected in the calibrated PDS product;
  at low energy they can shift energy and degrade imaging. This is a forward
  model/regime-gate issue, not a calibration uncertainty that can be ignored.

### Documented quality states that must not become fit masks

The new product schema must preserve these separately from the physical
loss-cone gate:

| State | Archive evidence | Required treatment |
|---|---|---|
| Missing calibrated datum | Flux sentinel is `9.999E-99`; PDS errata also identify corrected invalid constants | Mark missing; never turn into a tiny positive flux or a zero count |
| Saturation / dead-time limit | The correction is only trusted through 1.25; calibrated rows do not retain that source flag | Retain as unrecoverable provenance on calibrated inputs; recover and validate before any raw-code use |
| Direct-Sun aperture contamination | Occurs twice per spin; it makes a roughly 14° wide sunward false-electron feature | Use an explicit geometric contamination gate or an explicitly fitted component |
| Low-energy spacecraft-potential distortion | PDS says energies shift and imaging degrades at energies comparable to the potential | Require a physics/energy-leverage gate, not a generic residual cut |
| Telemetry or processing gap | The Level-0 archive publishes outage information; calibrated descriptions allow other gaps | Preserve source-file and sweep identifiers and an archive-status reason |

The archived, post-processed SCD look directions are the observation geometry.
In particular, shadow-driven spin-up is reconstructed on the ground, and the
PDS documentation says the per-energy cell azimuths change during the sweep.
New code must use those archived directions rather than recreate them from a
fixed spin cadence.

## Raw-telemetry availability

PDS also publishes the Lunar Prospector Level-0 archive. Its merged telemetry
files are binary, have detached labels, contain raw science plus engineering
telemetry, and are stored in 4–6 hour files. The [Level-0 archive SIS]
(https://pds-ppi.igpp.ucla.edu/archive1/LP_0010/DOCUMENT/VOLSIS.HTM) and the
[Level-0 archive description]
(https://pds-ppi.igpp.ucla.edu/volume/LP_0010/AAREADME.HTM) identify the
`MERGED/` files and their labels. The archive also includes outage information.

**Project status:** raw telemetry is publicly available. This repository has a
narrow format-1/2 packet-assembly and code-decompression probe, but no
mission-wide downloader/label parser, event-time reconstruction, configuration
decoder, calibrated-product conversion, or cross-check. It must therefore be
treated as *available but unusable as an inference input today*.

### Feasibility-spike evidence

The first reference candidate has been located in the Level-0 index:

| Item | Value |
|---|---|
| Level-0 file | `LP_0001/MERGED/D01X/M9801523.B` |
| Labelled Earth-received interval | 1998-01-15 23:43 to 1998-01-16 03:38 UTC |
| Calibrated reference sweep time | 1998-01-16 00:00:24 UTC |
| Fixed record layout | 7,047 records × 472 bytes |
| Downloaded-file size | 3,326,184 bytes (matches the label) |
| Downloaded-file SHA-256 | `5a83f8e8b8c39bdc213e8673a8ee1e0a34d0a2a2c21dcfffe0364f9fc02aff3b` |
| MAG/ER payload per record | 168 bytes |

This establishes access to a small candidate raw product and its detached label.
It does **not** establish a packet-to-sweep match: Level-0 records are tagged by
Earth-received time, while the calibrated table is tagged by reconstructed event
time. The decoder must first recover the telemetry frame/major-frame sequence,
the ER packet layout, compressed counts, and relevant control state. Only then
can it align and compare a decoded 15×88 sweep with the calibrated anchor.

The first executable probe is `src/losscone/level0.py`, exposed by
`scripts/diagnostics/level0_er3d_probe.py`. It verifies only the documented
format-1/2 frame assembly and counter-code lower endpoints. It accepts a
sequence only when its Digital Subcom software version is consistently 1 or 2;
it explicitly rejects version 3, whose 3-D bytes use a different layout, plus
missing or mixed version evidence. It has no calibrated-flux conversion and
must remain a Phase-1 probe until the packet-to-product cross-check succeeds.

**Spike result and decision:** the probe successfully extracts complete
compressed 15×88 arrays from `M9801523.B`; the first assembled sweep begins at
Level-0 record 39, has codes from 0 to 137, and decompresses to lower endpoints
from 0 to 3,200 counts. This proves that raw code extraction is feasible. It
does not establish a usable count likelihood, however. The same telemetry-format
document leaves the per-anode efficiency table and the exact on-board
phase-space/arithmetic procedure as `TBS`; both are needed to recover trusted
per-cell exposure/geometric response and to reproduce calibrated flux.

Therefore the project will **not** call raw-code likelihood inference feasible
for Phase 4. The working Phase-4 route is an explicitly conditional
calibrated-flux likelihood (or quasi-likelihood) with full-chain parametric
bootstrap. It must not be called Poisson or multinomial count likelihood. A
future raw-code path requires an independently sourced efficiency/response table
and a decoded-to-calibrated reference-sweep cross-check.

### What Level-0 adds, and why it remains insufficient

The Level-0 documents make the raw path more precisely specified, not yet
usable:

- ER is spin-synchronous (nominally 12 rpm). A real-time 3-D product is spread
  over 36 ER frame types after a 16-spin accumulation. Event time is
  reconstructible from major-frame time, spin phase, spin period, and header
  state; the raw record time is Earth-received time, so light time and delayed
  playback latency still need to be applied and validated against a calibrated
  timestamp.
- Each raw sample is an 8-bit logarithmic code: 4-bit exponent plus 4-bit
  hidden-bit mantissa. Both normal and pre-scaled values are truncated, so the
  documented decoded value is the *lower endpoint* of a count interval, not an
  observed count. A future code-space model must use interval probabilities.
- Pitch-angle accumulations can be pre-scaled by eight energy bands, either by
  command or after a prior accumulation saturates above 524,287 counts. Header
  bits record that state. The 3-D frame layout also has format/version and
  configuration dependencies; the current format-1/2 assembly probe is not a
  mission-wide decoder.
- Total and rate counters diagnose losses from pulse-height rejection, about
  300 ns pulse-processing dead time, FIFO overflow, and anode masks. They do
  not recover the cell-level losses or response needed for a likelihood.
- The SIS gives an approximate full-instrument geometric factor,
  `0.07 E_keV cm² sr keV`, over 256 effective anodes. It says the per-anode
  response varies significantly through ADC nonlinearity and MCP gain. The
  required efficiency factors are the missing `TBS` table. The calibrated
  3-D product instead describes a `0.02 cm² sr` product-level conversion.
  These values are not interchangeable response models.

Consequently, the present raw extractor establishes **code extraction only**.
It does not establish raw counts, per-cell exposure, response, timing, or a
calibrated-flux reconstruction.

### Phase-1b coarse association result

`src/losscone/raw_calibration.py` and
`scripts/diagnostics/level0_calibrated_match.py` now provide the first
response-independent raw-to-calibrated association harness. It reads the
calibrated 3-D table directly (without loader cleaning or pseudo-count creation),
retains each 15-row sweep's UTC span, decodes Level-0 Earth-receive time, and
ranks only time-near pairs by per-energy, scale-free angular log correlation.
It writes a provenance report with hashes and every candidate; it never writes
raw matrices by default.

The initial run used `M9801523.B` and `3D980116.TAB`. Its structural frame
assembly found 166 complete raw sequences; applying the Digital Subcom
format/version gate accepts 165 format-1/2 sequences. It found 1,032 complete
calibrated sweeps and 2,084 time-near pairs in a ±600 s search. Of 159 raw
sweeps with at least one candidate, 147
had their highest angular-shape score at an Earth-receive time 55–70 s before
the end of the corresponding calibrated energy sweep. The median offset was
`-63.716 s`; the median best score was `0.659`, and the highest was `0.842`.

This is strong evidence of a consistent *coarse association* and of compatible
3-D ordering. It is not an event-time reconstruction: the reported offset still
contains downlink/instrument latency and must be explained by decoding the ER
header plus Digital Subcom phase/period state. It also says nothing about
absolute response, cell exposure, dead-time correction, or raw-count statistics.

An additional public-archive search found no anode-efficiency table, MAG/ER
Commanding document, exact phase-space arithmetic reference, or calibration
software. This is consistent with the `TBS` placeholders in the source SIS, not
merely a missing local copy. The next raw milestone is therefore state and timing
provenance (header, Digital Subcom, command history, and detached label), followed
by a held-out association test. A response reconstruction remains conditional on
external calibration artifacts or a separately validated empirical emulator.

### Three deliberately distinct recovery claims

The project must use these labels exactly:

| Claim | Current status | Minimum evidence required |
|---|---|---|
| Structural association | Supported for the Phase-1b reference run | Complete raw frames, source hashes, and a ranked calibrated-sweep candidate report |
| Empirical archive emulator | Plausible but not implemented | State-aware raw-code inputs, held-out 3-D recovery, and independent aggregate checks against the public low- and high-resolution products |
| Physical detector response / count likelihood | Unsupported | Published or independently recovered anode response, exposure, control-state, and onboard-arithmetic inputs, followed by calibrated-product reproduction |

An empirical archive emulator predicts the archived calibrated products from
public raw telemetry and public state variables. It is not a first-principles
instrument-response model and any inference built on it must be called an
**archive-emulation likelihood**, not a Poisson, multinomial, or detector-count
likelihood. The derived-product documentation's compression-noise and
electrostatic-optics statements confirm that the missing response structure was
material to the historical calibration; they do not supply its coefficients.

## Observation-model decision

No new code may call the existing `count` or `count_err` helpers raw telemetry.
Phase 3 may build a deterministic response-folded mean model from calibrated
flux and geometry. Phase 4's likelihood choice is gated by a short Level-0
feasibility spike:

1. Download one labelled Level-0 merged telemetry file and its associated
   documentation.
2. Identify and decode ER packets for one calibrated reference sweep.
3. Reproduce its 15×88 calibrated product closely enough to account for
   compression intervals, exposure, pre-scale/control state, dead time,
   background, anode response, timing, and geometric conversion.
4. Record which per-cell quantities are recoverable without undocumented
   assumptions.

If this succeeds, a code-level telemetry decoder becomes a prerequisite for the
count/code likelihood. If it fails or shows unrecoverable inputs, Phase 4 must
use an explicitly conditional calibrated-flux quasi-likelihood and bootstrap the
same calibrated-data process. It must not call that construction Poisson or
multinomial count likelihood.

## Selection, quality, and independence contract

The following are separate decisions and must be recorded per sweep in the new
batch schema:

| Decision | Current status | Requirement for new inference |
|---|---|---|
| Instrumental validity | Present loader removes invalid time/B sweeps | Preserve reason codes; add source-backed dead-time/sunlight handling when recoverable |
| Magnetic connection/polarity | Computed downstream | Mandatory pre-fit physical gate |
| Forward-model validity | Not represented | Record adiabaticity, energy leverage, and beam-edge consistency outcomes |
| Detection/selection | Legacy normalized-flux masks | Model or report the selection rule; never hide it in a fit mask |
| Statistical unit | 15 rows are one sweep | Treat the sweep/visit as the correlated unit, not 15 independent observations |

## Reference-sweep anchor

[`tests/fixtures/er3d_19980116_spec_0001.json`](../../tests/fixtures/er3d_19980116_spec_0001.json)
names an unmodified calibrated table and the first complete 15-row spectrum.
The manifest records the source-file SHA-256 and structural expectations. It does
not commit mission data into Git; a test verifies it only when the local PDS data
are available. Future Level-0 decoding and response integration must reproduce
or explicitly explain this anchor before scaling to other data.

## Phase-1 exit criteria

- This contract and reference anchor remain valid against a fresh local data
  download.
- The Level-0 feasibility spike has a written result: raw-code extraction works,
  but a count likelihood is not accepted without the missing response inputs,
  interval-censoring treatment, and a calibrated reconstruction anchor.
- The Phase-2 `ΔU` schema names the observation level and selection state rather
  than inheriting legacy `U_surface` semantics.
