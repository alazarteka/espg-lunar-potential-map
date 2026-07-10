# Experiment 002 Source Ledger

This ledger fixes the exact archive products and the distinction between
documented facts, derived consequences, and unresolved assumptions. Downloaded
copies live under the ignored `artifacts/experiments/experiment-002/source/`
directory; checksums make the analysis reproducible without tracking binaries.

## Level-0 Telemetry

| Product | Role | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| [`M9911914.B`](https://pds.nasa.gov/data/lp-l-6-position-v1.0/lp_0016/merged/d11x/m9911914.b) | Merged raw telemetry, 1999-04-29 14:13-18:21 | 3,438,520 | `68676e664c7a98bbbd9930ef31c0452b58ef5c814302ba061bdfba064ecbc629` |
| [`M9911914.LBL`](https://pds.nasa.gov/data/lp-l-6-position-v1.0/lp_0016/merged/d11x/m9911914.lbl) | Fixed-record layout and coverage | 9,634 | `808143881ce5e51612e4a8ea2bf8b9c1a5a07f4682059ee4ed6aca8981ce6b73` |
| [`MAGERFMT.PDF`](https://pds.nasa.gov/data/lp-l-6-position-v1.0/lp_0016/document/magerfmt.pdf) | Governing 1999 MAG/ER packet format | 120,170 | `3dbdd3abe815075e33dc75ee84abf01572d9cf743a2b79a313f202a542b0cb7c` |
| [`LVL0SIS.PDF`](https://pds.nasa.gov/data/lp-l-6-position-v1.0/lp_0016/document/lvl0sis.pdf) | Merged archive structure and timing | 334,261 | `25aad86222ba4773ecf6c488ed3b815984f5d4465cc9745899fcf34198322aed` |

The merged file contains 7,285 records of 472 bytes. Each record contains one
168-byte MAG/ER packet at one-indexed bytes 287-454. Real-time ER frame types
0-35 collectively transmit one 1,320-byte array: 15 energies by 88 directions.
The exact Format-1 fragments are transcribed in `raw_telemetry.py`.

The matching sun-pulse ancillary is `S9911914.B`/`.LBL`. It is required for an
independent spin-phase and look-direction reconstruction and has not yet been
downloaded. Its absence does not prevent byte-block reconstruction, but it does
prevent a final timing crosswalk.

## Level-1 Calibrated Product

| Product | Role | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `data/1999/091_120APR/3D990429.TAB` | Local calibrated 3-D differential flux | 25,177,575 | `de8358ba1456c2777e45271a1b193f95f11e866b10ae7d09d136498efc97f702` |
| [`3D990429.LBL`](https://pds-ppi.igpp.ucla.edu/archive1/LP_1204/DATA/ER_3DIST/99_04APR/3D990429.LBL) | Table schema and time semantics | 14,400 | `121cb1c9e700eed02c8210966026e7d1f52f2d1fdf5c0f31912a4b03badd113b` |
| [`ER3D_DS.CAT`](https://pds-ppi.igpp.ucla.edu/archive1/LP_1204/CATALOG/ER3D_DS.CAT) | Calibration chain and caveats | 27,520 | `be65b98dfa12c96fe0ef2fd07a72d933efd5598416669678ad01ee835876742b` |

Spectrum 653 has one shared native spectrum time, Unix `925397048`
(1999-04-29 14:44:08 UTC). The displayed times on its 15 energy rows are
synthetically incremented by an approximate spin period. They do not describe a
70-second physical accumulation.

The direction table `data/theta.tab` has SHA-256
`395202a7237f65253072440908226467bcbae66f5e7247f5b46e39e1489cd005`.

## Documented Acquisition And Calibration Chain

The MAG/ER SIS Appendix G documents 32 analyzer sweeps per spin, two sweeps
combined per 22.5-degree phase sector, and a 3-D sample acquired during one
half-spin and sampled once per 16 spins. Section 2.3.6 gives the basic exposure

```text
t_basic = spin_period / 256.
```

Ordinary cells receive one basic accumulation; polar map cells combine two or
four. For the archive's latitude-band populations `N_azimuth = {16, 8, 4}`,
the documented nominal relation is

```text
multiplicity = 16 / N_azimuth
t_bin = spin_period / (16 * N_azimuth).
```

At a five-second spin this gives 0.01953125, 0.0390625, or 0.078125 seconds.
The current production assumption `2.5 / N_azimuth` is uniformly eight times
larger. This invalidates the current count scale but cannot alone explain a
benchmark reduced deviance near 1,421; a coherent factor-of-eight reduction
would still leave it near 178.

The PDS data-set catalog documents the ground chain:

1. Decompress each logarithmic counter byte.
2. Divide by an integration time depending on energy and solid angle.
3. Apply nonparalyzable dead-time correction with approximately 0.3 microsecond
   dead time: `r_corrected = r / (1 - r * tau)`.
4. Mask values requiring a correction factor above 1.25.
5. Leave the approximately 10 counts/s background unsubtracted.
6. Divide by `0.02 cm^2 sr * E_center` to obtain differential flux.

Consequently, a nominal flux-only inverse must first use

```text
r_corrected = 0.02 * E * F
r = r_corrected / (1 + tau * r_corrected)
C_proxy = r * t_bin
```

and then test `C_proxy` against the raw byte's compression interval. The
existing `F * 0.02 * E * dt` calculation instead produces a dead-time-corrected
count equivalent and treats it as an exact independent Poisson observation.

## Compression And Prescaling

Appendix G section 2.6 defines an 8-bit code with a four-bit exponent and
four-bit mantissa. Truncation, rather than rounding, means decompression gives
the lower endpoint of the represented original-count interval. The experiment's
`compression.py` implements that table and leaves code `0xff` open above because
the documentation does not establish its saturation endpoint.

The real-time header's prescale bits govern long pitch-angle accumulations. The
SIS does not assign them to the fixed-sector 88-by-15 3-D counters. Experiment
002 therefore applies no 3-D prescale factor absent contrary authoritative
evidence.

## Unresolved Instrument Information

- The catalog says 3-D integration time depends on energy as well as solid
  angle, while the telemetry SIS establishes angular multiplicity but does not
  provide a complete energy-dependent exposure table.
- The exact dead time is given only approximately.
- Detailed anode-efficiency calibration is described as substantial but its
  table/arithmetic is not supplied in the archive documentation.
- The energy-control-table state is not completely present in the four-byte
  real-time header.
- Pairing spectrum 653 to one raw block requires the documented `TAcc`
  calculation, including spin phase, half-spin index, readout latency, and the
  ground direction correction. Timestamp proximity alone is not sufficient.

These gaps are why a flux-only inversion is a diagnostic proxy, not yet a
formal recovered-count likelihood.
