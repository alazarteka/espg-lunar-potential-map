# Experiment 008 Replay: ECLIPJ2000 v1

## Status

Replay prepared on 2026-07-14. Scientific outcomes are not yet recorded in
this directory.

## Reason

The official Lunar Prospector attitude product expresses spin-axis right
ascension and declination in Mean-of-J2000 ecliptic coordinates. The original
Experiment 008 instead interpreted those components as equatorial `J2000`.
This replay regenerates every geometry-dependent artifact with both the
attitude and LP-to-Sun vector in SPICE frame `ECLIPJ2000`, followed by
`pxform("ECLIPJ2000", "IAU_MOON", et)`.

The original `results/`, supplement, and result manifest remain immutable.
Replay outputs are written here; large caches are isolated under
`artifacts/experiment-008/eclipj2000-v1/`.

## Frozen Scope

- Use the original input manifest, configuration, Amendment 03 decision tree,
  basis conventions, seeds, windows, noise levels, and recovery gates.
- Replace only the canonical coordinate-frame interpretation.
- Preserve corrected-attitude interpolation and legacy-attitude sensitivity.
- Add an `equatorial` sensitivity reproducing the original Experiment 008
  canonical interpretation.
- Preserve `frame_mixed` and `submitted_legacy` as submitted-code controls.
- Re-run geometry, basis preparation, all primary and extended phases, plots,
  numerical validation, scientific synthesis, supplement, and manifest.
- Do not read fitted-potential archives.

## Execution

All commands set:

```bash
export EXPERIMENT_008_RUN_ID=eclipj2000-v1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
```

The dependency order remains geometry extraction and merge, spatial basis,
primary/grid/static analyses, temporal preparation and controls, sensitivities,
column normalization, injections, localized caps, blocked validation, plots,
scientific synthesis, supplement, and final manifest.

## Interpretation Rule

The original outcome is superseded only if the unchanged frozen decision tree
classifies the corrected replay differently. Similarity to the prior numerical
result is evidence of robustness, not a success criterion.

