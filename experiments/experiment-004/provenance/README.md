# Provenance

`halekas-cleanroom-reconstruction.md` is a byte-for-byte snapshot of
`scratch/halekas-cleanroom/reconstruction.md` taken on 2026-07-11 before
Experiment 004 implementation began.

- SHA-256: `b641f84864ba6d8a81b922335e780c09b3e0c9887f0473151c63031ab1ab275f`
- Source status: clean-room synthesis from the Halekas et al. (2008) paper,
  instrument documentation, retrieved lineage papers, figure forensics, and
  clearly labeled lower-confidence inferences.
- Purpose: freeze the evidentiary basis used to preregister the Experiment 004
  historical reconstruction.

The snapshot is evidence, not executable specification. Publicly undocumented
choices remain experimental factors, especially beam amplitude handling,
normalization, objective weighting, and exact grid bounds.

## Figure 5 Data Panel

Phase A4 adds two byte-for-byte derived artifacts from the clean-room Figure 5
forensics:

- `figure5-data-values.npy`: color-calibrated data-panel values, shape `89 x 44`.
  SHA-256: `4a33f6d3c0d5686c2a6a2d727660ffab38f310fc4bdf3fb5018d38181df697ec`.
- `figure5-data-native.png`: native embedded data-panel raster, shape `89 x 44`.
  SHA-256: `10386337e763b5113766cfa7967628997dfbf431cdd9ad31fd680d02a9fd1dbe`.

Both were extracted from Figure 5 of the archived Halekas et al. (2008) PDF.
The calibrated values use the panel color bar and retain the display's
`0.1-10` clipping; white pixels in the native raster identify plotted gaps.
