# r>=0.3 bound artifact check (Priority 0)

**Date:** 2026-07-20
**Script:** `scripts/diagnostics/r_bound_artifact_check.py`
**Raw JSON:** `artifacts/diagnostics/r_bound_artifact_check.json`

## Verdict

**REFUTES** the "~100 V is an r-bound artifact" hypothesis **for the current
normalized forward model + Halekas-style objective**: weak-mirror truth-zero
injections recover a median near zero (about -14 V with the default floor;
about +1 V with the floor removed), with `frac_u_in_80_120 = 0` in both panels.

This does **not** prove the archive median near -108 V is physical — only that
the linearized fresh-look mechanism does not reproduce under this synthetic
chain. Archive `r-hat` census (A2) still requires a batch NPZ:

```bash
uv run python scripts/diagnostics/r_bound_artifact_check.py \
  --batch-npz artifacts/potential_cache/.../potential_batch_....npz
```

## Implication for D2

- Keep legacy `LOSS_CONE_BS_OVER_BM_MIN = 0.3` on Halekas/Lillis paths (unchanged).
- D2 profile-CI path uses `PROFILE_CI_R_MIN = 0.02` (no identifying floor).
- Removing the legacy floor is a **sensitivity study**, not a required CI PR
  coupling, until A2 shows archive `r-hat` pinned at 0.3.

## Re-run

```bash
uv run python scripts/diagnostics/r_bound_artifact_check.py --n-per-cell 4
```
