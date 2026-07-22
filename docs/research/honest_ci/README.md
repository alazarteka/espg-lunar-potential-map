# Honest confidence-set research record

**Status:** active research record, 2026-07-22. No document in this directory
establishes an operational confidence interval for Lunar Prospector ER surface
potential.

This directory preserves the complete GPT Pro research exchange that examined
whether assumptions weaker than independent Poisson counting can support an
honest frequentist confidence set for spacecraft-relative lunar surface
potential. The imported model outputs are preserved verbatim. The Codex
reconciliation records which conclusions survive independent comparison with
the current branch, draft PR 14, the reference telemetry, and primary sources.

## Reading order

1. [`research_prompt.md`](research_prompt.md) — original research assignment.
2. [`original_memo.md`](original_memo.md) — first GPT Pro solution.
3. [`adversarial_review.md`](adversarial_review.md) — independent mathematical
   referee report.
4. [`revised_memo.md`](revised_memo.md) — adjudicated and revised solution.
5. [`codex_reconciliation.md`](codex_reconciliation.md) — source-aware project
   disposition and corrections.

## Provenance

The first four files were imported without editing. Their SHA-256 hashes at
import were:

| File | SHA-256 |
|---|---|
| `research_prompt.md` | `354d4ad2e68140c1f7b3bce123a359a54ce9d6643541ca87207327c38a981fbe` |
| `original_memo.md` | `c7e9bfc8e85fd4d9e76a26fad88714ec1703fac4bb16de86134ffd2cc41056a9` |
| `adversarial_review.md` | `8f4298c1ce77648507835a19c48090daa3dd340191acbfc9760e76a9723e54bb` |
| `revised_memo.md` | `c00f3d4607311f4b96a266d1208d5ea57385fa43be2b2ec845fb45529bcde619` |

The research prompt referred to a 15 MB source bundle containing two Git
snapshots, public PDS documents, and raw/calibrated reference days. That ZIP is
not tracked because it duplicates repository history and mission data. The
relevant source material remains identified in
[`../../architecture/er_measurement_contract.md`](../../architecture/er_measurement_contract.md)
and in the prompt's source manifest.

## Interpretation rule

Treat the GPT memos as research arguments, not primary evidence. Instrument and
archive claims remain subordinate to the PDS sources. Mathematical claims are
conditional on their explicitly declared model classes. In particular:

- the intended current LP experiment is **unresolved (D)** because its complete
  observable law and response uncertainty class are not available;
- broader classes allowing arbitrary response, selection, mixing, or only code
  support have explicit **nonidentification (C)** results; and
- positive A/B results describe conditional mathematical routes whose premises
  have not been established for the present archive.
