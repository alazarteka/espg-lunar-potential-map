# Scientific Experiments

This directory contains version-controlled scientific experiments that inform
production methods and manuscript claims. Experiments are numbered in execution
order and preserve hypotheses, protocols, code, outputs, and decisions together.

## Naming and Branching

- Directory: `experiments/experiment-NNN/`
- Branch: `experiments`
- Commit: conventional commit describing the experiment or its result

Experiments accumulate on the shared `experiments` branch while they add only
self-contained protocols, scripts, and compact results. A production change
should be implemented on a separate source branch after the experiment's
decision has been reviewed; experimental code should not silently become a
production default.

## Required Record

Every experiment should record:

- Scientific question and competing hypotheses.
- Parent observation or decision that motivated the experiment.
- Repository commit and branch used.
- Frozen input data and selection rules.
- Controlled variables and planned comparisons.
- Exact reproduction commands.
- Machine-readable or inspectable outputs.
- Deviations from the preregistered protocol.
- Pass/fail or model-selection criteria.
- Final interpretation, limitations, and next decision.

Experiment code may use private repository APIs when clearly labeled. Results
that affect production behavior must later be implemented through normal source,
test, and documentation paths.

Large mission data remain under `data/` or `artifacts/` and should not be copied
into an experiment directory. Compact figures and tables that are necessary to
audit a scientific decision should be tracked even when their file type is
globally ignored.

## Status

| Experiment | Question | Status | Decision |
| --- | --- | --- | --- |
| [001](experiment-001/README.md) | Does current Cash reproduce the Halekas benchmark, and why does it fail? | Complete | Smooth Lillis approximately reproduces the benchmark; current pure-Poisson Cash fails. |
| [002](experiment-002/README.md) | Can the calibrated LP ER flux product support a raw-count Poisson likelihood? | Complete | Partial recovery: raw intervals are recoverable, but calibrated flux is not an exact Poisson observation. |
| [003](experiment-003/README.md) | Which observation/uncertainty model best preserves physics and calibrates uncertainty? | Complete | No candidate passes all gates; matched estimators work, but the shared forward model fails real-data validation. |
| [004](experiment-004/README.md) | Can a clean-room Halekas reconstruction recover the benchmark before modern extensions? | Complete | The beam scale is robust; the boundary does not independently identify the published voltage/field-ratio pair. |
| [005](experiment-005/README.md) | When are beam voltage and loss-cone corroboration defensible across one complete interval? | Complete | Attempt 03 validates 150-350 eV beams, but low/high-energy cells and 6% null acceptance fail the full gate; no real pilot. |
| [006](experiment-006/README.md) | Can a polarity-aware beam estimator support calibrated selective inference in a prospectively tested 100-500 eV reporting window? | Running | D0 froze disjoint 12/12 polarity-aware supports; implementation, calibration, and confirmation remain unopened. |

## Status Vocabulary

- **Planned:** protocol exists; execution has not begun.
- **Running:** code or data work is in progress.
- **Blocked:** a named upstream scientific decision is required.
- **Complete:** planned outputs and decision record exist.
- **Accepted:** result has been reviewed and may motivate production work.
- **Rejected:** approach failed its gate; the negative result remains preserved.
