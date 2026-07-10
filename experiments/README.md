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
| [002](experiment-002/README.md) | Can the calibrated LP ER flux product support a raw-count Poisson likelihood? | Running | Pending |
| [003](experiment-003/README.md) | Which observation/uncertainty model best preserves physics and calibrates uncertainty? | Blocked by 002 | Pending |
| [004](experiment-004/README.md) | How sensitive is retrieval to the instrument-response and fixed-shape assumptions? | Blocked by 003 | Pending |
| [005](experiment-005/README.md) | Does the frozen method pass a complete one-day scientific validation pilot? | Blocked by 003-004 | Pending |

## Status Vocabulary

- **Planned:** protocol exists; execution has not begun.
- **Running:** code or data work is in progress.
- **Blocked:** a named upstream scientific decision is required.
- **Complete:** planned outputs and decision record exist.
- **Accepted:** result has been reviewed and may motivate production work.
- **Rejected:** approach failed its gate; the negative result remains preserved.
