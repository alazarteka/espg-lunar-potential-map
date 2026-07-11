# Scientific Implications of the Halekas Clean-Room Reconstruction

**Date:** 2026-07-11

**Status:** Frozen interpretation before Experiment 004 results

## Central Conclusion

The project has localized the correct Halekas et al. (2008) benchmark sweep
and approximately recovered its voltage with the smooth Lillis estimator, but
it has not yet reproduced the historical Halekas retrieval. The current
estimator is a hybrid of two distinct toolchain branches.

The reconstructed Halekas estimator uses a sharp loss-cone boundary, a finite
interior floor, the complete normalized distribution in log space, low-energy
channels, a response-limited field-aligned secondary beam, 16 native 11.25
degree pitch bins, a per-measurement spacecraft potential, and strong quality
control. Experiment 003 instead used the 88 angular sectors directly, a smooth
boundary, a fixed 15 eV Gaussian beam, the Lillis relative-flux mask, and a
fixed `U_sc = +11 V` across its panel.

Agreement on spectrum 653 is therefore a positive control, not evidence that
the current production forward model is historically or physically correct.
Compensating model choices can recover one voltage while diverging elsewhere.

## Scientific Meaning of the Beam

The upward surface-secondary beam is not merely a nuisance component. Above
roughly 100 V, loss-cone-only information saturates and the beam energy becomes
the strongest voltage marker. Its center is constrained by

\[
E_{beam} = e\left(U_{sc}-U_M\right),
\]

when the beam can reach the spacecraft.

Lineage evidence indicates that the observed beam width is dominated by the
instrument response, with effective `Delta E / E` near 0.48. A fixed absolute
15 eV Gaussian width is therefore not defensible over the full voltage range.
The model should be evaluated through the energy passband or represented as an
approximately one-channel feature whose width scales with beam energy.

The Experiment 003 beam-amplitude rail is consequently not yet a physical
measurement. It may represent an incorrect energy width, angular footprint,
bin integration, amplitude parameterization, or spectra outside the method's
validity regime.

## Disposition of Experiments 001-003

### Experiment 001

Durable findings are the exact benchmark localization, potential conventions,
geometry audit, approximate Lillis agreement, and catastrophic inadequacy of
the original production Cash model. The result should be described as
benchmark localization and diagnostic agreement, not a faithful Halekas
reproduction.

### Experiment 002

The telemetry result stands. Format-3 Level-0 data can recover compressed count
intervals when paired with Level-1 products, while calibrated flux alone is not
an exact independent Poisson observation. Count reconstruction and historical
method reconstruction are separate questions.

### Experiment 003

The no-advance decision stands. Its strongest justified conclusion is that no
tested observation model is production-ready when attached to the shared
hybrid forward model, Lillis mask, fixed spacecraft potential, and unvetted
confirmation panel.

Matched simulations verify the numerical estimators only under their assumed
forward model. They do not validate that model against lunar electron physics.
Experiment 003 therefore does not establish that count-based inference is
inferior to log-flux or robust shape inference.

The historical cone floor is also not automatically a generative expected
count rate. A count likelihood must physically or statistically account for
backscatter, cone refilling, beam counts, response, background, and remaining
model discrepancy before assigning raw-count precision.

## Consequences for Experiment 004

Experiment 004 must proceed in an evidence-ordered sequence:

1. Reproduce the best-supported historical estimator on spectrum 653.
2. Require agreement in distribution morphology, voltage, field ratio, beam
   location, and boundary location rather than voltage alone.
3. Estimate beam and loss-cone voltages separately as a physical consistency
   check.
4. Apply connectivity, pitch coverage, contrast, beam visibility, rail, error,
   contamination, and manual-review gates before interpreting a panel.
5. Only after the baseline is understood, compare modern response-convolved,
   conditional-shape, and count-based alternatives on their defensible domains.

A broad factorial search before the historical baseline would confound method
reconstruction with tuning and create too many opportunities for accidental
benchmark agreement.

## Project-Level Consequences

The global spatiotemporal identifiability result is not overturned. A faithful
retrieval and stricter selection will probably reduce the usable sample, which
would make the inverse problem at least as underconstrained. Nevertheless,
quantitative engineering statistics and mission-wide retrieval products must
be considered provisional until the measurement layer passes validation.

The strongest paper-level evidence chain is:

1. Reconstruct and validate the demonstrated per-measurement observable where
   the physics and data permit it.
2. Define and report the usable envelope and complete selection function.
3. Show what spatial and temporal information still cannot be recovered from
   those accepted measurements.

This preserves the scientific ambition while preventing a numerically precise
inverse analysis from resting on an unvalidated retrieval.

## Remaining Epistemic Limits

The clean-room reconstruction does not recover every private implementation
choice. Exact beam-amplitude handling, normalization, objective weighting, and
grid details remain uncertain and must be represented as discrete sensitivity
branches.

An unweighted squared-log residual does not acquire a formal 68% confidence
interpretation merely by applying a `Delta chi2 = 1` threshold. Historical
Monte Carlo propagation or modern empirical coverage calibration is required
unless a calibrated generative likelihood is established.
