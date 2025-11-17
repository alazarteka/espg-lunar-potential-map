
# Fitter Error Threshold Analysis

**Date:** 2025-08-08
**Updated:** 2025-11-16

## 1. Objective

This document details the analysis performed to establish a quantitative threshold for determining the quality of a kappa distribution fit to the Lunar Prospector electron flux data. The goal is to create a statistically-backed cutoff value for the chi-squared error, which will allow the automated classification of fits as either "good" or "questionable."

## 2. Methodology

The analysis was conducted in two main phases:

1.  **Initial Data Collection:** The `scripts/dev/error_distribution_analysis.py` script was run across the entire available dataset. This script iterates through all `.TAB` files, performs a kappa fit for every energy spectrum, and records the resulting chi-squared error.
2.  **Outlier Filtering and Re-analysis:** The initial results showed that the error distribution was heavily skewed by a small number of catastrophic fits with extremely high chi-squared values. To get a clearer picture of the typical error distribution, these outliers were filtered out, and the statistics were re-calculated on the cleaned dataset.

## 3. Results

### 3.1. Raw (Unfiltered) Data

The initial run produced the following statistics from **558,356** successful fits:

*   **Mean error:** 1.03e+11
*   **Median error:** 1911.52
*   **95th percentile:** 2.67e+11
*   **99th percentile:** 2.99e+12

The mean and percentile values were clearly dominated by extreme outliers, making them unsuitable for defining a useful threshold.

### 3.2. Filtered Data

To address the outlier issue, a filter was applied to the dataset, removing all fits with a chi-squared error greater than **1,000,000**. This removed the most extreme outliers while preserving the bulk of the data.

The analysis on the remaining **433,699** fits yielded the following statistics:

*   **Mean error:** 47,693.11
*   **Median error:** 1422.25
*   **95th percentile:** 215,075.43
*   **99th percentile:** 657,088.07

These filtered statistics provide a much more realistic view of the error distribution for typical fits.

## 4. Justification for the Threshold

Based on the filtered data, we have chosen the **95th percentile** as our official threshold for a "good" fit. This leads to the following definition:

**`FIT_ERROR_THRESHOLD = 215000`**

This threshold is justified for the following reasons:

1.  **Statistical Significance:** It is a statistically robust value that separates the vast majority (95%) of reasonable fits from the top 5% of fits that have a significantly higher error.
2.  **Practicality:** It provides a clear, quantitative, and automated way to flag potentially problematic fits for further review without being overly aggressive and discarding a large amount of data.
3.  **Robustness:** By being derived from the filtered data, it is not skewed by the catastrophic outliers and reflects the true distribution of errors for the bulk of the data.

Any fit with a chi-squared value above this threshold will be considered "questionable" and should be treated with caution in subsequent analysis.

## 5. Current Implementation Status (Updated 2025-11-16)

**⚠️ DECISION PENDING:**

The current code in `src/config.py` uses:

```python
FIT_ERROR_THRESHOLD = 21_500_000_000  # chi-squared threshold for a good fit
```

This value (2.15×10¹⁰) is approximately **100,000× larger** than the statistically-derived threshold documented above (2.15×10⁵).

### 5.1. Comparison of Options

| Threshold Value | Percentile | Fits Rejected | Rationale |
|----------------|------------|---------------|-----------|
| **215,000** | 95th (filtered) | ~5% | Statistically justified, rejects questionable fits |
| **657,000** | 99th (filtered) | ~1% | Middle ground, keeps more data while still filtering poor fits |
| **21,500,000,000** | ~99th+ (unfiltered) | <1% | Current code value, very permissive, accepts almost everything |

### 5.2. Impact on Analysis

The threshold is used in two places:

1. **`src/kappa.py:528`** - Sets `is_good_fit` boolean flag (informational only)
2. **`src/potential_mapper/pipeline.py:276`** - **Actively filters data**: fits with chi² > threshold are **discarded** from surface potential calculations

With the current threshold of 2.15×10¹⁰:
- Nearly all fits are accepted (only catastrophic failures rejected)
- Pipeline includes data from the top ~5% of error values that would be rejected by the documented threshold
- Trade-off: maximum data retention vs. potential inclusion of poor-quality fits

### 5.3. Open Questions

Before finalizing the threshold, the following should be investigated:

1. **Data loss assessment:** How many measurements would be lost by using the stricter 215,000 threshold?
2. **Spatial coverage:** Does the stricter threshold create gaps in spatial coverage?
3. **Science impact:** Are the marginal fits (chi² between 215k and 21.5B) scientifically useful or problematic?
4. **Validation:** Can we cross-validate fits in this range against other indicators (e.g., consistency with neighboring measurements)?

### 5.4. Recommendation

The choice of threshold should be documented and justified based on science requirements:

- **If maximizing spatial coverage is critical:** Keep current permissive threshold but add validation checks
- **If data quality is paramount:** Use the 95th percentile (215,000) from statistical analysis
- **If balanced approach needed:** Consider 99th percentile (657,000) as middle ground

**TODO:** Make explicit decision and update both `src/config.py` and this documentation accordingly.
