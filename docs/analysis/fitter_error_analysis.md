
# Fitter Error Threshold Analysis

**Date:** 2025-08-08

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
