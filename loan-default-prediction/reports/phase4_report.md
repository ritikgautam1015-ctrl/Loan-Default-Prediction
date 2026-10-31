# Phase 4 Model Optimization Report

## Executive Summary

The selected Phase 3 model was rechecked, calibration candidates were compared,
and a development operating threshold was chosen using validation F1 with a
documented recall tie-break. The test split was used only for the final
evaluation after those choices were fixed.

The result is suitable for subsequent research and explainability/API
prototyping, but not production credit decisions: discrimination is modest,
default recall depends strongly on the operating point, and no real intervention
costs or outcomes are available.

## Dataset

The source contains 255,347 loan applications and 18 columns. The target is
`Default`; the observed default prevalence is 11.61%
on validation. The model uses the Phase 2 processed feature representation and
the original validation/test rows were deterministically aligned to it.

## Selected Model

- Model: Logistic Regression
- Calibrated: false
- Model version: 1.0.0
- Selection basis: validation ROC-AUC from Phase 3; no test-based selection.

## Validation Performance

- ROC-AUC: 0.7592
- PR-AUC: 0.3346
- Precision: 0.2652
- Recall: 0.5823
- F1: 0.3644
- Brier score: 0.0905
- Confusion matrix (TN, FP / FN, TP): `[[26678, 7176], [1858, 2590]]`
- Default rate: 11.61%

## Threshold Analysis

Selected validation threshold: **0.15**. F1 is the primary
threshold criterion; thresholds within 0.005 of the maximum F1 are treated as
similar and the option with better default recall is preferred, then precision.
At the selected threshold, precision is 0.2652,
default recall is 0.5823, F1 is 0.3644,
false-positive rate is 0.2120, and
false-negative rate is 0.4177.
At threshold 0.50, validation precision/recall/F1 were
0.6268/0.0683/
0.1233; moving to 0.15 increases
default recall while also increasing false-positive referrals.

This is a development/portfolio operating threshold, not a bank credit-policy
threshold. A real policy requires false-positive/false-negative costs, recovery
rates, loan exposure, intervention costs, expected loss, and risk appetite.

## Calibration

Calibration is selected only when validation Brier score improves by at least
5% while ROC-AUC and PR-AUC each decline by no more than 0.01. Decision:
**retain the uncalibrated Phase 3 model**.
The candidate metrics are in `calibration_metrics.json`; the reliability plot is
`calibration_curve.png`.

## Error Analysis

- False positives: 7,176
- False negatives: 1,858
- True positives: 2,590
- True negatives: 26,678

Grouped error rates and false-positive/false-negative distributions are in
`error_analysis_by_feature.csv`, `false_negative_analysis.csv`, and
`false_positive_analysis.csv`. These are descriptive associations, not causal
findings. Stability group metrics are in `stability_analysis.csv`; they do not
constitute a fairness assessment.

## Risk Bands

Risk bands use a transparent +/- 0.05 probability neighborhood around the
selected threshold: Low below threshold-0.05, Medium from threshold-0.05 up to
threshold+0.05, and High at/above threshold+0.05. Loan exposure is a sum of
`LoanAmount` and is not a loss estimate. Counts and rates are in
`risk_band_analysis.csv`.

## NPA Simulation

**SIMULATED / ASSUMED SCENARIO:** the assumed intervention success rate is
30%. On validation, 4,448
actual defaults include 2,590 captured by the operating
threshold. The scenario estimates 777.0
prevented defaults, 3671.0 remaining
defaults, and 17.47% potential NPA reduction.
Captured, missed, and total default loan exposures are scenario diagnostics only.
Total actual-default loan exposure is 645,627,356;
captured-default exposure is 413,916,211;
missed-default exposure is 231,711,145.
These amounts are loan exposure proxies, not loss values. Full scenario inputs and
outputs are in `npa_simulation.json`.

This is a scenario analysis, not a measured reduction in real-world NPA. Actual
NPA reduction requires prospective intervention and observed repayment outcomes.

## Final Test Performance

The final test set was evaluated once after calibration and threshold decisions
were fixed. No test result was used to change the model or threshold.

- ROC-AUC: 0.7659
- PR-AUC: 0.3394
- Accuracy: 0.7669
- Precision: 0.2702
- Recall: 0.5922
- F1: 0.3711
- Brier score: 0.0900
- Confusion matrix (TN, FP / FN, TP): `[[26741, 7114], [1814, 2634]]`

## Limitations

- The dataset may not represent future applicants or changing economic conditions.
- No observed intervention, restructuring, collection, recovery, or realized-loss outcomes are available.
- The intervention success rate is assumed, not measured; estimated NPA reduction is simulated.
- The selected threshold optimizes portfolio F1 under a simple tie-break and does not encode bank costs or risk appetite.
- A single holdout test estimate does not establish out-of-time or external generalization.
- Group comparisons are not causal and do not establish fairness.
