"""Run Phase 4 validation optimization, thresholding, and final evaluation."""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.features import engineer_features  # noqa: E402
from src.data.loader import EXPECTED_COLUMNS, load_dataset, validate_dataset  # noqa: E402
from src.data.preprocessing import RANDOM_STATE, transformed_frame  # noqa: E402
from src.models.calibration import compare_calibration, probability_metrics  # noqa: E402
from src.models.error_analysis import (  # noqa: E402
    ANALYSIS_FEATURES,
    confusion_counts,
    create_error_analysis,
    grouped_error_analysis,
    risk_band_analysis,
    simulate_intervention,
    stability_analysis,
)
from src.models.evaluate import evaluate_model, evaluate_probabilities  # noqa: E402
from src.models.threshold import analyze_thresholds, select_operating_threshold  # noqa: E402
from src.models.train import (  # noqa: E402
    PROJECT_ROOT,
    build_serving_pipeline,
    load_preprocessor,
    load_processed_split,
)
from sklearn.model_selection import train_test_split

MODEL_PATH = PROJECT_ROOT / "models"
REPORT_PATH = PROJECT_ROOT / "reports"
PLOT_PATHS = {
    "roc": REPORT_PATH / "roc_curve.png",
    "precision_recall": REPORT_PATH / "precision_recall_curve.png",
    "calibration": REPORT_PATH / "calibration_curve.png",
}
INTERVENTION_SUCCESS_RATE = 0.30
MODEL_VERSION = "1.0.0"
NUMERICAL_COLUMNS = [
    "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
    "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio",
]
CATEGORICAL_COLUMNS = [
    "Education", "EmploymentType", "MaritalStatus", "HasMortgage",
    "HasDependents", "LoanPurpose", "HasCoSigner",
]


def load_original_split(split_name: str) -> pd.DataFrame:
    """Recreate a Phase 2 split and return its untouched original columns.

    The deterministic stratified partition is reproduced from Phase 2. Only
    requested row positions are materialized; test rows are requested last.
    """
    if split_name not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown split {split_name!r}.")
    raw = load_dataset()
    validate_dataset(raw)
    if list(raw.columns) != EXPECTED_COLUMNS:
        raise ValueError("Raw dataset columns no longer match the Phase 2 schema.")
    if raw.duplicated().any():
        raw = raw.drop_duplicates().copy()
    for column in NUMERICAL_COLUMNS + ["Default"]:
        converted = pd.to_numeric(raw[column], errors="coerce")
        invalid = raw[column].notna() & converted.isna()
        if invalid.any():
            raise ValueError(f"Invalid numeric values found in raw column {column!r}.")
        raw[column] = converted
    for column in CATEGORICAL_COLUMNS:
        raw[column] = raw[column].astype("string").str.strip().astype("object")

    target = raw["Default"].to_numpy(dtype="int64")
    row_positions = np.arange(len(raw))
    train_positions, remainder_positions = train_test_split(
        row_positions,
        test_size=0.30,
        stratify=target,
        random_state=RANDOM_STATE,
    )
    validation_positions, test_positions = train_test_split(
        remainder_positions,
        test_size=0.50,
        stratify=target[remainder_positions],
        random_state=RANDOM_STATE,
    )
    selected_positions = {
        "train": train_positions,
        "validation": validation_positions,
        "test": test_positions,
    }[split_name]
    original = raw.iloc[selected_positions].copy()
    return original.drop(columns="LoanID")


def aligned_original_features(
    original: pd.DataFrame, processed: pd.DataFrame, preprocessor
) -> tuple[pd.DataFrame, pd.Series]:
    """Prove deterministic raw rows match the saved Phase 2 encoded split."""
    target = original["Default"].astype("int64").reset_index(drop=True)
    raw_predictors = original.drop(columns="Default").reset_index(drop=True)
    expected_encoded = transformed_frame(
        preprocessor,
        engineer_features(raw_predictors),
        target,
    )
    processed = processed.reset_index(drop=True)
    if len(expected_encoded) != len(processed):
        raise ValueError("Raw and processed split sizes do not match.")
    if not np.array_equal(target.to_numpy(), processed["Default"].to_numpy(dtype=int)):
        raise ValueError("Raw and processed split target order does not match.")
    if not expected_encoded.columns.equals(processed.columns):
        raise ValueError("Raw and processed split feature schemas do not match.")
    if not np.allclose(
        expected_encoded.iloc[:, :-1].to_numpy(),
        processed.iloc[:, :-1].to_numpy(),
        rtol=1e-10,
        atol=1e-10,
    ):
        raise ValueError("Raw-to-processed features do not match Phase 2 output.")
    return raw_predictors, target


def _save_validation_plots(target, probabilities, calibration_series: dict[str, np.ndarray]) -> None:
    REPORT_PATH.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(target, probabilities)
    metrics = probability_metrics(target, probabilities)
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.plot(fpr, tpr, label=f"Selected model (ROC-AUC={metrics['ROC_AUC']:.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    axis.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="Validation ROC curve")
    axis.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PATHS["roc"], dpi=150)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(target, probabilities)
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.plot(recall, precision, label=f"Average Precision={metrics['PR_AUC']:.3f}")
    axis.set(xlabel="Recall", ylabel="Precision", title="Validation precision-recall curve")
    axis.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PATHS["precision_recall"], dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 5))
    for name, candidate_probability in calibration_series.items():
        observed, predicted = calibration_curve(
            target, candidate_probability, n_bins=10, strategy="quantile"
        )
        axis.plot(predicted, observed, marker="o", label=name)
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    axis.set(
        xlabel="Mean predicted probability",
        ylabel="Observed default frequency",
        title="Validation calibration curve",
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PATHS["calibration"], dpi=150)
    plt.close(fig)


def _write_report(
    validation_metrics: dict[str, object],
    calibration_results: dict[str, object],
    calibration_selected: str | None,
    selected_threshold: float,
    threshold_row: pd.Series,
    reference_threshold_row: pd.Series,
    errors: pd.DataFrame,
    risk_bands: pd.DataFrame,
    scenario: dict[str, object],
    stability: pd.DataFrame,
    test_metrics: dict[str, object],
    metadata: dict[str, object],
) -> None:
    error_counts = errors["Error_Type"].value_counts().to_dict()
    validation_cm = validation_metrics["Confusion_Matrix"]
    test_cm = test_metrics["Confusion_Matrix"]
    report = f"""# Phase 4 Model Optimization Report

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
`Default`; the observed default prevalence is {validation_metrics['Default_Rate']:.2%}
on validation. The model uses the Phase 2 processed feature representation and
the original validation/test rows were deterministically aligned to it.

## Selected Model

- Model: {metadata['model_name']}
- Calibrated: {str(metadata['calibrated']).lower()}
- Model version: {metadata['model_version']}
- Selection basis: validation ROC-AUC from Phase 3; no test-based selection.

## Validation Performance

- ROC-AUC: {validation_metrics['ROC_AUC']:.4f}
- PR-AUC: {validation_metrics['PR_AUC']:.4f}
- Precision: {validation_metrics['Precision']:.4f}
- Recall: {validation_metrics['Recall']:.4f}
- F1: {validation_metrics['F1']:.4f}
- Brier score: {validation_metrics['Brier_Score']:.4f}
- Confusion matrix (TN, FP / FN, TP): `{validation_cm}`
- Default rate: {validation_metrics['Default_Rate']:.2%}

## Threshold Analysis

Selected validation threshold: **{selected_threshold:.2f}**. F1 is the primary
threshold criterion; thresholds within 0.005 of the maximum F1 are treated as
similar and the option with better default recall is preferred, then precision.
At the selected threshold, precision is {threshold_row['Precision']:.4f},
default recall is {threshold_row['Recall']:.4f}, F1 is {threshold_row['F1']:.4f},
false-positive rate is {threshold_row['False_Positive_Rate']:.4f}, and
false-negative rate is {threshold_row['False_Negative_Rate']:.4f}.
At threshold 0.50, validation precision/recall/F1 were
{reference_threshold_row['Precision']:.4f}/{reference_threshold_row['Recall']:.4f}/
{reference_threshold_row['F1']:.4f}; moving to {selected_threshold:.2f} increases
default recall while also increasing false-positive referrals.

This is a development/portfolio operating threshold, not a bank credit-policy
threshold. A real policy requires false-positive/false-negative costs, recovery
rates, loan exposure, intervention costs, expected loss, and risk appetite.

## Calibration

Calibration is selected only when validation Brier score improves by at least
5% while ROC-AUC and PR-AUC each decline by no more than 0.01. Decision:
**{'selected ' + calibration_selected if calibration_selected else 'retain the uncalibrated Phase 3 model'}**.
The candidate metrics are in `calibration_metrics.json`; the reliability plot is
`calibration_curve.png`.

## Error Analysis

- False positives: {error_counts.get('False Positive', 0):,}
- False negatives: {error_counts.get('False Negative', 0):,}
- True positives: {error_counts.get('Correct Positive', 0):,}
- True negatives: {error_counts.get('Correct Negative', 0):,}

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
{scenario['intervention_success_rate_assumed']:.0%}. On validation, {scenario['actual_defaults']:,}
actual defaults include {scenario['defaults_captured']:,} captured by the operating
threshold. The scenario estimates {scenario['estimated_prevented_defaults']:.1f}
prevented defaults, {scenario['estimated_remaining_defaults']:.1f} remaining
defaults, and {scenario['estimated_npa_reduction']:.2%} potential NPA reduction.
Captured, missed, and total default loan exposures are scenario diagnostics only.
Total actual-default loan exposure is {scenario['actual_default_loan_exposure']:,.0f};
captured-default exposure is {scenario['captured_default_loan_exposure']:,.0f};
missed-default exposure is {scenario['missed_default_loan_exposure']:,.0f}.
These amounts are loan exposure proxies, not loss values. Full scenario inputs and
outputs are in `npa_simulation.json`.

This is a scenario analysis, not a measured reduction in real-world NPA. Actual
NPA reduction requires prospective intervention and observed repayment outcomes.

## Final Test Performance

The final test set was evaluated once after calibration and threshold decisions
were fixed. No test result was used to change the model or threshold.

- ROC-AUC: {test_metrics['ROC_AUC']:.4f}
- PR-AUC: {test_metrics['PR_AUC']:.4f}
- Accuracy: {test_metrics['Accuracy']:.4f}
- Precision: {test_metrics['Precision']:.4f}
- Recall: {test_metrics['Recall']:.4f}
- F1: {test_metrics['F1']:.4f}
- Brier score: {test_metrics['Brier_Score']:.4f}
- Confusion matrix (TN, FP / FN, TP): `{test_cm}`

## Limitations

- The dataset may not represent future applicants or changing economic conditions.
- No observed intervention, restructuring, collection, recovery, or realized-loss outcomes are available.
- The intervention success rate is assumed, not measured; estimated NPA reduction is simulated.
- The selected threshold optimizes portfolio F1 under a simple tie-break and does not encode bank costs or risk appetite.
- A single holdout test estimate does not establish out-of-time or external generalization.
- Group comparisons are not causal and do not establish fairness.
"""
    (REPORT_PATH / "phase4_report.md").write_text(report, encoding="utf-8")


def main() -> int:
    print("=" * 40)
    print("PHASE 4 - MODEL OPTIMIZATION")
    print("=" * 40)
    REPORT_PATH.mkdir(parents=True, exist_ok=True)
    metadata_path = MODEL_PATH / "best_model_metadata.json"
    phase3_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    print("\nLoading selected model...")
    original_pipeline = joblib.load(MODEL_PATH / "best_model.joblib")
    preprocessor = load_preprocessor()
    train_processed = load_processed_split("train")
    validation_processed = load_processed_split("validation")
    x_train = train_processed.drop(columns="Default")
    y_train = train_processed["Default"].astype("int64")
    raw_validation = load_original_split("validation")
    x_validation_raw, y_validation = aligned_original_features(
        raw_validation, validation_processed, preprocessor
    )

    print("\nValidation evaluation...")
    base_probabilities = original_pipeline.predict_proba(x_validation_raw)[:, 1]
    base_probability_metrics = probability_metrics(y_validation, base_probabilities)
    print(f"ROC-AUC: {base_probability_metrics['ROC_AUC']:.4f}")
    print(f"PR-AUC: {base_probability_metrics['PR_AUC']:.4f}")

    print("\nCalibration analysis...")
    original_classifier = original_pipeline.named_steps["classifier"]
    calibrated_classifier, calibration_method, calibration_analysis = compare_calibration(
        original_classifier,
        x_train,
        y_train,
        validation_processed.drop(columns="Default"),
        y_validation,
    )
    candidate_metrics = calibration_analysis["metrics"]
    if calibration_method is None:
        selected_pipeline = original_pipeline
        validation_probabilities = base_probabilities
        selected_metrics = candidate_metrics["uncalibrated"]
        print("Calibration did not meet the predefined benefit and discrimination criteria; retaining the original model.")
    else:
        selected_pipeline = build_serving_pipeline(preprocessor, calibrated_classifier)
        validation_probabilities = selected_pipeline.predict_proba(x_validation_raw)[:, 1]
        selected_metrics = probability_metrics(y_validation, validation_probabilities)
        candidate_metrics["selected"] = selected_metrics
        print(f"Selected {calibration_method} calibration based on validation results.")

    candidate_methods = [name for name in ("sigmoid", "isotonic") if name in candidate_metrics]
    best_candidate_method = min(
        candidate_methods, key=lambda name: candidate_metrics[name]["Brier_Score"]
    )
    calibrated_pipeline_candidate = build_serving_pipeline(
        preprocessor, calibration_analysis["candidates"][best_candidate_method]
    )
    joblib.dump(calibrated_pipeline_candidate, MODEL_PATH / "calibrated_model.joblib")
    joblib.dump(selected_pipeline, MODEL_PATH / "best_model.joblib")

    print("\nAnalyzing thresholds...")
    threshold_analysis = analyze_thresholds(y_validation, validation_probabilities)
    selected_threshold = select_operating_threshold(threshold_analysis, f1_tolerance=0.005)
    threshold_analysis["Probability_Source"] = (
        calibration_method if calibration_method else "uncalibrated"
    )
    threshold_analysis.to_csv(REPORT_PATH / "threshold_analysis.csv", index=False)
    threshold_row = threshold_analysis.loc[
        np.isclose(threshold_analysis["Threshold"], selected_threshold)
    ].iloc[0]
    reference_threshold_row = threshold_analysis.loc[
        np.isclose(threshold_analysis["Threshold"], 0.50)
    ].iloc[0]
    print(f"Selected threshold: {selected_threshold:.2f} (validation F1={threshold_row['F1']:.4f})")

    validation_evaluation = evaluate_model(
        selected_pipeline, x_validation_raw, y_validation, threshold=selected_threshold
    )
    validation_metrics = {
        **selected_metrics,
        **{
            metric: validation_evaluation[metric]
            for metric in ("Accuracy", "Precision", "Recall", "F1", "Confusion_Matrix")
        },
        "Default_Rate": float(y_validation.mean()),
    }
    error_frame = create_error_analysis(
        x_validation_raw, y_validation, validation_probabilities, selected_threshold
    )
    error_counts = confusion_counts(error_frame)
    error_frame.to_csv(REPORT_PATH / "error_analysis.csv", index=False)
    grouped_errors = grouped_error_analysis(error_frame, ANALYSIS_FEATURES)
    grouped_errors.to_csv(REPORT_PATH / "error_analysis_by_feature.csv", index=False)
    grouped_errors.loc[grouped_errors["False_Negative_Count"] > 0].to_csv(
        REPORT_PATH / "false_negative_analysis.csv", index=False
    )
    grouped_errors.loc[grouped_errors["False_Positive_Count"] > 0].to_csv(
        REPORT_PATH / "false_positive_analysis.csv", index=False
    )

    stability = stability_analysis(error_frame, validation_probabilities)
    stability.to_csv(REPORT_PATH / "stability_analysis.csv", index=False)
    risk_bands = risk_band_analysis(
        x_validation_raw, y_validation, validation_probabilities, selected_threshold
    )
    risk_bands.to_csv(REPORT_PATH / "risk_band_analysis.csv", index=False)
    scenario = simulate_intervention(
        y_validation,
        validation_probabilities,
        x_validation_raw["LoanAmount"],
        selected_threshold,
        INTERVENTION_SUCCESS_RATE,
    )
    scenario["label"] = "SIMULATED / ASSUMED; not measured real-world NPA reduction"
    (REPORT_PATH / "npa_simulation.json").write_text(
        json.dumps(scenario, indent=2), encoding="utf-8"
    )

    calibration_series = {"Uncalibrated": base_probabilities}
    for method in candidate_methods:
        calibration_series[method.title()] = calibration_analysis["candidates"][method].predict_proba(
            validation_processed.drop(columns="Default")
        )[:, 1]
    calibration_series["Selected"] = validation_probabilities
    _save_validation_plots(y_validation, validation_probabilities, calibration_series)

    calibration_report = {
        "methodology": {
            "calibration_fit_data": "training split only",
            "validation_used_for": "candidate comparison and selection",
            "minimum_brier_relative_improvement": 0.05,
            "maximum_roc_auc_drop": 0.01,
            "maximum_pr_auc_drop": 0.01,
            "selected_method": calibration_method,
            "calibrated_model_selected": calibration_method is not None,
        },
        "validation_metrics": candidate_metrics,
        "best_calibrated_candidate": best_candidate_method,
    }
    (REPORT_PATH / "calibration_metrics.json").write_text(
        json.dumps(calibration_report, indent=2), encoding="utf-8"
    )

    print("\nError analysis...")
    print(f"False positives: {error_counts['FP']:,}; false negatives: {error_counts['FN']:,}")
    print("\nRisk-band analysis...")
    print(risk_bands.to_string(index=False))
    print("\nNPA scenario simulation...")
    print(f"Assumed intervention success rate: {INTERVENTION_SUCCESS_RATE:.0%}")
    print(f"Estimated scenario reduction: {scenario['estimated_npa_reduction']:.2%} (not measured)")

    final_model_metadata = {
        **phase3_metadata,
        "model_version": MODEL_VERSION,
        "validation_roc_auc": validation_metrics["ROC_AUC"],
        "test_roc_auc": None,
        "validation_pr_auc": validation_metrics["PR_AUC"],
        "test_pr_auc": None,
        "classification_threshold": selected_threshold,
        "threshold_selection": "Validation F1; within 0.005 of max F1, prefer recall then precision.",
        "calibrated": calibration_method is not None,
        "calibration_method": calibration_method,
        "calibration_selection_rule": "At least 5% Brier improvement; ROC-AUC and PR-AUC drops <= 0.01.",
        "validation_metrics": validation_metrics,
        "intervention_success_rate_assumed": INTERVENTION_SUCCESS_RATE,
        "random_state": 42,
    }

    # The test split is first loaded after the calibration and threshold decisions are saved.
    print("\nFinal test evaluation...")
    processed_test = load_processed_split("test")
    raw_test = load_original_split("test")
    x_test_raw, y_test = aligned_original_features(raw_test, processed_test, preprocessor)
    test_probabilities = selected_pipeline.predict_proba(x_test_raw)[:, 1]
    test_metrics = {
        **evaluate_probabilities(y_test, test_probabilities, threshold=selected_threshold),
        "Brier_Score": float(brier_score_loss(y_test, test_probabilities)),
    }
    final_model_metadata["test_roc_auc"] = test_metrics["ROC_AUC"]
    final_model_metadata["test_pr_auc"] = test_metrics["PR_AUC"]
    final_model_metadata["test_metrics"] = test_metrics
    final_model_metadata["test_confusion_matrix"] = test_metrics["Confusion_Matrix"]
    final_model_metadata.pop("threshold", None)
    metadata_path.write_text(json.dumps(final_model_metadata, indent=2), encoding="utf-8")

    _write_report(
        validation_metrics,
        calibration_report,
        calibration_method,
        selected_threshold,
        threshold_row,
        reference_threshold_row,
        error_frame,
        risk_bands,
        scenario,
        stability,
        test_metrics,
        final_model_metadata,
    )

    print(f"Test ROC-AUC: {test_metrics['ROC_AUC']:.4f}")
    print(f"Test PR-AUC: {test_metrics['PR_AUC']:.4f}")
    print(f"Test Recall: {test_metrics['Recall']:.4f}")
    print(f"Test F1: {test_metrics['F1']:.4f}")
    print("Phase 4 completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())