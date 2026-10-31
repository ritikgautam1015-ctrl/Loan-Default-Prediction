"""Run Phase 3 baseline training, tuning, selection, and final evaluation."""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.evaluate import DEFAULT_THRESHOLD, evaluate_model  # noqa: E402
from src.models.train import (  # noqa: E402
    PROJECT_ROOT,
    build_baseline_models,
    build_serving_pipeline,
    load_preprocessor,
    load_processed_split,
    separate_features_target,
)
from src.models.tune import tune_model  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models"
REPORT_PATH = PROJECT_ROOT / "reports"
METRIC_COLUMNS = ["ROC_AUC", "PR_AUC", "Accuracy", "Precision", "Recall", "F1"]


def _print_metrics(name: str, metrics: dict[str, object]) -> None:
    print(f"{name}: ROC-AUC={metrics['ROC_AUC']:.4f}, PR-AUC={metrics['PR_AUC']:.4f}, "
          f"Accuracy={metrics['Accuracy']:.4f}, Precision={metrics['Precision']:.4f}, "
          f"Recall={metrics['Recall']:.4f}, F1={metrics['F1']:.4f}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: {metrics['Confusion_Matrix']}")


def _comparison_row(
    name: str, tuned: bool, evaluation_set: str, metrics: dict[str, object]
) -> dict[str, object]:
    return {
        "Model": name,
        "Tuned": tuned,
        "Evaluation_Set": evaluation_set,
        **{metric: metrics[metric] for metric in METRIC_COLUMNS},
    }


def main() -> int:
    print("=" * 40)
    print("PHASE 3 - MODEL TRAINING")
    print("=" * 40)
    print("\nLoading processed data...")
    train = load_processed_split("train")
    validation = load_processed_split("validation")
    x_train, y_train = separate_features_target(train)
    x_validation, y_validation = separate_features_target(validation)
    if not x_train.columns.equals(x_validation.columns):
        raise ValueError("Training and validation feature columns do not match.")
    print(f"Train: {len(train)}\nValidation: {len(validation)}")
    print(f"Target excluded from feature matrix: {'Default' not in x_train.columns}")
    print(f"Default threshold: {DEFAULT_THRESHOLD:.2f}")

    baseline_rows: list[dict[str, object]] = []
    print("\nTraining baseline models...")
    for name, model in build_baseline_models().items():
        print(f"\nTraining {name}...")
        model.fit(x_train, y_train)
        metrics = evaluate_model(model, x_validation, y_validation)
        _print_metrics(name, metrics)
        baseline_rows.append(_comparison_row(name, False, "Validation", metrics))

    REPORT_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    baseline_comparison = pd.DataFrame(baseline_rows)
    baseline_comparison.to_csv(REPORT_PATH / "baseline_model_comparison.csv", index=False)
    print("\n========================================")
    print("BASELINE COMPARISON (VALIDATION)")
    print("========================================")
    print(baseline_comparison[["Model", *METRIC_COLUMNS]].to_string(index=False))

    tuned_models: dict[str, object] = {}
    tuned_metrics: dict[str, dict[str, object]] = {}
    tuned_parameters: dict[str, dict[str, object]] = {}
    tuned_rows: list[dict[str, object]] = []
    for family in ("Logistic Regression", "Random Forest", "XGBoost"):
        print(f"\nTuning {family} (training data only)...")
        model, parameters, cv_score = tune_model(family, x_train, y_train)
        metrics = evaluate_model(model, x_validation, y_validation)
        tuned_models[family] = model
        tuned_metrics[family] = metrics
        tuned_parameters[family] = parameters
        _print_metrics(f"Tuned {family}", metrics)
        print(f"Best training CV ROC-AUC: {cv_score:.4f}")
        print(f"Best parameters: {parameters}")
        tuned_rows.append(_comparison_row(family, True, "Validation", metrics))

    tuned_comparison = pd.DataFrame(tuned_rows)
    comparison = pd.concat([baseline_comparison, tuned_comparison], ignore_index=True)
    comparison.to_csv(REPORT_PATH / "model_comparison.csv", index=False)
    print("\n========================================")
    print("TUNED MODEL COMPARISON (VALIDATION)")
    print("========================================")
    print(tuned_comparison[["Model", *METRIC_COLUMNS]].to_string(index=False))

    selected_name = max(
        tuned_metrics,
        key=lambda name: (tuned_metrics[name]["ROC_AUC"], name),
    )
    selected_model = tuned_models[selected_name]
    selected_validation_metrics = tuned_metrics[selected_name]

    model_filenames = {
        "Logistic Regression": "logistic_regression.joblib",
        "Random Forest": "random_forest.joblib",
        "XGBoost": "xgboost.joblib",
    }
    for name, model in tuned_models.items():
        joblib.dump(model, MODEL_PATH / model_filenames[name])

    serving_model = build_serving_pipeline(load_preprocessor(), selected_model)
    joblib.dump(serving_model, MODEL_PATH / "best_model.joblib")

    print("\n========================================")
    print("FINAL MODEL")
    print("========================================")
    print(f"Selected model: {selected_name}")
    print(f"Validation ROC-AUC: {selected_validation_metrics['ROC_AUC']:.4f}")

    # Test is first read only after validation-based selection and model saving.
    test = load_processed_split("test")
    x_test, y_test = separate_features_target(test)
    if not x_train.columns.equals(x_test.columns):
        raise ValueError("Training and test feature columns do not match.")
    test_metrics = evaluate_model(selected_model, x_test, y_test)
    _print_metrics("Final test result", test_metrics)
    test_row = _comparison_row(selected_name, True, "Test (final)", test_metrics)
    comparison = pd.concat([comparison, pd.DataFrame([test_row])], ignore_index=True)
    comparison.to_csv(REPORT_PATH / "model_comparison.csv", index=False)

    metadata = {
        "model_name": selected_name,
        "target": "Default",
        "primary_metric": "ROC-AUC",
        "validation_roc_auc": selected_validation_metrics["ROC_AUC"],
        "test_roc_auc": test_metrics["ROC_AUC"],
        "random_state": 42,
        "threshold": DEFAULT_THRESHOLD,
        "tuned_parameters": tuned_parameters[selected_name],
        "test_metrics": {metric: test_metrics[metric] for metric in METRIC_COLUMNS},
        "test_confusion_matrix": test_metrics["Confusion_Matrix"],
        "serving_input": "Raw predictor columns; LoanID and Default are not required.",
        "preprocessing_artifact": "Phase 2 fitted preprocessor.joblib",
    }
    (MODEL_PATH / "best_model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    target_met = test_metrics["ROC_AUC"] >= 0.85
    print("\n========================================")
    print("FINAL TEST RESULT")
    print("========================================")
    for metric in METRIC_COLUMNS:
        print(f"{metric.replace('_', '-')} : {test_metrics[metric]:.4f}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: {test_metrics['Confusion_Matrix']}")
    if target_met:
        print("Target achieved.")
    else:
        print(f"Target not achieved. Actual ROC-AUC: {test_metrics['ROC_AUC']:.4f}")
    print("\nSaved models, metadata, and validation/final-test comparison report.")
    print("Phase 3 completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())