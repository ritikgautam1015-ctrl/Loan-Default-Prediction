"""Track the already-selected Phase 4 model in local MLflow."""

import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import joblib
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
import mlflow
import mlflow.sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitoring.config import (  # noqa: E402
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    PROJECT_ROOT,
    REPORT_MLFLOW_PATH,
)

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
MODEL_METADATA_PATH = PROJECT_ROOT / "models" / "best_model_metadata.json"
PHASE2_METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "feature_metadata.json"
TRACKED_ARTIFACTS = [
    MODEL_PATH,
    MODEL_METADATA_PATH,
    PROJECT_ROOT / "data" / "processed" / "preprocessor.joblib",
    PROJECT_ROOT / "reports" / "model_comparison.csv",
    PROJECT_ROOT / "reports" / "phase4_report.md",
]


def _tracking_uri(value: str | Path) -> str:
    text = str(value)
    if text.startswith(("file:", "http://", "https://", "databricks")):
        return text
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve().as_uri()


def _classifier(model: Any) -> Any:
    return model.steps[-1][1] if hasattr(model, "steps") else model


def _model_parameters(model: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    classifier = _classifier(model)
    params = classifier.get_params(deep=False)
    name = metadata["model_name"]
    keys = {
        "Logistic Regression": ["C", "solver", "max_iter", "class_weight"],
        "Random Forest": [
            "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
            "class_weight",
        ],
        "XGBoost": [
            "n_estimators", "learning_rate", "max_depth", "subsample",
            "colsample_bytree",
        ],
    }.get(name, [])
    logged = {
        "model_type": name,
        "model_version": str(metadata.get("model_version", "unknown")),
        "decision_threshold": metadata.get("classification_threshold", metadata.get("threshold")),
        "calibrated": metadata.get("calibrated", False),
        "random_state": metadata.get("random_state", 42),
    }
    logged.update({key: params[key] for key in keys if key in params})
    return {
        key: str(value)
        for key, value in logged.items()
        if value is not None or key == "class_weight"
    }


def _phase2_data_metadata() -> dict[str, Any]:
    if not PHASE2_METADATA_PATH.is_file():
        return {}
    return json.loads(PHASE2_METADATA_PATH.read_text(encoding="utf-8"))


def _metrics(metadata: dict[str, Any]) -> dict[str, float]:
    logged: dict[str, float] = {}
    for split in ("validation", "test"):
        values = metadata.get(f"{split}_metrics", {})
        for name in (
            "ROC_AUC", "PR_AUC", "Accuracy", "Precision", "Recall", "F1", "Brier_Score"
        ):
            value = values.get(name)
            if isinstance(value, (int, float)):
                logged[f"{split}_{name.lower()}"] = float(value)
    return logged


def run_tracking(
    tracking_uri: str | Path = MLFLOW_TRACKING_URI,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    summary_path: str | Path | None = None,
    log_model: bool = True,
    log_artifacts: bool = True,
) -> dict[str, Any]:
    """Log Phase 4 metadata/model without fitting or changing the model."""
    if not MODEL_PATH.is_file() or not MODEL_METADATA_PATH.is_file():
        raise FileNotFoundError("Phase 4 final model and metadata must exist before tracking.")
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    phase2 = _phase2_data_metadata()

    resolved_uri = _tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(resolved_uri)
    experiment = mlflow.set_experiment(experiment_name)
    tags = {
        "project": "loan-default-prediction",
        "phase": "phase-7",
        "task": "binary-classification",
        "target": "Default",
        "python_version": platform.python_version(),
        "dataset": "data/raw/Loan_default.csv",
        "model_version": str(metadata.get("model_version", "unknown")),
    }
    sizes = {
        "training_rows": phase2.get("train_rows"),
        "validation_rows": phase2.get("validation_rows"),
        "test_rows": phase2.get("test_rows"),
        "feature_count": len(phase2.get("output_feature_columns", [])) or None,
    }
    tags.update({key: str(value) for key, value in sizes.items() if value is not None})
    metric_values = _metrics(metadata)
    params = _model_parameters(model, metadata)

    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        mlflow.set_tags(tags)
        mlflow.log_params(params)
        mlflow.log_metrics(metric_values)

        if log_artifacts:
            for artifact in TRACKED_ARTIFACTS:
                if artifact.is_file():
                    mlflow.log_artifact(str(artifact), artifact_path="phase_artifacts")
        if log_model:
            mlflow.sklearn.log_model(
                sk_model=model,
                name="best_model",
                serialization_format="pickle",
                metadata={
                    "model_version": str(metadata.get("model_version", "unknown")),
                    "decision_threshold": str(params.get("decision_threshold", "")),
                },
            )

        summary = {
            "experiment_name": experiment_name,
            "run_id": run.info.run_id,
            "model_type": metadata["model_name"],
            "model_version": str(metadata.get("model_version", "unknown")),
            "decision_threshold": float(
                metadata.get("classification_threshold", metadata.get("threshold", 0.5))
            ),
            "tracking_uri": resolved_uri,
            "logged_parameters": params,
            "logged_metrics": metric_values,
            "training_rows": sizes["training_rows"],
            "validation_rows": sizes["validation_rows"],
            "test_rows": sizes["test_rows"],
            "feature_count": sizes["feature_count"],
            "model_logged": bool(log_model),
        }

    destination = Path(summary_path) if summary_path else REPORT_MLFLOW_PATH / "mlflow_run_summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    summary = run_tracking()
    print("MLflow tracking completed.")
    print(f"Experiment: {summary['experiment_name']}")
    print(f"Run ID: {summary['run_id']}")
    print(f"Model: {summary['model_type']} ({summary['model_version']})")
    print(f"Metrics logged: {len(summary['logged_metrics'])}")
    print("Summary: reports/mlflow/mlflow_run_summary.json")
    print("Open the local UI with: mlflow ui --backend-store-uri ./mlruns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())