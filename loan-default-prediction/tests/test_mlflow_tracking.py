"""Local MLflow tracking tests without a remote server or model retraining."""

import json

import mlflow

from scripts.run_mlflow_tracking import run_tracking


def test_mlflow_run_logs_phase4_values_and_writes_summary(tmp_path) -> None:
    tracking_path = tmp_path / "mlruns"
    summary_path = tmp_path / "summary.json"
    summary = run_tracking(
        tracking_uri=tracking_path.as_uri(),
        experiment_name="loan-default-prediction-test",
        summary_path=summary_path,
        log_model=False,
        log_artifacts=False,
    )
    assert summary["run_id"]
    assert summary["model_type"] == "Logistic Regression"
    assert summary["decision_threshold"] == 0.15
    assert "validation_roc_auc" in summary["logged_metrics"]
    assert "test_brier_score" in summary["logged_metrics"]
    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved_summary["run_id"] == summary["run_id"]

    run = mlflow.get_run(summary["run_id"])
    assert run.data.params["C"] == "0.01"
    assert run.data.params["class_weight"] == "None"
    assert run.data.params["decision_threshold"] == "0.15"
    assert run.data.metrics["validation_roc_auc"] == summary["logged_metrics"]["validation_roc_auc"]