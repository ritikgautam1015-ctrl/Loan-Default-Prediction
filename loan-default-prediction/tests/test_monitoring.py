"""Lightweight tests for monitoring schemas, probabilities, and configuration."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from monitoring.config import (
    CURRENT_DATA_PATH,
    DRIFT_SHARE_THRESHOLD,
    MLFLOW_EXPERIMENT_NAME,
    REFERENCE_DATA_PATH,
)
from monitoring.data_loader import data_quality_summary, load_monitoring_data
from monitoring.evidently_monitor import _drift_metrics, predict_processed
from src.models.train import PROJECT_ROOT


def test_reference_and_current_monitoring_schemas_align(tmp_path) -> None:
    reference_path = tmp_path / "reference.csv"
    current_path = tmp_path / "current.csv"
    pd.DataFrame({"feature_a": [0.0, 1.0], "Default": [0, 1]}).to_csv(
        reference_path, index=False
    )
    pd.DataFrame({"feature_a": [2.0, 3.0]}).to_csv(current_path, index=False)
    loaded = load_monitoring_data(reference_path, current_path)
    assert loaded.feature_columns == ["feature_a"]
    assert "Default" in loaded.reference
    assert "Default" not in loaded.current
    assert loaded.schema_differences["missing_from_current"] == []


def test_missing_required_model_columns_fail_clearly(tmp_path) -> None:
    reference_path = tmp_path / "reference.csv"
    current_path = tmp_path / "current.csv"
    pd.DataFrame({"feature_a": [0.0], "feature_b": [1.0]}).to_csv(
        reference_path, index=False
    )
    pd.DataFrame({"feature_a": [2.0]}).to_csv(current_path, index=False)
    with pytest.raises(ValueError, match="missing required model input columns"):
        load_monitoring_data(reference_path, current_path)


def test_monitoring_probabilities_and_threshold_predictions_are_bounded() -> None:
    class DummyClassifier:
        feature_names_in_ = np.array(["feature_a", "feature_b"])

        def predict_proba(self, features):
            probabilities = np.array([0.10, 0.90])[: len(features)]
            return np.column_stack([1 - probabilities, probabilities])

    model = Pipeline([("preprocessor", "passthrough"), ("classifier", DummyClassifier())])

    features = pd.DataFrame({"feature_a": [1.0, 2.0], "feature_b": [0.0, 1.0]})
    probabilities = predict_processed(model, features)
    threshold = 0.15
    predictions = (probabilities >= threshold).astype(int)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert predictions.tolist() == [0, 1]


def test_quality_summary_and_monitor_configuration() -> None:
    quality = data_quality_summary(pd.DataFrame({"x": [1.0, None], "y": [1, 1]}))
    assert quality["missing_value_count"] == 1
    assert quality["duplicate_rows"] == 0
    assert REFERENCE_DATA_PATH == PROJECT_ROOT / "data" / "processed" / "train.csv"
    assert CURRENT_DATA_PATH == PROJECT_ROOT / "data" / "processed" / "validation.csv"
    assert DRIFT_SHARE_THRESHOLD == pytest.approx(0.30)
    assert MLFLOW_EXPERIMENT_NAME == "loan-default-prediction"


def test_drift_summary_json_has_expected_keys() -> None:
    payload = {
        "reference_rows": 10,
        "current_rows": 8,
        "feature_drift_detected": False,
        "prediction_drift_detected": False,
        "target_drift_available": False,
        "target_drift_detected": False,
        "drifted_features": [],
        "monitoring_timestamp": "2026-09-29T00:00:00+00:00",
    }
    serialized = json.loads(json.dumps(payload))
    assert {
        "reference_rows", "current_rows", "feature_drift_detected",
        "prediction_drift_detected", "target_drift_available",
        "target_drift_detected", "drifted_features", "monitoring_timestamp",
    }.issubset(serialized)


def test_evidently_drift_scores_respect_metric_direction() -> None:
    class FakeSnapshot:
        def dict(self):
            return {
                "metrics": [
                    {
                        "config": {
                            "type": "evidently:metric_v2:ValueDrift",
                            "column": "Default",
                            "method": "Jensen-Shannon distance",
                            "threshold": 0.05,
                        },
                        "value": 0.000001,
                    },
                    {
                        "config": {
                            "type": "evidently:metric_v2:ValueDrift",
                            "column": "Age",
                            "method": "chi-square p_value",
                            "threshold": 0.05,
                        },
                        "value": 0.001,
                    },
                ]
            }

    drifted, values, methods = _drift_metrics(FakeSnapshot())
    assert drifted == {"Default": False, "Age": True}
    assert values["Default"] == pytest.approx(0.000001)
    assert methods["Default"] == "Jensen-Shannon distance"