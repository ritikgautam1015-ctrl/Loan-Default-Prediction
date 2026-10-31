"""Evidently reports for feature, prediction, target drift, and data quality."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from sklearn.pipeline import Pipeline

from monitoring.config import (
    CURRENT_DATA_PATH,
    DRIFT_SHARE_THRESHOLD,
    DRIFT_TEST_THRESHOLD,
    FEATURE_METADATA_PATH,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    REFERENCE_DATA_PATH,
    REPORT_MONITORING_PATH,
)
from monitoring.data_loader import data_quality_summary, load_monitoring_data
from src.models.evaluate import evaluate_probabilities


def _drift_metrics(
    snapshot: Any,
) -> tuple[dict[str, bool], dict[str, float], dict[str, str]]:
    drifted: dict[str, bool] = {}
    values: dict[str, float] = {}
    methods: dict[str, str] = {}
    for metric in snapshot.dict().get("metrics", []):
        config = metric.get("config", {})
        if not str(config.get("type", "")).endswith(":ValueDrift"):
            continue
        column = config.get("column")
        if not column:
            continue
        value = float(metric.get("value", 0.0))
        method = str(config.get("method", ""))
        threshold = float(config.get("threshold", DRIFT_TEST_THRESHOLD))
        method_lower = method.lower().replace("-", "_").replace(" ", "_")
        if "p_value" in method_lower:
            is_drifted = value < threshold
        else:
            is_drifted = value > threshold
        drifted[str(column)] = is_drifted
        values[str(column)] = value
        methods[str(column)] = method
    return drifted, values, methods


def predict_processed(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Use the exact Phase 4 classifier on already Phase 2-transformed inputs."""
    if isinstance(model, Pipeline):
        classifier = model.steps[-1][1]
        model_columns = list(getattr(classifier, "feature_names_in_", []))
        if model_columns and set(model_columns).issubset(features.columns):
            return classifier.predict_proba(features.loc[:, model_columns])[:, 1]
        return model.predict_proba(features)[:, 1]
    return model.predict_proba(features)[:, 1]


def _load_feature_metadata() -> dict[str, Any]:
    if FEATURE_METADATA_PATH.is_file():
        import json

        return json.loads(FEATURE_METADATA_PATH.read_text(encoding="utf-8"))
    return {}


def _append_html_note(path: Path, section: str) -> None:
    html = path.read_text(encoding="utf-8")
    closing_body = html.lower().rfind("</body>")
    if closing_body >= 0:
        html = html[:closing_body] + section + html[closing_body:]
    else:
        html += section
    path.write_text(html, encoding="utf-8")


def run_monitoring(
    reference_path: str | Path = REFERENCE_DATA_PATH,
    current_path: str | Path = CURRENT_DATA_PATH,
    model_path: str | Path = MODEL_PATH,
    metadata_path: str | Path = MODEL_METADATA_PATH,
    preprocessor_path: str | Path = PREPROCESSOR_PATH,
    output_path: str | Path = REPORT_MONITORING_PATH,
) -> dict[str, Any]:
    """Compare a current processed batch with training reference data.

    Current rows may omit Default. When labels exist, target drift and standard
    current-batch metrics are also reported. This function never fits a model.
    """
    import json

    reference_path = Path(reference_path)
    current_path = Path(current_path)
    metadata_path = Path(metadata_path)
    model_path = Path(model_path)
    preprocessor_path = Path(preprocessor_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    data = load_monitoring_data(reference_path, current_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Final Phase 4 model not found: {model_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Phase 4 model metadata not found: {metadata_path}")
    if not preprocessor_path.is_file():
        raise FileNotFoundError(f"Phase 2 fitted preprocessor not found: {preprocessor_path}")
    model = joblib.load(model_path)
    # Loading verifies the expected fitted Phase 2 artifact is present; no refit occurs.
    joblib.load(preprocessor_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    threshold = float(metadata["classification_threshold"])

    reference_features = data.reference.loc[:, data.feature_columns]
    current_features = data.current.loc[:, data.feature_columns]
    reference_probability = predict_processed(model, reference_features)
    current_probability = predict_processed(model, current_features)
    reference_monitoring = reference_features.copy()
    current_monitoring = current_features.copy()
    reference_monitoring["prediction_probability"] = reference_probability
    current_monitoring["prediction_probability"] = current_probability
    reference_monitoring["prediction"] = (reference_probability >= threshold).astype(int)
    current_monitoring["prediction"] = (current_probability >= threshold).astype(int)
    labels_available = "Default" in data.reference and "Default" in data.current
    if labels_available:
        reference_monitoring["Default"] = data.reference["Default"].to_numpy()
        current_monitoring["Default"] = data.current["Default"].to_numpy()

    prediction_columns = ["prediction_probability", "prediction"]
    metrics = [
        DataDriftPreset(
            columns=data.feature_columns,
            drift_share=DRIFT_SHARE_THRESHOLD,
            threshold=DRIFT_TEST_THRESHOLD,
        ),
        DataDriftPreset(
            columns=prediction_columns,
            drift_share=DRIFT_SHARE_THRESHOLD,
            threshold=DRIFT_TEST_THRESHOLD,
        ),
    ]
    if labels_available:
        metrics.append(
            DataDriftPreset(
                columns=["Default"],
                drift_share=DRIFT_SHARE_THRESHOLD,
                threshold=DRIFT_TEST_THRESHOLD,
            )
        )
    snapshot = Report(metrics).run(
        current_data=current_monitoring,
        reference_data=reference_monitoring,
    )
    snapshot.save_html(str(output_path / "drift_report.html"))

    quality_snapshot = Report([DataSummaryPreset()]).run(current_data=current_monitoring)
    quality_snapshot.save_html(str(output_path / "data_quality_report.html"))

    drifted_columns, drift_values, drift_methods = _drift_metrics(snapshot)
    drifted_features = [
        feature for feature in data.feature_columns if drifted_columns.get(feature, False)
    ]
    drifted_predictions = [
        feature for feature in prediction_columns if drifted_columns.get(feature, False)
    ]
    target_drift_value = drift_values.get("Default") if labels_available else None
    target_drift_method = drift_methods.get("Default") if labels_available else None
    feature_share = len(drifted_features) / len(data.feature_columns) if data.feature_columns else 0.0
    prediction_share = len(drifted_predictions) / len(prediction_columns)

    reference_quality = data_quality_summary(reference_monitoring)
    current_quality = data_quality_summary(current_monitoring, reference_monitoring)
    schema_differences = data.schema_differences
    quality_summary = {
        "reference": reference_quality,
        "current": current_quality,
        "schema_differences": schema_differences,
        "schema_mismatch": bool(
            schema_differences["missing_from_current"]
            or schema_differences["unexpected_in_current"]
            or schema_differences["dtype_mismatches"]
        ),
    }
    summary: dict[str, Any] = {
        "reference_rows": int(len(data.reference)),
        "current_rows": int(len(data.current)),
        "reference_data_path": str(reference_path),
        "current_data_path": str(current_path),
        "model_version": str(metadata.get("model_version", "unknown")),
        "decision_threshold": threshold,
        "feature_drift_detected": feature_share >= DRIFT_SHARE_THRESHOLD,
        "feature_drift_share": float(feature_share),
        "feature_drift_share_threshold": DRIFT_SHARE_THRESHOLD,
        "feature_drift_metric_threshold": DRIFT_TEST_THRESHOLD,
        "feature_drift_values": {
            feature: drift_values[feature]
            for feature in data.feature_columns if feature in drift_values
        },
        "drifted_features": drifted_features,
        "prediction_drift_detected": prediction_share >= DRIFT_SHARE_THRESHOLD,
        "prediction_drift_share": float(prediction_share),
        "drifted_prediction_columns": drifted_predictions,
        "target_drift_available": bool(labels_available),
        "target_drift_detected": bool(
            labels_available
            and target_drift_value is not None
            and drifted_columns.get("Default", False)
        ),
        "target_drift_value": target_drift_value,
        "target_drift_method": target_drift_method,
        "target_drift_metric_threshold": DRIFT_TEST_THRESHOLD if labels_available else None,
        "data_quality": quality_summary,
        "monitoring_timestamp": datetime.now(timezone.utc).isoformat(),
        "interpretation": (
            "Drift is a distribution-change signal that may require investigation; "
            "it does not by itself prove model performance degradation."
        ),
    }

    performance_path = output_path / "performance_summary.json"
    if labels_available:
        current_metrics = evaluate_probabilities(
            data.current["Default"], current_probability, threshold=threshold
        )
        current_metrics["Brier_Score"] = float(
            np.mean((current_probability - data.current["Default"].to_numpy()) ** 2)
        )
        performance_path.write_text(
            json.dumps(current_metrics, indent=2), encoding="utf-8"
        )
        summary["current_performance_available"] = True
        summary["current_performance"] = current_metrics
    else:
        summary["current_performance_available"] = False
        if performance_path.exists():
            performance_path.unlink()

    drift_interpretation = (
        "<section style='font-family:Arial,sans-serif;padding:16px;margin:16px;"
        "background:#f3f6fa;border-left:4px solid #356a9a'>"
        "<h2>Monitoring interpretation</h2>"
        f"<p>Feature drift noteworthy: {summary['feature_drift_detected']} "
        f"({len(drifted_features)}/{len(data.feature_columns)} features flagged; "
        f"configured share threshold {DRIFT_SHARE_THRESHOLD:.0%}).</p>"
        f"<p>Prediction drift noteworthy: {summary['prediction_drift_detected']}.</p>"
        f"<p>Target drift available: {labels_available}; detected: "
        f"{summary['target_drift_detected']}.</p>"
        "<p>Drift indicates distribution changes and may warrant investigation; "
        "it does not by itself prove lower model accuracy or predictive performance.</p>"
        "</section>"
    )
    _append_html_note(output_path / "drift_report.html", drift_interpretation)
    quality_interpretation = (
        "<section style='font-family:Arial,sans-serif;padding:16px;margin:16px;"
        "background:#f3f6fa;border-left:4px solid #356a9a'>"
        "<h2>Reference/current data quality checks</h2>"
        f"<p>Current schema mismatch: {quality_summary['schema_mismatch']}.</p>"
        f"<p>Missing current values: {current_quality['missing_value_count']}; "
        f"duplicate current rows: {current_quality['duplicate_rows']}.</p>"
        f"<p>Missing model feature columns: "
        f"{schema_differences['missing_from_current']}.</p>"
        f"<p>Unexpected current columns: "
        f"{schema_differences['unexpected_in_current']}.</p>"
        f"<p>Feature dtype mismatches: {schema_differences['dtype_mismatches']}.</p>"
        "</section>"
    )
    _append_html_note(output_path / "data_quality_report.html", quality_interpretation)

    (output_path / "drift_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary