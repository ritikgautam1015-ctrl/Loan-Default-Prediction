"""Probability calibration comparison using training-only calibration fits."""

from copy import deepcopy

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.models.train import RANDOM_STATE

CALIBRATION_BRIER_IMPROVEMENT = 0.05
MAX_DISCRIMINATION_DROP = 0.01


def probability_metrics(target, probabilities) -> dict[str, float]:
    """Return ranking and probability-quality metrics."""
    actual = np.asarray(target, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    if probability.ndim != 1 or len(actual) != len(probability):
        raise ValueError("Target and probabilities must be equal-length one-dimensional arrays.")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probabilities must be finite values between 0 and 1.")
    return {
        "ROC_AUC": float(roc_auc_score(actual, probability)),
        "PR_AUC": float(average_precision_score(actual, probability)),
        "Brier_Score": float(brier_score_loss(actual, probability)),
    }


def compare_calibration(
    estimator,
    train_features,
    train_target,
    validation_features,
    validation_target,
    minimum_brier_improvement: float = CALIBRATION_BRIER_IMPROVEMENT,
    maximum_auc_drop: float = MAX_DISCRIMINATION_DROP,
) -> tuple[object, str | None, dict[str, object]]:
    """Compare sigmoid/isotonic calibrators fit on training data only.

    Calibration is accepted only when Brier score improves by at least 5% and
    neither ROC-AUC nor PR-AUC falls by more than the specified tolerance.
    """
    baseline_probability = estimator.predict_proba(validation_features)[:, 1]
    baseline_metrics = probability_metrics(validation_target, baseline_probability)
    candidates: dict[str, object] = {}
    results: dict[str, dict[str, float]] = {"uncalibrated": baseline_metrics}
    accepted: list[tuple[str, object, dict[str, float]]] = []

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    for method in ("sigmoid", "isotonic"):
        calibrated = CalibratedClassifierCV(
            estimator=deepcopy(estimator), method=method, cv=cv, n_jobs=-1
        )
        calibrated.fit(train_features, train_target)
        calibrated_probability = calibrated.predict_proba(validation_features)[:, 1]
        metrics = probability_metrics(validation_target, calibrated_probability)
        candidates[method] = calibrated
        results[method] = metrics
        sufficient_brier_gain = metrics["Brier_Score"] <= baseline_metrics["Brier_Score"] * (
            1 - minimum_brier_improvement
        )
        acceptable_ranking = (
            metrics["ROC_AUC"] >= baseline_metrics["ROC_AUC"] - maximum_auc_drop
            and metrics["PR_AUC"] >= baseline_metrics["PR_AUC"] - maximum_auc_drop
        )
        if sufficient_brier_gain and acceptable_ranking:
            accepted.append((method, calibrated, metrics))

    if accepted:
        method, selected, metrics = min(accepted, key=lambda candidate: candidate[2]["Brier_Score"])
        results["selected"] = metrics
        return selected, method, {"metrics": results, "candidates": candidates}
    results["selected"] = baseline_metrics
    return estimator, None, {"metrics": results, "candidates": candidates}