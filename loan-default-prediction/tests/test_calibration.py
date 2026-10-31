"""Tests for probability metrics and training-only calibration candidates."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.models.calibration import compare_calibration, probability_metrics


def test_probability_metrics_and_calibration_candidates_are_valid() -> None:
    features, target = make_classification(
        n_samples=400,
        n_features=8,
        n_informative=5,
        weights=[0.75, 0.25],
        random_state=42,
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        features, target, test_size=0.3, stratify=target, random_state=42
    )
    estimator = LogisticRegression(max_iter=1000, random_state=42).fit(x_train, y_train)
    probabilities = estimator.predict_proba(x_validation)[:, 1]
    metrics = probability_metrics(y_validation, probabilities)
    assert 0 <= metrics["Brier_Score"] <= 1
    assert 0 <= metrics["ROC_AUC"] <= 1
    assert 0 <= metrics["PR_AUC"] <= 1

    selected, method, analysis = compare_calibration(
        estimator, x_train, y_train, x_validation, y_validation
    )
    assert method in {None, "sigmoid", "isotonic"}
    assert "uncalibrated" in analysis["metrics"]
    assert {"sigmoid", "isotonic"}.issubset(analysis["candidates"])
    for candidate in [selected, *analysis["candidates"].values()]:
        calibrated_probabilities = candidate.predict_proba(x_validation)[:, 1]
        assert np.all((calibrated_probabilities >= 0) & (calibrated_probabilities <= 1))
