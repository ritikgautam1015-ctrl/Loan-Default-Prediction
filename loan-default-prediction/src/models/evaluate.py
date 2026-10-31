"""Threshold-consistent binary classification metrics."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DEFAULT_THRESHOLD = 0.50


def evaluate_model(model, features, target, threshold: float = DEFAULT_THRESHOLD) -> dict[str, object]:
    """Return ranking and threshold-based metrics for Default=1."""
    probabilities = model.predict_proba(features)[:, 1]
    return evaluate_probabilities(target, probabilities, threshold)


def evaluate_probabilities(target, probabilities, threshold: float = DEFAULT_THRESHOLD) -> dict[str, object]:
    """Return metrics from one already-computed Default=1 probability vector."""
    target = np.asarray(target, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if target.ndim != 1 or probabilities.ndim != 1 or len(target) != len(probabilities):
        raise ValueError("Target and probabilities must be equal-length one-dimensional arrays.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite values between 0 and 1.")
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(target, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    return {
        "ROC_AUC": float(roc_auc_score(target, probabilities)),
        "PR_AUC": float(average_precision_score(target, probabilities)),
        "Accuracy": float(accuracy_score(target, predictions)),
        "Precision": float(precision_score(target, predictions, zero_division=0)),
        "Recall": float(recall_score(target, predictions, zero_division=0)),
        "F1": float(f1_score(target, predictions, zero_division=0)),
        "Confusion_Matrix": matrix.tolist(),
        "TN": int(true_negative),
        "FP": int(false_positive),
        "FN": int(false_negative),
        "TP": int(true_positive),
        "Threshold": float(threshold),
    }