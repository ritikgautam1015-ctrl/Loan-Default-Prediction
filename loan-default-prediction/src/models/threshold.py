"""Validation-only threshold analysis and operating-point selection."""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

THRESHOLDS = np.round(np.arange(0.10, 0.901, 0.05), 2).tolist()


def _validated_arrays(target, probabilities) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(target, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    if actual.ndim != 1 or probability.ndim != 1 or len(actual) != len(probability):
        raise ValueError("Target and probabilities must be equal-length one-dimensional arrays.")
    if not np.isin(actual, [0, 1]).all():
        raise ValueError("Target values must be binary (0 or 1).")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probabilities must be finite values between 0 and 1.")
    return actual, probability


def analyze_thresholds(target, probabilities, thresholds=THRESHOLDS) -> pd.DataFrame:
    """Calculate classification metrics at each supplied probability threshold."""
    actual, probability = _validated_arrays(target, probabilities)
    rows = []
    for threshold in thresholds:
        if not 0 <= float(threshold) <= 1:
            raise ValueError("Every threshold must be between 0 and 1.")
        predicted = (probability >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if tn + fp else 0.0
        rows.append(
            {
                "Threshold": float(threshold),
                "Precision": float(precision_score(actual, predicted, zero_division=0)),
                "Recall": float(recall_score(actual, predicted, zero_division=0)),
                "F1": float(f1_score(actual, predicted, zero_division=0)),
                "Accuracy": float((predicted == actual).mean()),
                "Specificity": float(specificity),
                "False_Positive_Rate": float(fp / (fp + tn)) if fp + tn else 0.0,
                "False_Negative_Rate": float(fn / (fn + tp)) if fn + tp else 0.0,
                "True_Positive_Rate": float(tp / (tp + fn)) if tp + fn else 0.0,
                "TP": int(tp),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
            }
        )
    return pd.DataFrame(rows)


def select_operating_threshold(
    threshold_metrics: pd.DataFrame, f1_tolerance: float = 0.005
) -> float:
    """Choose by F1; within tolerance, prefer recall then precision.

    The tolerance makes near-ties explicit and favors catching more actual
    defaults without silently replacing F1 with a recall-only objective.
    """
    required = {"Threshold", "F1", "Recall", "Precision"}
    if not required.issubset(threshold_metrics.columns) or threshold_metrics.empty:
        raise ValueError(f"Threshold metrics must contain {sorted(required)} and at least one row.")
    if not 0 <= f1_tolerance < 1:
        raise ValueError("F1 tolerance must be in [0, 1).")
    maximum_f1 = float(threshold_metrics["F1"].max())
    candidates = threshold_metrics.loc[threshold_metrics["F1"] >= maximum_f1 - f1_tolerance]
    selected = candidates.sort_values(
        ["Recall", "Precision", "Threshold"], ascending=[False, False, False]
    ).iloc[0]
    return float(selected["Threshold"])