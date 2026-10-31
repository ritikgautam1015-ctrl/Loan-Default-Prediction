"""Tests for operating-threshold analysis and F1-first selection."""

import numpy as np
import pandas as pd

from src.models.threshold import analyze_thresholds, select_operating_threshold


def test_threshold_metrics_and_confusion_counts_are_consistent() -> None:
    target = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])
    results = analyze_thresholds(target, probabilities, thresholds=[0.5])
    row = results.iloc[0]
    assert row["TP"] == 2
    assert row["TN"] == 2
    assert row["FP"] == 0
    assert row["FN"] == 0
    assert row["Precision"] == row["Recall"] == row["F1"] == 1.0
    assert row["Specificity"] == row["True_Positive_Rate"] == 1.0
    assert row["False_Positive_Rate"] == row["False_Negative_Rate"] == 0.0


def test_threshold_analysis_rejects_invalid_probabilities() -> None:
    try:
        analyze_thresholds([0, 1], [0.2, 1.2])
    except ValueError as error:
        assert "between 0 and 1" in str(error)
    else:
        raise AssertionError("Expected invalid probabilities to be rejected.")


def test_threshold_selection_uses_f1_then_recall_tiebreak() -> None:
    metrics = pd.DataFrame(
        {
            "Threshold": [0.30, 0.40, 0.50],
            "F1": [0.800, 0.797, 0.700],
            "Recall": [0.60, 0.80, 0.50],
            "Precision": [0.90, 0.70, 1.00],
        }
    )
    selected = select_operating_threshold(metrics, f1_tolerance=0.005)
    assert selected == 0.40
