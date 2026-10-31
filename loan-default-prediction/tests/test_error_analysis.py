"""Tests for confusion labels, grouped errors, risk bands, and NPA scenarios."""

import numpy as np
import pandas as pd

from src.models.error_analysis import (
    confusion_counts,
    create_error_analysis,
    risk_band_analysis,
    simulate_intervention,
)


def test_error_types_and_confusion_counts() -> None:
    features = pd.DataFrame({"Age": [21, 30, 40, 55], "LoanAmount": [100, 200, 300, 400]})
    errors = create_error_analysis(features, [0, 0, 1, 1], [0.1, 0.8, 0.9, 0.2], 0.5)
    assert errors["Error_Type"].tolist() == [
        "Correct Negative", "False Positive", "Correct Positive", "False Negative"
    ]
    assert errors["Predicted"].tolist() == [0, 1, 1, 0]
    assert errors["Prediction"].tolist() == ["No Default", "Default", "Default", "No Default"]
    assert confusion_counts(errors) == {"TN": 1, "FP": 1, "FN": 1, "TP": 1}


def test_risk_bands_and_intervention_scenario_are_bounded() -> None:
    features = pd.DataFrame({"LoanAmount": [100, 200, 300, 400, 500]})
    target = np.array([0, 1, 1, 0, 1])
    probabilities = np.array([0.05, 0.35, 0.45, 0.62, 0.9])
    bands = risk_band_analysis(features, target, probabilities, selected_threshold=0.4)
    assert set(bands["Risk_Band"]) == {"Low Risk", "Medium Risk", "High Risk"}
    assert bands["Customer_Count"].sum() == len(target)

    scenario = simulate_intervention(
        target, probabilities, features["LoanAmount"], threshold=0.4,
        intervention_success_rate=0.3,
    )
    assert scenario["estimated_prevented_defaults"] <= scenario["actual_defaults"]
    assert scenario["estimated_remaining_defaults"] >= 0
    assert 0 <= scenario["estimated_npa_reduction"] <= 1
    assert "SIMULATED / ASSUMED" in scenario["label"]
