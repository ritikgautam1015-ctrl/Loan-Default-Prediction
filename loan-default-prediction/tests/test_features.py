"""Tests for deterministic, target-independent feature engineering."""

import pandas as pd

from src.data.features import ENGINEERED_FEATURES, engineer_features


def example_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "DTIRatio": [0.1, 0.3, 0.5],
            "Income": [1000, 0, 2000],
            "LoanAmount": [500, 250, 0],
            "MonthsEmployed": [12, 36, 72],
            "CreditScore": [550, 700, 780],
            "Default": [0, 1, 0],
        }
    )


def test_engineered_features_are_created_without_target() -> None:
    result = engineer_features(example_frame())
    assert set(ENGINEERED_FEATURES).issubset(result.columns)
    assert "Default" not in result.columns
    assert result["DTI_Category"].tolist() == ["Low", "Medium", "High"]
    assert result["EmploymentCategory"].tolist() == ["New", "Moderate", "Experienced"]
    assert result["CreditRiskCategory"].tolist() == ["Poor", "Good", "VeryGood"]


def test_ratios_are_safe_for_zero_denominators() -> None:
    result = engineer_features(example_frame())
    assert result.loc[1, "IncomeLoanRatio"] == 0
    assert pd.isna(result.loc[1, "LoanIncomeRatio"])
    assert pd.isna(result.loc[2, "IncomeLoanRatio"])
    assert result.loc[2, "LoanIncomeRatio"] == 0


def test_feature_engineering_does_not_change_input() -> None:
    original = example_frame()
    engineer_features(original)
    assert "IncomeLoanRatio" not in original
    assert "Default" in original
