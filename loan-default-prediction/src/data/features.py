"""Small, deterministic features available at loan application time."""

import numpy as np
import pandas as pd

ENGINEERED_FEATURES = [
    "DTI_Category",
    "IncomeLoanRatio",
    "LoanIncomeRatio",
    "EmploymentCategory",
    "CreditRiskCategory",
]


def engineer_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with five features derived only from input predictors.

    DTI: Low <= 0.20, Medium <= 0.40, High > 0.40.
    Employment: New < 24 months, Moderate < 60 months, Experienced >= 60.
    Credit score bands: Poor < 580, Fair < 670, Good < 740, VeryGood >= 740.
    Undefined ratios (missing or non-positive denominator) remain missing for
    the downstream numerical imputer to handle.
    """
    features = dataframe.copy()
    if "Default" in features.columns:
        features = features.drop(columns="Default")
    if "DTIRatio" in features:
        features["DTI_Category"] = pd.cut(
            features["DTIRatio"],
            bins=[-np.inf, 0.20, 0.40, np.inf],
            labels=["Low", "Medium", "High"],
            include_lowest=True,
        ).astype("object")
    if {"Income", "LoanAmount"}.issubset(features.columns):
        amount = features["LoanAmount"].where(features["LoanAmount"] > 0)
        features["IncomeLoanRatio"] = features["Income"] / amount
    if {"Income", "LoanAmount"}.issubset(features.columns):
        income = features["Income"].where(features["Income"] > 0)
        features["LoanIncomeRatio"] = features["LoanAmount"] / income
    if "MonthsEmployed" in features:
        features["EmploymentCategory"] = pd.cut(
            features["MonthsEmployed"],
            bins=[-np.inf, 24, 60, np.inf],
            labels=["New", "Moderate", "Experienced"],
            right=False,
        ).astype("object")
    if "CreditScore" in features:
        features["CreditRiskCategory"] = pd.cut(
            features["CreditScore"],
            bins=[-np.inf, 580, 670, 740, np.inf],
            labels=["Poor", "Fair", "Good", "VeryGood"],
            right=False,
        ).astype("object")
    return features
