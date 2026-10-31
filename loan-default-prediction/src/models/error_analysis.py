"""Validation error, grouped performance, risk-band, and scenario summaries."""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

NUMERICAL_FEATURES = [
    "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
    "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio",
]
CATEGORICAL_FEATURES = [
    "Education", "EmploymentType", "MaritalStatus", "HasMortgage",
    "HasDependents", "LoanPurpose", "HasCoSigner",
]
ANALYSIS_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def create_error_analysis(original_features, target, probabilities, threshold: float) -> pd.DataFrame:
    """Attach probability, predicted class, and error type to original rows."""
    actual = np.asarray(target, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    if len(original_features) != len(actual) or len(actual) != len(probability):
        raise ValueError("Original features, target, and probabilities must have equal lengths.")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probabilities must be finite values between 0 and 1.")
    predicted = (probability >= threshold).astype(int)
    types = np.select(
        [(actual == 0) & (predicted == 0), (actual == 1) & (predicted == 1),
         (actual == 0) & (predicted == 1)],
        ["Correct Negative", "Correct Positive", "False Positive"],
        default="False Negative",
    )
    result = original_features.reset_index(drop=True).copy()
    result["Actual"] = actual
    result["Predicted"] = predicted
    result["Probability"] = probability
    result["Prediction"] = np.where(predicted == 1, "Default", "No Default")
    result["Error_Type"] = types
    return result


def confusion_counts(error_frame: pd.DataFrame) -> dict[str, int]:
    """Return TN, FP, FN, TP counts for a row-level error frame."""
    matrix = confusion_matrix(error_frame["Actual"], error_frame["Predicted"], labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


def _group_labels(values: pd.Series, numeric: bool, minimum_group_size: int) -> pd.Series:
    if numeric:
        unique_count = values.nunique(dropna=True)
        if unique_count < 2:
            labels = pd.Series("All", index=values.index, dtype="object")
        else:
            quantiles = min(5, unique_count)
            labels = pd.qcut(values, q=quantiles, duplicates="drop").astype("object")
        labels = labels.where(values.notna(), "Missing")
    else:
        labels = values.astype("string").fillna("Missing").astype("object")
    labels = labels.astype("string").fillna("Missing")
    counts = labels.value_counts(dropna=False)
    small = counts[counts < minimum_group_size].index
    return labels.where(~labels.isin(small), "Other")


def grouped_error_analysis(
    error_frame: pd.DataFrame,
    features: list[str] = ANALYSIS_FEATURES,
    minimum_group_size: int = 100,
) -> pd.DataFrame:
    """Summarize default/prediction/error rates across sufficiently large groups."""
    rows: list[dict[str, object]] = []
    numeric_set = set(NUMERICAL_FEATURES)
    for feature in features:
        if feature not in error_frame:
            continue
        grouped = error_frame.copy()
        grouped["Group"] = _group_labels(
            grouped[feature], feature in numeric_set, minimum_group_size
        )
        for label, group in grouped.groupby("Group", observed=True, dropna=False):
            positives = int(group["Actual"].sum())
            negatives = len(group) - positives
            false_negatives = int(((group["Actual"] == 1) & (group["Predicted"] == 0)).sum())
            false_positives = int(((group["Actual"] == 0) & (group["Predicted"] == 1)).sum())
            rows.append(
                {
                    "Feature": feature,
                    "Group": str(label),
                    "Sample_Count": int(len(group)),
                    "Default_Rate": float(group["Actual"].mean()),
                    "Predicted_Default_Rate": float(group["Predicted"].mean()),
                    "False_Negative_Count": false_negatives,
                    "False_Negative_Rate": float(false_negatives / positives) if positives else 0.0,
                    "False_Positive_Count": false_positives,
                    "False_Positive_Rate": float(false_positives / negatives) if negatives else 0.0,
                }
            )
    return pd.DataFrame(rows)


def stability_analysis(
    error_frame: pd.DataFrame,
    probabilities,
    features: list[str] | None = None,
    minimum_group_size: int = 200,
) -> pd.DataFrame:
    """Compare validation AUC, default rate, and recall across selected groups."""
    from sklearn.metrics import roc_auc_score, recall_score

    probabilities = np.asarray(probabilities, dtype=float)
    selected = features or ["Age", "Income", "CreditScore", "LoanPurpose", "EmploymentType"]
    rows = []
    numeric_set = set(NUMERICAL_FEATURES)
    for feature in selected:
        if feature not in error_frame:
            continue
        labels = _group_labels(error_frame[feature], feature in numeric_set, minimum_group_size)
        for label, indices in labels.groupby(labels, observed=True).groups.items():
            positions = np.asarray(list(indices), dtype=int)
            group = error_frame.iloc[positions]
            actual = group["Actual"].to_numpy(dtype=int)
            group_probability = probabilities[positions]
            if len(group) < minimum_group_size:
                continue
            rows.append(
                {
                    "Feature": feature,
                    "Group": str(label),
                    "Sample_Count": int(len(group)),
                    "Default_Rate": float(actual.mean()),
                    "Recall": float(recall_score(actual, group["Predicted"], zero_division=0)),
                    "ROC_AUC": float(roc_auc_score(actual, group_probability))
                    if np.unique(actual).size == 2 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def risk_band_analysis(
    original_features: pd.DataFrame,
    target,
    probabilities,
    selected_threshold: float,
    band_width: float = 0.05,
) -> pd.DataFrame:
    """Summarize simple risk bands; medium spans the threshold neighborhood."""
    actual = np.asarray(target, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    low_cutoff = max(0.0, selected_threshold - band_width)
    high_cutoff = min(1.0, selected_threshold + band_width)
    band = np.select(
        [probability < low_cutoff, probability < high_cutoff],
        ["Low Risk", "Medium Risk"],
        default="High Risk",
    )
    exposure = (
        pd.to_numeric(original_features["LoanAmount"], errors="coerce").to_numpy(dtype=float)
        if "LoanAmount" in original_features else np.full(len(actual), np.nan)
    )
    records = []
    for name in ("Low Risk", "Medium Risk", "High Risk"):
        mask = band == name
        records.append(
            {
                "Risk_Band": name,
                "Customer_Count": int(mask.sum()),
                "Actual_Defaults": int(actual[mask].sum()),
                "Default_Rate": float(actual[mask].mean()) if mask.any() else 0.0,
                "Average_Predicted_Probability": float(probability[mask].mean()) if mask.any() else 0.0,
                "Total_Loan_Exposure": float(np.nansum(exposure[mask])),
            }
        )
    return pd.DataFrame(records)


def simulate_intervention(
    target,
    probabilities,
    loan_amount,
    threshold: float,
    intervention_success_rate: float = 0.30,
) -> dict[str, float | int]:
    """Estimate a scenario outcome from captured validation defaults only."""
    actual = np.asarray(target, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    exposure = np.asarray(loan_amount, dtype=float)
    if not (len(actual) == len(probability) == len(exposure)):
        raise ValueError("Target, probabilities, and exposure must have equal lengths.")
    if not 0 <= intervention_success_rate <= 1:
        raise ValueError("Intervention success rate must be between 0 and 1.")
    predicted_high_risk = probability >= threshold
    defaults = actual == 1
    captured = defaults & predicted_high_risk
    missed = defaults & ~predicted_high_risk
    actual_defaults = int(defaults.sum())
    captured_defaults = int(captured.sum())
    prevented = captured_defaults * intervention_success_rate
    return {
        "actual_defaults": actual_defaults,
        "predicted_high_risk_borrowers": int(predicted_high_risk.sum()),
        "defaults_captured": captured_defaults,
        "potentially_preventable_defaults": captured_defaults,
        "intervention_success_rate_assumed": float(intervention_success_rate),
        "estimated_prevented_defaults": float(prevented),
        "estimated_remaining_defaults": float(max(0, actual_defaults - prevented)),
        "estimated_npa_reduction": float(prevented / actual_defaults) if actual_defaults else 0.0,
        "actual_default_loan_exposure": float(np.nansum(exposure[defaults])),
        "captured_default_loan_exposure": float(np.nansum(exposure[captured])),
        "missed_default_loan_exposure": float(np.nansum(exposure[missed])),
        "classification_threshold": float(threshold),
        "label": "SIMULATED / ASSUMED; not measured real-world NPA reduction",
    }