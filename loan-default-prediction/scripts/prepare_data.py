"""Run Phase 2 validation, feature preparation, splitting and preprocessing."""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.features import ENGINEERED_FEATURES, engineer_features  # noqa: E402
from src.data.loader import DATASET_PATH, EXPECTED_COLUMNS, load_dataset, validate_dataset  # noqa: E402
from src.data.preprocessing import (  # noqa: E402
    RANDOM_STATE,
    build_preprocessor,
    feature_frame,
    get_feature_columns,
    split_dataset,
    transformed_frame,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
NUMERICAL_COLUMNS = [
    "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
    "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio",
]
CATEGORICAL_COLUMNS = [
    "Education", "EmploymentType", "MaritalStatus", "HasMortgage",
    "HasDependents", "LoanPurpose", "HasCoSigner",
]


def validate_types(dataframe: pd.DataFrame) -> None:
    """Reject unparseable numeric values with sample offending values."""
    for column in NUMERICAL_COLUMNS + ["Default"]:
        converted = pd.to_numeric(dataframe[column], errors="coerce")
        invalid = dataframe[column].notna() & converted.isna()
        if invalid.any():
            examples = dataframe.loc[invalid, column].astype(str).unique()[:5].tolist()
            raise ValueError(f"Column {column!r} contains non-numeric values: {examples}")
        dataframe[column] = converted
    dataframe["Default"] = dataframe["Default"].astype("int64")


def quality_report(dataframe: pd.DataFrame) -> dict[str, object]:
    """Collect missingness, duplicates, domain flags, IQR and category stats."""
    missing = dataframe.isna().sum()
    missing_table = pd.DataFrame(
        {"Missing Count": missing, "Missing %": missing / len(dataframe) * 100}
    )
    print("\nMissing values by column:\n", missing_table.to_string())
    if int(missing.sum()) == 0:
        print("No missing-value imputation is required for the current dataset.")

    duplicate_rows = int(dataframe.duplicated().sum())
    duplicate_ids = int(dataframe["LoanID"].duplicated(keep=False).sum())
    print(f"Total duplicate rows: {duplicate_rows}")
    print(f"Rows with duplicate LoanID values: {duplicate_ids}")

    checks = {
        "Age <= 0": dataframe["Age"] <= 0,
        "Age < 18": dataframe["Age"] < 18,
        "Age > 100": dataframe["Age"] > 100,
        "Income <= 0": dataframe["Income"] <= 0,
        "LoanAmount <= 0": dataframe["LoanAmount"] <= 0,
        "CreditScore < 300": dataframe["CreditScore"] < 300,
        "CreditScore > 850": dataframe["CreditScore"] > 850,
        "MonthsEmployed < 0": dataframe["MonthsEmployed"] < 0,
        "NumCreditLines < 0": dataframe["NumCreditLines"] < 0,
        "InterestRate < 0": dataframe["InterestRate"] < 0,
        "LoanTerm <= 0": dataframe["LoanTerm"] <= 0,
        "DTIRatio < 0": dataframe["DTIRatio"] < 0,
    }
    domain_counts = {name: int(mask.sum()) for name, mask in checks.items()}
    print("\nSuspicious domain values (reported, not removed):")
    for name, count in domain_counts.items():
        print(f"{name}: {count}")

    outlier_counts: dict[str, int] = {}
    for column in NUMERICAL_COLUMNS:
        values = dataframe[column].dropna()
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        count = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())
        outlier_counts[column] = count
        print(f"IQR outliers, {column}: {count} (retained for review)")

    category_summaries: dict[str, dict[str, int]] = {}
    formatting_changes: dict[str, int] = {}
    for column in CATEGORICAL_COLUMNS:
        before = dataframe[column].astype("string")
        stripped = before.str.strip()
        formatting_changes[column] = int((before != stripped).fillna(False).sum())
        dataframe[column] = stripped.astype("object")
        category_summaries[column] = {
            str(key): int(value) for key, value in dataframe[column].value_counts(dropna=False).items()
        }
        print(f"\n{column} frequencies:\n{dataframe[column].value_counts(dropna=False).to_string()}")
        normalized_values = dataframe[column].dropna().astype(str)
        casing_groups: dict[str, set[str]] = {}
        for value in normalized_values.unique():
            casing_groups.setdefault(value.casefold(), set()).add(value)
        if any(len(variants) > 1 for variants in casing_groups.values()):
            print(f"Formatting review: {column} has mixed capitalization; values were preserved.")
    return {
        "missing_by_column": {str(key): int(value) for key, value in missing.items()},
        "duplicate_rows": duplicate_rows,
        "duplicate_loan_id_rows": duplicate_ids,
        "domain_flag_counts": domain_counts,
        "iqr_outlier_counts": outlier_counts,
        "categorical_value_counts": category_summaries,
        "trimmed_category_values": formatting_changes,
    }


def main() -> int:
    print("=" * 40)
    print("PHASE 2 - DATA PREPARATION")
    print("=" * 40)
    print("\nLoading dataset...")
    dataframe = load_dataset()
    print(f"Rows: {len(dataframe)}\nColumns: {len(dataframe.columns)}")

    print("\nValidating data...")
    validate_dataset(dataframe)
    if list(dataframe.columns) != EXPECTED_COLUMNS:
        raise ValueError("Dataset columns do not match the expected order.")
    validate_types(dataframe)
    quality = quality_report(dataframe)
    print("\nValidation: PASS")

    duplicate_count = int(dataframe.duplicated().sum())
    if duplicate_count:
        dataframe = dataframe.drop_duplicates().copy()
        print(f"Removed {duplicate_count} exact duplicate rows from prepared data.")
    print("\nRemoving identifier...")
    print("Removed: LoanID")
    features = engineer_features(dataframe.drop(columns=["LoanID", "Default"]))
    # Keep target alongside features only for stratified partitioning.
    model_data = features.copy()
    model_data["Default"] = dataframe["Default"]
    print("\nCreating features...")
    for name in ENGINEERED_FEATURES:
        print(f"- {name}")
    print("DTI thresholds: Low <= 0.20, Medium <= 0.40, High > 0.40")
    print("Employment thresholds (months): New < 24, Moderate 24-59, Experienced >= 60")
    print("Credit score bands: Poor < 580, Fair 580-669, Good 670-739, VeryGood >= 740")

    train, validation, test = split_dataset(model_data)
    print("\nSplitting dataset...")
    print(f"Train: {len(train)} ({len(train) / len(model_data):.2%})")
    print(f"Validation: {len(validation)} ({len(validation) / len(model_data):.2%})")
    print(f"Test: {len(test)} ({len(test) / len(model_data):.2%})")

    numerical, categorical = get_feature_columns(features)
    preprocessor = build_preprocessor(numerical, categorical)
    train_x, train_y = feature_frame(train)
    validation_x, validation_y = feature_frame(validation)
    test_x, test_y = feature_frame(test)
    print("\nFitting preprocessing pipeline...")
    preprocessor.fit(train_x)
    transformed_train = transformed_frame(preprocessor, train_x, train_y)
    transformed_validation = transformed_frame(preprocessor, validation_x, validation_y)
    transformed_test = transformed_frame(preprocessor, test_x, test_y)
    print("Preprocessing complete.")

    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    print("\nSaving processed datasets...")
    transformed_train.to_csv(PROCESSED_PATH / "train.csv", index=False)
    transformed_validation.to_csv(PROCESSED_PATH / "validation.csv", index=False)
    transformed_test.to_csv(PROCESSED_PATH / "test.csv", index=False)
    joblib.dump(preprocessor, PROCESSED_PATH / "preprocessor.joblib")
    metadata = {
        "numerical_features": numerical,
        "categorical_features": categorical,
        "engineered_features": ENGINEERED_FEATURES,
        "target": "Default",
        "dropped_columns": ["LoanID"],
        "random_state": RANDOM_STATE,
        "train_size": len(train) / len(model_data),
        "validation_size": len(validation) / len(model_data),
        "test_size": len(test) / len(model_data),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "output_feature_columns": list(transformed_train.columns[:-1]),
        "quality_report": quality,
    }
    (PROCESSED_PATH / "feature_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    for name in ("train.csv", "validation.csv", "test.csv", "preprocessor.joblib", "feature_metadata.json"):
        print(name)
    print("\nPhase 2 completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
