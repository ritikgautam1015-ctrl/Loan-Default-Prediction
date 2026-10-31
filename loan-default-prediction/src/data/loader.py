"""Load and inspect the untouched raw loan dataset."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "Loan_default.csv"
EXPECTED_COLUMNS = [
    "LoanID", "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
    "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio", "Education",
    "EmploymentType", "MaritalStatus", "HasMortgage", "HasDependents",
    "LoanPurpose", "HasCoSigner", "Default",
]
TARGET_COLUMN = "Default"


def load_dataset(path: Path | str = DATASET_PATH) -> pd.DataFrame:
    """Read the raw CSV without modifying its values."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    return pd.read_csv(dataset_path)


def validate_dataset(dataframe: pd.DataFrame) -> None:
    """Raise ValueError if required columns or binary target are invalid."""
    if dataframe.empty:
        raise ValueError("Dataset is empty.")
    missing = [column for column in EXPECTED_COLUMNS if column not in dataframe.columns]
    unexpected = [column for column in dataframe.columns if column not in EXPECTED_COLUMNS]
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing columns: {missing}")
        if unexpected:
            details.append(f"unexpected columns: {unexpected}")
        raise ValueError("Column validation failed (" + "; ".join(details) + ").")
    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing.")
    values = set(dataframe[TARGET_COLUMN].dropna().unique())
    if dataframe[TARGET_COLUMN].isna().any() or not values.issubset({0, 1}) or values != {0, 1}:
        raise ValueError(
            f"Target '{TARGET_COLUMN}' must contain both expected binary values 0 and 1 only; "
            f"found {sorted(values, key=str)}."
        )


def get_data_quality(dataframe: pd.DataFrame) -> dict[str, int]:
    """Return raw missing cell and duplicate row counts."""
    return {
        "missing_values": int(dataframe.isna().sum().sum()),
        "duplicate_rows": int(dataframe.duplicated().sum()),
    }


def print_dataset_summary(dataframe: pd.DataFrame) -> None:
    """Print counts, target distribution, quality metrics and dtypes."""
    default_counts = dataframe[TARGET_COLUMN].value_counts()
    default_count = int(default_counts.get(1, 0))
    non_default_count = int(default_counts.get(0, 0))
    quality = get_data_quality(dataframe)
    print("Dataset\n-------")
    print(f"Rows: {len(dataframe)}\nColumns: {len(dataframe.columns)}")
    print("\nTarget\n------")
    print(f"Target column: {TARGET_COLUMN}")
    print(f"Default = 0: {non_default_count}")
    print(f"Default = 1: {default_count}")
    print(f"Default rate: {default_count / len(dataframe) * 100:.2f}%")
    print("\nData Quality\n------------")
    print(f"Missing values: {quality['missing_values']}")
    print(f"Duplicate rows: {quality['duplicate_rows']}")
    print("\nData types\n----------")
    print(dataframe.dtypes.to_string())
