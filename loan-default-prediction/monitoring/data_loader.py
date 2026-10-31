"""Load and validate reference/current processed monitoring batches."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class MonitoringData:
    reference: pd.DataFrame
    current: pd.DataFrame
    feature_columns: list[str]
    schema_differences: dict[str, list[str]]


def load_monitoring_data(
    reference_path: str | Path,
    current_path: str | Path,
    target_column: str = "Default",
) -> MonitoringData:
    """Load aligned Phase 2 encoded datasets; reject missing model features."""
    reference_file = Path(reference_path)
    current_file = Path(current_path)
    for path in (reference_file, current_file):
        if not path.is_file():
            raise FileNotFoundError(f"Monitoring data file not found: {path}")

    reference = pd.read_csv(reference_file)
    current = pd.read_csv(current_file)
    if reference.empty or current.empty:
        raise ValueError("Reference and current monitoring datasets must not be empty.")

    feature_columns = [column for column in reference.columns if column != target_column]
    missing_features = sorted(set(feature_columns) - set(current.columns))
    if missing_features:
        raise ValueError(
            "Current dataset is missing required model input columns: "
            + ", ".join(missing_features)
        )

    missing_reference_columns = sorted(set(feature_columns) - set(current.columns))
    unexpected_current_columns = sorted(
        (set(current.columns) - set(reference.columns)) - {target_column}
    )
    dtype_mismatches = {
        str(column): {
            "reference": str(reference[column].dtype),
            "current": str(current[column].dtype),
        }
        for column in feature_columns
        if column in current and reference[column].dtype != current[column].dtype
    }
    reference_aligned = reference.loc[:, feature_columns + (
        [target_column] if target_column in reference.columns else []
    )].copy()
    current_columns = feature_columns + (
        [target_column] if target_column in current.columns else []
    )
    current_aligned = current.loc[:, current_columns].copy()

    return MonitoringData(
        reference=reference_aligned,
        current=current_aligned,
        feature_columns=feature_columns,
        schema_differences={
            "missing_from_current": missing_reference_columns,
            "unexpected_in_current": unexpected_current_columns,
            "dtype_mismatches": dtype_mismatches,
        },
    )


def data_quality_summary(
    data: pd.DataFrame, reference: pd.DataFrame | None = None
) -> dict[str, object]:
    """Calculate basic quality signals without mutating the input frame."""
    invalid_numeric: dict[str, int] = {}
    numeric_columns = (
        reference.select_dtypes(include=["number"]).columns
        if reference is not None
        else data.select_dtypes(include=["number"]).columns
    )
    for column in numeric_columns:
        if column not in data:
            continue
        converted = pd.to_numeric(data[column], errors="coerce")
        invalid = data[column].notna() & converted.isna()
        if invalid.any():
            invalid_numeric[str(column)] = int(invalid.sum())
    unexpected_categories: dict[str, list[str]] = {}
    if reference is not None:
        for column in reference.select_dtypes(include=["object", "string"]).columns:
            if column not in data:
                continue
            expected = set(reference[column].dropna().astype(str).unique())
            observed = set(data[column].dropna().astype(str).unique())
            unexpected = sorted(observed - expected)
            if unexpected:
                unexpected_categories[str(column)] = unexpected
    return {
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "missing_values_by_column": {
            str(column): int(count) for column, count in data.isna().sum().items()
        },
        "missing_value_count": int(data.isna().sum().sum()),
        "duplicate_rows": int(data.duplicated().sum()),
        "invalid_numeric_values_by_column": invalid_numeric,
        "unexpected_categorical_values_by_column": unexpected_categories,
        "dtypes": {str(column): str(dtype) for column, dtype in data.dtypes.items()},
    }