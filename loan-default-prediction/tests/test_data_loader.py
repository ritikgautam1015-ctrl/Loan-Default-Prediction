"""Tests for raw dataset loading and validation."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import (
    EXPECTED_COLUMNS,
    get_data_quality,
    load_dataset,
    validate_dataset,
)


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame([{column: 1 for column in EXPECTED_COLUMNS}])


def test_dataset_loads_successfully(tmp_path: Path) -> None:
    dataframe = valid_frame()
    path = tmp_path / "Loan_default.csv"
    dataframe.to_csv(path, index=False)
    assert load_dataset(path).shape == (1, len(EXPECTED_COLUMNS))


def test_dataset_has_exact_expected_columns() -> None:
    assert list(valid_frame().columns) == EXPECTED_COLUMNS


def test_default_exists() -> None:
    assert "Default" in valid_frame().columns


def test_default_accepts_only_binary_values() -> None:
    dataframe = pd.DataFrame([dict.fromkeys(EXPECTED_COLUMNS, 0), dict.fromkeys(EXPECTED_COLUMNS, 1)])
    validate_dataset(dataframe)
    dataframe.loc[0, "Default"] = 2
    with pytest.raises(ValueError, match="binary values"):
        validate_dataset(dataframe)


def test_empty_dataset_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_dataset(pd.DataFrame(columns=EXPECTED_COLUMNS))


def test_missing_columns_are_reported() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        validate_dataset(valid_frame().drop(columns="Age"))


def test_missing_value_detection() -> None:
    dataframe = valid_frame()
    dataframe.loc[0, "Income"] = float("nan")
    assert get_data_quality(dataframe)["missing_values"] == 1


def test_duplicate_row_detection() -> None:
    dataframe = pd.concat([valid_frame(), valid_frame()], ignore_index=True)
    assert get_data_quality(dataframe)["duplicate_rows"] == 1
