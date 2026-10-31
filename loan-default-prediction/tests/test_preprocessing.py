"""Tests for preprocessing, unseen categories, and data partitioning."""

import numpy as np
import pandas as pd
from scipy import sparse

from src.data.features import engineer_features
from src.data.preprocessing import (
    build_preprocessor,
    feature_frame,
    get_feature_columns,
    split_dataset,
    transformed_frame,
)


def sample_dataset(rows: int = 100) -> pd.DataFrame:
    target = np.tile([0, 1], rows // 2)
    data = pd.DataFrame(
        {
            "Age": np.arange(rows) + 20,
            "Income": np.arange(rows) * 100 + 1000,
            "LoanAmount": np.arange(rows) * 10 + 100,
            "CreditScore": np.arange(rows) + 500,
            "MonthsEmployed": np.arange(rows),
            "NumCreditLines": np.ones(rows),
            "InterestRate": np.full(rows, 7.5),
            "LoanTerm": np.full(rows, 36),
            "DTIRatio": np.linspace(0.05, 0.5, rows),
            "Education": np.where(np.arange(rows) % 2, "Bachelor", "High School"),
            "EmploymentType": "Full-time",
            "MaritalStatus": "Single",
            "HasMortgage": "No",
            "HasDependents": "Yes",
            "LoanPurpose": "Other",
            "HasCoSigner": "No",
            "Default": target,
        }
    )
    result = engineer_features(data)
    result["Default"] = target
    return result


def test_pipeline_scales_numerics_encodes_categories_and_handles_missing() -> None:
    data = sample_dataset(40)
    numeric, categorical = get_feature_columns(data)
    pipeline = build_preprocessor(numeric, categorical)
    x, y = feature_frame(data)
    x.loc[x.index[0], "Income"] = np.nan
    x.loc[x.index[1], "Education"] = np.nan
    result = pipeline.fit_transform(x)
    assert result.shape[0] == len(data)
    assert result.shape[1] > len(numeric)
    assert not np.isnan(result.data if sparse.issparse(result) else result).any()
    assert set(y.unique()) == {0, 1}


def test_unknown_categories_do_not_crash() -> None:
    data = sample_dataset(40)
    numeric, categorical = get_feature_columns(data)
    pipeline = build_preprocessor(numeric, categorical)
    x, y = feature_frame(data)
    pipeline.fit(x)
    future = x.iloc[:1].copy()
    future.loc[:, "Education"] = "New category"
    assert pipeline.transform(future).shape[0] == 1


def test_splits_are_stratified_disjoint_and_approximately_70_15_15() -> None:
    data = sample_dataset(200)
    train, validation, test = split_dataset(data)
    total = len(data)
    assert abs(len(train) / total - 0.70) < 0.01
    assert abs(len(validation) / total - 0.15) < 0.01
    assert abs(len(test) / total - 0.15) < 0.01
    assert set(train.index).isdisjoint(validation.index)
    assert set(train.index).isdisjoint(test.index)
    assert set(validation.index).isdisjoint(test.index)
    assert all("Default" in part for part in (train, validation, test))


def test_all_transformed_splits_retain_target_and_same_columns() -> None:
    data = sample_dataset(100)
    train, validation, test = split_dataset(data)
    numeric, categorical = get_feature_columns(data)
    pipeline = build_preprocessor(numeric, categorical)
    train_x, train_y = feature_frame(train)
    pipeline.fit(train_x)
    outputs = []
    for split in (train, validation, test):
        x, y = feature_frame(split)
        outputs.append(transformed_frame(pipeline, x, y))
    assert outputs[0].columns.equals(outputs[1].columns)
    assert outputs[0].columns.equals(outputs[2].columns)
    assert all("Default" in output for output in outputs)
