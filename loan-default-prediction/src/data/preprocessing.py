"""Reusable preprocessing and leakage-safe split helpers."""

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.features import ENGINEERED_FEATURES
from src.data.loader import TARGET_COLUMN

RANDOM_STATE = 42
BASE_NUMERICAL_FEATURES = [
    "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
    "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio",
]
BASE_CATEGORICAL_FEATURES = [
    "Education", "EmploymentType", "MaritalStatus", "HasMortgage",
    "HasDependents", "LoanPurpose", "HasCoSigner",
]


def get_feature_columns(dataframe: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numeric and categorical predictor names present in a frame."""
    numeric = BASE_NUMERICAL_FEATURES + [
        name for name in ("IncomeLoanRatio", "LoanIncomeRatio") if name in dataframe
    ]
    categorical = BASE_CATEGORICAL_FEATURES + [
        name for name in ("DTI_Category", "EmploymentCategory", "CreditRiskCategory")
        if name in dataframe
    ]
    absent = [name for name in numeric + categorical if name not in dataframe]
    if absent:
        raise ValueError(f"Expected model feature columns are missing: {absent}")
    return numeric, categorical


def build_preprocessor(numerical_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Create median/scale numeric and impute/one-hot categorical transforms."""
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        [
            ("numerical", numeric_pipeline, numerical_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def split_dataset(
    dataframe: pd.DataFrame, random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split feature/target rows 70/15/15, stratified by Default."""
    if TARGET_COLUMN not in dataframe:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is required for splitting.")
    train, remainder = train_test_split(
        dataframe,
        test_size=0.30,
        stratify=dataframe[TARGET_COLUMN],
        random_state=random_state,
    )
    validation, test = train_test_split(
        remainder,
        test_size=0.50,
        stratify=remainder[TARGET_COLUMN],
        random_state=random_state,
    )
    return train.copy(), validation.copy(), test.copy()


def feature_frame(split: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate predictors and the untouched binary target."""
    return split.drop(columns=[TARGET_COLUMN]), split[TARGET_COLUMN].astype("int64")


def transformed_frame(preprocessor: Any, features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """Transform predictors and append the unscaled target."""
    matrix = preprocessor.transform(features)
    try:
        names = preprocessor.get_feature_names_out()
    except AttributeError:
        names = [f"feature_{index}" for index in range(matrix.shape[1])]
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    result = pd.DataFrame(matrix, columns=names, index=features.index)
    result[TARGET_COLUMN] = target.to_numpy()
    return result
