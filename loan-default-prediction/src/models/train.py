"""Processed-data loading and baseline model definitions."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier

from src.data.features import engineer_features

RANDOM_STATE = 42
TARGET_COLUMN = "Default"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"


def load_processed_split(split: str) -> pd.DataFrame:
    """Load one named Phase 2 split without opening other split files."""
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown split {split!r}.")
    path = PROCESSED_PATH / f"{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Processed {split} data not found: {path}")
    dataframe = pd.read_csv(path)
    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(f"Processed {split} data is missing {TARGET_COLUMN!r}.")
    if not set(dataframe[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError(f"Processed {split} target must contain only 0 and 1.")
    return dataframe


def separate_features_target(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate predictors and ensure the target cannot enter a model matrix."""
    if TARGET_COLUMN not in dataframe:
        raise ValueError(f"Target column {TARGET_COLUMN!r} is required.")
    features = dataframe.drop(columns=TARGET_COLUMN)
    if TARGET_COLUMN in features.columns:
        raise AssertionError("Target leakage: Default is present in features.")
    return features, dataframe[TARGET_COLUMN].astype("int64")


def build_baseline_models() -> dict[str, object]:
    """Construct the requested fixed-seed baseline classifiers."""
    return {
        "Logistic Regression (unweighted)": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Logistic Regression (balanced)": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1,
        ),
    }


def load_preprocessor():
    """Load the fitted Phase 2 transformer without fitting it again."""
    path = PROCESSED_PATH / "preprocessor.joblib"
    if not path.is_file():
        raise FileNotFoundError(f"Phase 2 preprocessor not found: {path}")
    return joblib.load(path)


def build_serving_pipeline(preprocessor, classifier) -> Pipeline:
    """Bundle Phase 2 feature engineering, fitted preprocessing, and classifier.

    Input is the original predictor frame (LoanID and Default are optional and
    ignored by the transformer); the target classifier was fitted on the exact
    Phase 2 processed representation.
    """
    return Pipeline(
        [
            ("feature_engineering", FunctionTransformer(engineer_features, validate=False)),
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )