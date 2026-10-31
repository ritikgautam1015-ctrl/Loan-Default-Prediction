"""Focused tests for training helpers and model evaluation."""

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from src.data.features import engineer_features
from src.data.preprocessing import build_preprocessor, get_feature_columns
from src.models.evaluate import evaluate_model
from src.models.train import (
    build_serving_pipeline,
    load_processed_split,
    separate_features_target,
)


def test_processed_training_data_loads_with_binary_target() -> None:
    training = load_processed_split("train")
    features, target = separate_features_target(training)
    assert not features.empty
    assert "Default" not in features.columns
    assert set(target.unique()) == {0, 1}


def test_small_model_trains_and_reports_probability_metrics(tmp_path) -> None:
    features, target = make_classification(
        n_samples=160,
        n_features=10,
        n_informative=5,
        weights=[0.8, 0.2],
        random_state=42,
    )
    model = LogisticRegression(max_iter=1000, random_state=42).fit(features, target)
    probabilities = model.predict_proba(features)[:, 1]
    metrics = evaluate_model(model, features, target)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert 0 <= metrics["ROC_AUC"] <= 1
    assert 0 <= metrics["PR_AUC"] <= 1
    assert sum(sum(row) for row in metrics["Confusion_Matrix"]) == len(target)

    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    assert loaded.predict_proba(features).shape == (len(target), 2)


def test_serving_pipeline_reuses_phase2_preprocessing(tmp_path) -> None:
    rows = 40
    raw = pd.DataFrame(
        {
            "Age": np.arange(rows) + 20,
            "Income": np.arange(rows) * 500 + 1000,
            "LoanAmount": np.arange(rows) * 100 + 500,
            "CreditScore": np.arange(rows) + 500,
            "MonthsEmployed": np.arange(rows) * 2,
            "NumCreditLines": np.arange(rows) % 5 + 1,
            "InterestRate": np.arange(rows) / 10 + 2,
            "LoanTerm": np.full(rows, 36),
            "DTIRatio": np.arange(rows) / 100,
            "Education": np.where(np.arange(rows) % 2, "Bachelor", "High School"),
            "EmploymentType": np.where(np.arange(rows) % 2, "Full-time", "Part-time"),
            "MaritalStatus": np.where(np.arange(rows) % 2, "Single", "Married"),
            "HasMortgage": np.where(np.arange(rows) % 2, "Yes", "No"),
            "HasDependents": np.where(np.arange(rows) % 2, "Yes", "No"),
            "LoanPurpose": np.where(np.arange(rows) % 2, "Home", "Auto"),
            "HasCoSigner": np.where(np.arange(rows) % 2, "Yes", "No"),
        }
    )
    target = pd.Series(np.tile([0, 1], rows // 2), name="Default")
    engineered = engineer_features(raw)
    numerical, categorical = get_feature_columns(engineered)
    preprocessor = build_preprocessor(numerical, categorical).fit(engineered)
    processed = preprocessor.transform(engineered)
    classifier = LogisticRegression(max_iter=1000, random_state=42).fit(processed, target)
    pipeline = build_serving_pipeline(preprocessor, classifier)
    path = tmp_path / "best_model.joblib"
    joblib.dump(pipeline, path)
    loaded = joblib.load(path)
    probabilities = loaded.predict_proba(raw)
    assert probabilities.shape == (rows, 2)
    assert np.all((probabilities >= 0) & (probabilities <= 1))