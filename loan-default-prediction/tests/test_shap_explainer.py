"""Small-sample tests for reusable SHAP explanations."""

import json
import joblib

import pandas as pd
import pytest

from scripts.optimize_model import load_original_split
from src.explainability.shap_explainer import LoanSHAPExplainer


@pytest.fixture(scope="module")
def explained_sample():
    validation = load_original_split("validation").drop(columns="Default")
    sample = validation.sample(n=12, random_state=42).reset_index(drop=True)
    explainer = LoanSHAPExplainer(background_size=10)
    background = sample.iloc[:10]
    explainer.create_explainer(background)
    shap_values = explainer.calculate_shap_values(sample, background_data=background)
    return explainer, sample, shap_values


def test_saved_model_loads_and_selects_a_supported_explainer(explained_sample) -> None:
    explainer, _, _ = explained_sample
    assert explainer.model is not None
    assert explainer.model_type in {"Logistic Regression", "Random Forest", "XGBoost"}
    assert explainer.explainer_type in {"LinearExplainer", "TreeExplainer"}


def test_feature_names_match_fitted_preprocessor(explained_sample) -> None:
    explainer, _, _ = explained_sample
    assert len(explainer.get_feature_names()) == 43
    assert "CreditScore" in explainer.feature_names
    assert any(name.startswith("Education_") for name in explainer.feature_names)


def test_shap_dimensions_and_global_importance(explained_sample) -> None:
    explainer, _, shap_values = explained_sample
    assert shap_values.values.shape == (12, len(explainer.feature_names))
    importance = explainer.get_global_importance(shap_values)
    assert importance.columns.tolist() == ["feature", "mean_abs_shap", "rank"]
    assert importance["rank"].tolist() == list(range(1, len(importance) + 1))
    assert importance["mean_abs_shap"].is_monotonic_decreasing


def test_local_explanation_is_json_serializable(explained_sample) -> None:
    explainer, sample, _ = explained_sample
    result = explainer.explain_prediction(sample.iloc[[0]])
    assert {"default_probability", "predicted_class", "threshold", "top_factors"}.issubset(result)
    assert result["threshold"] == pytest.approx(0.15)
    assert 0 <= result["default_probability"] <= 1
    assert result["predicted_class"] in {0, 1}
    assert result["top_factors"]
    assert json.loads(json.dumps(result)) == result


def test_single_loan_dictionary_is_supported(explained_sample) -> None:
    explainer, sample, _ = explained_sample
    result = explainer.explain_single_prediction(sample.iloc[0].to_dict())
    assert isinstance(result["top_positive_factors"], list)
    assert isinstance(result["top_negative_factors"], list)
    assert all("contributed to the model's" in item["explanation"] for item in result["top_factors"])


def test_separate_classifier_reuses_saved_preprocessor(explained_sample, tmp_path) -> None:
    pipeline_explainer, sample, _ = explained_sample
    classifier_path = tmp_path / "classifier.joblib"
    metadata_path = tmp_path / "metadata.json"
    joblib.dump(pipeline_explainer.classifier, classifier_path)
    metadata_path.write_text(json.dumps({"classification_threshold": 0.15}), encoding="utf-8")
    explainer = LoanSHAPExplainer(
        model_path=classifier_path,
        metadata_path=metadata_path,
        preprocessor_path=pipeline_explainer.preprocessor_path,
        background_size=10,
    )
    background = sample.iloc[:10]
    explainer.create_explainer(background)
    result = explainer.explain_prediction(sample.iloc[[0]])
    assert 0 <= result["default_probability"] <= 1
    assert result["threshold"] == pytest.approx(0.15)
