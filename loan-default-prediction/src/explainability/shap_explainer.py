"""SHAP explanations for the saved loan-default model and Phase 2 features."""

from pathlib import Path
import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.features import engineer_features
from src.data.preprocessing import BASE_CATEGORICAL_FEATURES, BASE_NUMERICAL_FEATURES
from src.models.train import PROJECT_ROOT

RANDOM_STATE = 42
RAW_FEATURES = BASE_NUMERICAL_FEATURES + BASE_CATEGORICAL_FEATURES


class LoanSHAPExplainer:
    """Build global/local SHAP explanations without changing the trained model."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        preprocessor_path: str | Path | None = None,
        background_size: int = 100,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.model_path = Path(model_path or PROJECT_ROOT / "models" / "best_model.joblib")
        self.metadata_path = Path(
            metadata_path or PROJECT_ROOT / "models" / "best_model_metadata.json"
        )
        self.preprocessor_path = Path(
            preprocessor_path or PROJECT_ROOT / "data" / "processed" / "preprocessor.joblib"
        )
        self.background_size = int(background_size)
        self.random_state = int(random_state)
        self.model: Any = None
        self.classifier: Any = None
        self.preprocessor: Any = None
        self.metadata: dict[str, Any] = {}
        self.explainer: Any = None
        self.feature_names: list[str] = []
        self.model_type = ""
        self.explainer_type = ""
        self.output_units = ""
        self.load_model()

    def load_model(self) -> Any:
        """Load the Phase 4 model, metadata, and already-fitted preprocessor."""
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Final model not found: {self.model_path}")
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"Model metadata not found: {self.metadata_path}")
        self.model = joblib.load(self.model_path)
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))

        if isinstance(self.model, Pipeline):
            if len(self.model.steps) < 2:
                raise ValueError("Saved model pipeline must include preprocessing and a classifier.")
            self.classifier = self.model.steps[-1][1]
            self.preprocessor = self.model.named_steps.get("preprocessor")
        else:
            if not self.preprocessor_path.is_file():
                raise FileNotFoundError(f"Phase 2 preprocessor not found: {self.preprocessor_path}")
            self.classifier = self.model
            self.preprocessor = joblib.load(self.preprocessor_path)

        self.feature_names = self.get_feature_names()
        if isinstance(self.classifier, LogisticRegression):
            self.model_type = "Logistic Regression"
            self.explainer_type = "LinearExplainer"
            self.output_units = "log-odds for Default=1"
        elif isinstance(self.classifier, RandomForestClassifier):
            self.model_type = "Random Forest"
            self.explainer_type = "TreeExplainer"
            self.output_units = "raw class-1 model output"
        elif self.classifier.__class__.__name__ == "XGBClassifier":
            self.model_type = "XGBoost"
            self.explainer_type = "TreeExplainer"
            self.output_units = "raw margin (log-odds) for Default=1"
        else:
            raise TypeError(
                f"Unsupported final classifier type: {type(self.classifier).__name__}"
            )
        return self.model

    def get_feature_names(self) -> list[str]:
        """Return names from the fitted Phase 2 transformer, retaining one-hot columns."""
        preprocessor = self.preprocessor
        if preprocessor is None and isinstance(self.model, Pipeline):
            preprocessor = self.model.named_steps.get("preprocessor")
        if preprocessor is None:
            return []
        if hasattr(preprocessor, "get_feature_names_out"):
            names = preprocessor.get_feature_names_out().tolist()
        else:
            names = list(getattr(preprocessor, "feature_names_in_", []))
        return [str(name).removeprefix("num__").removeprefix("cat__") for name in names]

    def _normalize_loan_data(self, loan_data: pd.DataFrame | dict[str, Any] | Any) -> pd.DataFrame:
        if isinstance(loan_data, dict):
            frame = pd.DataFrame([loan_data])
        elif isinstance(loan_data, pd.DataFrame):
            frame = loan_data.copy()
        else:
            frame = pd.DataFrame(loan_data)
        if len(frame) != 1:
            raise ValueError("A local explanation requires exactly one loan row.")
        missing = [feature for feature in RAW_FEATURES if feature not in frame.columns]
        if missing:
            raise ValueError(f"Loan input is missing required raw features: {missing}")
        return frame.loc[:, RAW_FEATURES].reset_index(drop=True)

    def _transform_raw(self, loan_data: pd.DataFrame) -> pd.DataFrame:
        if isinstance(self.model, Pipeline):
            transformed = self.model[:-1].transform(loan_data)
        else:
            transformed = self.preprocessor.transform(engineer_features(loan_data))
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        return pd.DataFrame(transformed, columns=self.feature_names)

    def _predict_default_probability(self, loan_data: pd.DataFrame) -> float:
        if isinstance(self.model, Pipeline):
            return float(self.model.predict_proba(loan_data)[:, 1][0])
        transformed = self._transform_raw(loan_data)
        return float(self.classifier.predict_proba(transformed)[:, 1][0])

    def _default_background(self) -> pd.DataFrame:
        from scripts.optimize_model import load_original_split

        validation = load_original_split("validation")
        predictors = validation.drop(columns="Default", errors="ignore")
        size = min(self.background_size, len(predictors))
        return predictors.sample(n=size, random_state=self.random_state).loc[:, RAW_FEATURES]

    @staticmethod
    def _positive_class_values(values: Any) -> np.ndarray:
        if isinstance(values, list):
            if len(values) < 2:
                raise ValueError("SHAP output does not contain Default=1 values.")
            values = values[1]
        array = np.asarray(values)
        if array.ndim == 3:
            if array.shape[-1] == 2:
                array = array[..., 1]
            elif array.shape[0] == 2:
                array = array[1]
            else:
                raise ValueError(f"Cannot identify positive-class SHAP axis in shape {array.shape}.")
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise ValueError(f"Expected two-dimensional SHAP values, got shape {array.shape}.")
        return array.astype(float)

    @staticmethod
    def _positive_base_values(base_values: Any, rows: int) -> np.ndarray:
        base = np.asarray(base_values)
        if base.ndim == 0:
            return np.full(rows, float(base))
        if base.ndim == 1 and base.size == 2:
            return np.full(rows, float(base[1]))
        if base.ndim == 2 and base.shape[-1] == 2:
            base = base[:, 1]
        return np.broadcast_to(base.astype(float), (rows,)).copy()

    def create_explainer(self, background_data: pd.DataFrame | None = None) -> Any:
        """Create a model-appropriate explainer from a small validation background."""
        import shap

        background_raw = self._default_background() if background_data is None else background_data
        background_raw = background_raw.drop(columns="Default", errors="ignore")
        background_raw = background_raw.loc[:, RAW_FEATURES]
        background_transformed = self._transform_raw(background_raw)
        if self.model_type == "Logistic Regression":
            self.explainer = shap.LinearExplainer(self.classifier, background_transformed)
        else:
            self.explainer = shap.TreeExplainer(
                self.classifier,
                data=background_transformed,
                feature_perturbation="interventional",
                model_output="raw",
            )
        return self.explainer

    def calculate_shap_values(
        self,
        loan_data: pd.DataFrame,
        background_data: pd.DataFrame | None = None,
    ) -> Any:
        """Return a SHAP Explanation for Default=1 on original raw input rows."""
        import shap

        raw = loan_data.drop(columns="Default", errors="ignore")
        raw = raw.loc[:, RAW_FEATURES].reset_index(drop=True)
        if self.explainer is None:
            self.create_explainer(background_data)
        transformed = self._transform_raw(raw)
        result = self.explainer.shap_values(transformed)
        values = self._positive_class_values(result)
        base_values = self._positive_base_values(self.explainer.expected_value, len(transformed))
        return shap.Explanation(
            values=values,
            base_values=base_values,
            data=transformed.to_numpy(),
            feature_names=self.feature_names,
        )

    def get_global_importance(self, shap_values: Any) -> pd.DataFrame:
        """Rank transformed features by mean absolute SHAP contribution."""
        values = self._positive_class_values(
            shap_values.values if hasattr(shap_values, "values") else shap_values
        )
        if values.shape[1] != len(self.feature_names):
            raise ValueError("SHAP feature count does not match fitted preprocessing output.")
        importance = pd.DataFrame(
            {
                "feature": self.feature_names,
                "mean_abs_shap": np.abs(values).mean(axis=0),
            }
        )
        importance = importance.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        importance["rank"] = np.arange(1, len(importance) + 1)
        return importance[["feature", "mean_abs_shap", "rank"]]

    def _feature_description(self, feature: str, raw_loan: pd.Series) -> str:
        if feature in raw_loan.index:
            return f"{feature} (input={raw_loan[feature]})"
        for category in BASE_CATEGORICAL_FEATURES:
            prefix = f"{category}_"
            if feature.startswith(prefix):
                return (
                    f"{category}={feature[len(prefix):]} one-hot indicator "
                    f"(input category={raw_loan[category]})"
                )
        return feature

    def explain_single_prediction(self, loan_data: pd.DataFrame | dict[str, Any]) -> dict[str, Any]:
        """Return JSON-safe probability, threshold decision, and SHAP factors."""
        raw = self._normalize_loan_data(loan_data)
        explanation = self.calculate_shap_values(raw)
        shap_row = np.asarray(explanation.values[0], dtype=float)
        probability = self._predict_default_probability(raw)
        threshold = float(self.metadata.get("classification_threshold", self.metadata.get("threshold", 0.5)))
        predicted_class = int(probability >= threshold)
        raw_loan = raw.iloc[0]

        ranked_indices = np.argsort(np.abs(shap_row))[::-1]

        def factor(index: int) -> dict[str, Any]:
            value = float(shap_row[index])
            direction = "increases_risk" if value >= 0 else "decreases_risk"
            description = self._feature_description(self.feature_names[index], raw_loan)
            risk_text = "higher" if value >= 0 else "lower"
            return {
                "feature": self.feature_names[index],
                "description": description,
                "shap_value": value,
                "direction": direction,
                "explanation": (
                    f"{description} contributed to the model's {risk_text} predicted "
                    f"default risk ({value:+.4f} {self.output_units})."
                ),
            }

        positive = [factor(index) for index in ranked_indices if shap_row[index] > 0][:5]
        negative = [factor(index) for index in ranked_indices if shap_row[index] < 0][:5]
        top_factors = [factor(index) for index in ranked_indices[:10]]
        base_value = float(np.asarray(explanation.base_values).reshape(-1)[0])
        return {
            "default_probability": probability,
            "predicted_class": predicted_class,
            "threshold": threshold,
            "base_value": base_value,
            "model_output_units": self.output_units,
            "top_factors": top_factors,
            "top_positive_factors": positive,
            "top_negative_factors": negative,
        }

    def explain_prediction(self, loan_data: pd.DataFrame | dict[str, Any]) -> dict[str, Any]:
        """Return a compact JSON-serializable explanation for future API use."""
        explanation = self.explain_single_prediction(loan_data)
        return {
            "default_probability": float(explanation["default_probability"]),
            "predicted_class": int(explanation["predicted_class"]),
            "threshold": float(explanation["threshold"]),
            "model_output_units": str(explanation["model_output_units"]),
            "top_factors": [
                {
                    "feature": str(factor["feature"]),
                    "description": str(factor["description"]),
                    "shap_value": float(factor["shap_value"]),
                    "direction": str(factor["direction"]),
                }
                for factor in explanation["top_factors"]
            ],
        }