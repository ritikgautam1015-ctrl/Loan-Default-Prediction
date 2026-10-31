"""In-memory model inference and Phase 5 SHAP response service."""

import logging
from typing import Any

import pandas as pd

from src.explainability.shap_explainer import LoanSHAPExplainer

logger = logging.getLogger(__name__)

RISK_BAND_MEDIUM_THRESHOLD = 0.30
RISK_BAND_HIGH_THRESHOLD = 0.60
MAX_EXPLANATION_FACTORS = 5


class PredictionService:
    """Load the saved Phase 4 model and Phase 5 explainer once for the API."""

    def __init__(self) -> None:
        self.shap_explainer = LoanSHAPExplainer()
        self.model = self.shap_explainer.model
        self.metadata = self.shap_explainer.metadata
        self.model_version = str(self.metadata["model_version"])
        self.decision_threshold = float(self.metadata["classification_threshold"])
        if not 0 <= self.decision_threshold <= 1:
            raise ValueError("Saved model threshold must be between 0 and 1.")
        if not hasattr(self.model, "predict_proba"):
            raise TypeError("Saved final model must expose predict_proba().")

        self.shap_explainer.create_explainer()
        logger.info(
            "Model loaded successfully. Model=%s Version=%s",
            self.metadata["model_name"],
            self.model_version,
        )

    @staticmethod
    def _risk_band(probability: float) -> str:
        if probability < RISK_BAND_MEDIUM_THRESHOLD:
            return "Low"
        if probability < RISK_BAND_HIGH_THRESHOLD:
            return "Medium"
        return "High"

    def _predict(self, loan: dict[str, Any]) -> dict[str, Any]:
        features = pd.DataFrame([loan])
        probability = float(self.model.predict_proba(features)[0, 1])
        predicted_default = int(probability >= self.decision_threshold)
        return {
            "default_probability": round(probability, 4),
            "predicted_default": predicted_default,
            "decision_threshold": self.decision_threshold,
            "risk_band": self._risk_band(probability),
            "model_version": self.model_version,
        }

    def predict(self, loan: dict[str, Any]) -> dict[str, Any]:
        logger.info("Prediction request received.")
        response = self._predict(loan)
        logger.info("Prediction completed successfully.")
        return response

    def explain(self, loan: dict[str, Any]) -> dict[str, Any]:
        logger.info("Explanation request received.")
        prediction = self._predict(loan)
        explanation = self.shap_explainer.explain_prediction(loan)
        factors = sorted(
            explanation["top_factors"],
            key=lambda factor: abs(float(factor["shap_value"])),
            reverse=True,
        )[:MAX_EXPLANATION_FACTORS]
        response_factors = [
            {
                "feature": str(factor["feature"]),
                "shap_value": float(factor["shap_value"]),
                "direction": str(factor["direction"]),
            }
            for factor in factors
        ]
        feature_names = list(dict.fromkeys(factor["feature"] for factor in response_factors))
        if feature_names:
            summary = (
                "The model's predicted default risk is mainly influenced by "
                + ", ".join(feature_names)
                + "."
            )
        else:
            summary = "No dominant SHAP factors were available for this prediction."
        logger.info("Explanation completed successfully.")
        return {
            **prediction,
            "summary": summary,
            "top_factors": response_factors,
        }