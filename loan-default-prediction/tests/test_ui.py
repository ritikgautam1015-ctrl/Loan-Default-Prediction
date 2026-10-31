"""Tests for safe Streamlit-to-FastAPI request handling."""

from types import SimpleNamespace

import pytest
import requests

from ui.app import api_post, category_options


class FakeResponse:
    def __init__(self, payload, status_code: int = 200):
        self.payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 300

    def json(self):
        return self.payload


def test_prediction_response_is_validated_and_returned(monkeypatch) -> None:
    payload = {
        "default_probability": 0.12,
        "predicted_default": 0,
        "decision_threshold": 0.15,
        "risk_band": "Low",
        "model_version": "1.0.0",
    }
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: FakeResponse(payload))
    assert api_post("http://example", "/predict", {}) == payload


def test_malformed_success_response_is_friendly_error(monkeypatch) -> None:
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: FakeResponse({"ok": True}))
    with pytest.raises(RuntimeError, match="unexpected response format"):
        api_post("http://example", "/predict", {})


def test_unavailable_api_is_friendly_error(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise requests.ConnectionError("internal host detail")

    monkeypatch.setattr(requests, "post", fail)
    with pytest.raises(RuntimeError, match="make sure the FastAPI backend is running"):
        api_post("http://example", "/predict", {})


def test_explanation_response_rejects_invalid_factor(monkeypatch) -> None:
    payload = {
        "default_probability": 0.12,
        "predicted_default": 0,
        "decision_threshold": 0.15,
        "risk_band": "Low",
        "model_version": "1.0.0",
        "summary": "Model explanation.",
        "top_factors": [{"feature": "Age", "shap_value": "bad", "direction": "increases_risk"}],
    }
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: FakeResponse(payload))
    with pytest.raises(RuntimeError, match="invalid response format"):
        api_post("http://example", "/explain", {})


def test_category_options_use_saved_phase2_categories() -> None:
    metadata = {
        "quality_report": {
            "categorical_value_counts": {"Education": {"PhD": 1, "Bachelor's": 3}}
        }
    }
    assert category_options(metadata, "Education") == ["Bachelor's", "PhD"]