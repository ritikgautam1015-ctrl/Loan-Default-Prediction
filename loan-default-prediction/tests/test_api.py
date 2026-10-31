"""API contract tests using the saved model and Phase 5 SHAP artifacts."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

VALID_LOAN = {
    "Age": 35,
    "Income": 60000,
    "LoanAmount": 25000,
    "CreditScore": 680,
    "MonthsEmployed": 60,
    "NumCreditLines": 4,
    "InterestRate": 8.5,
    "LoanTerm": 36,
    "DTIRatio": 0.32,
    "Education": "Bachelor's",
    "EmploymentType": "Full-time",
    "MaritalStatus": "Married",
    "HasMortgage": "Yes",
    "HasDependents": "No",
    "LoanPurpose": "Home",
    "HasCoSigner": "No",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root_returns_service_information(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "service": "Loan Default Prediction API",
        "version": "1.0.0",
        "status": "running",
    }


def test_health_reports_loaded_metadata_version(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True
    assert response.json()["model_version"] == "1.0.0"


def test_predict_returns_valid_response(client: TestClient) -> None:
    response = client.post("/predict", json=VALID_LOAN)
    assert response.status_code == 200
    payload = response.json()
    assert 0 <= payload["default_probability"] <= 1
    assert payload["predicted_default"] in {0, 1}
    assert payload["decision_threshold"] == pytest.approx(0.15)
    assert payload["risk_band"] in {"Low", "Medium", "High"}
    assert payload["model_version"] == "1.0.0"


def test_invalid_numerical_input_returns_422(client: TestClient) -> None:
    invalid = {**VALID_LOAN, "Age": 0}
    assert client.post("/predict", json=invalid).status_code == 422


def test_target_and_identifier_cannot_be_supplied(client: TestClient) -> None:
    assert client.post("/predict", json={**VALID_LOAN, "Default": 0}).status_code == 422
    assert client.post("/predict", json={**VALID_LOAN, "LoanID": "loan-1"}).status_code == 422


def test_unknown_categories_are_supported(client: TestClient) -> None:
    response = client.post("/predict", json={**VALID_LOAN, "Education": "Future category"})
    assert response.status_code == 200


def test_explain_returns_up_to_five_signed_factors(client: TestClient) -> None:
    response = client.post("/explain", json=VALID_LOAN)
    assert response.status_code == 200
    payload = response.json()
    assert 0 <= payload["default_probability"] <= 1
    assert payload["predicted_default"] in {0, 1}
    assert payload["decision_threshold"] == pytest.approx(0.15)
    assert payload["model_version"] == "1.0.0"
    assert len(payload["top_factors"]) <= 5
    assert payload["summary"]
    for factor in payload["top_factors"]:
        assert isinstance(factor["shap_value"], float)
        assert factor["direction"] in {"increases_risk", "decreases_risk"}


def test_prediction_and_explanation_share_decision_fields(client: TestClient) -> None:
    prediction = client.post("/predict", json=VALID_LOAN).json()
    explanation = client.post("/explain", json=VALID_LOAN).json()
    for key in ("default_probability", "predicted_default", "decision_threshold", "risk_band", "model_version"):
        assert explanation[key] == prediction[key]