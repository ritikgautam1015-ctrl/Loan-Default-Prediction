"""Validated request and response contracts for the prediction API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

LOAN_EXAMPLE = {
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


class LoanApplication(BaseModel):
    """Original borrower predictors; identifiers and target are forbidden."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [LOAN_EXAMPLE]},
    )

    Age: float = Field(gt=0, description="Borrower age in years.")
    Income: float = Field(ge=0, description="Borrower income.")
    LoanAmount: float = Field(ge=0, description="Requested loan amount.")
    CreditScore: float = Field(ge=0, description="Borrower credit score.")
    MonthsEmployed: float = Field(ge=0, description="Employment duration in months.")
    NumCreditLines: float = Field(ge=0, description="Number of existing credit lines.")
    InterestRate: float = Field(ge=0, description="Proposed loan interest rate.")
    LoanTerm: float = Field(gt=0, description="Loan term in months.")
    DTIRatio: float = Field(ge=0, description="Debt-to-income ratio.")
    Education: str = Field(min_length=1)
    EmploymentType: str = Field(min_length=1)
    MaritalStatus: str = Field(min_length=1)
    HasMortgage: str = Field(min_length=1)
    HasDependents: str = Field(min_length=1)
    LoanPurpose: str = Field(min_length=1)
    HasCoSigner: str = Field(min_length=1)


class PredictionResponse(BaseModel):
    """Probability and threshold-based classification for one application."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "default_probability": 0.72,
                    "predicted_default": 1,
                    "decision_threshold": 0.15,
                    "risk_band": "High",
                    "model_version": "1.0.0",
                }
            ]
        }
    )

    default_probability: float = Field(ge=0, le=1)
    predicted_default: Literal[0, 1]
    decision_threshold: float = Field(ge=0, le=1)
    risk_band: Literal["Low", "Medium", "High"]
    model_version: str


class TopFactor(BaseModel):
    """One signed SHAP contribution for an API response."""

    feature: str
    shap_value: float
    direction: Literal["increases_risk", "decreases_risk"]


class ExplanationResponse(PredictionResponse):
    """Prediction result plus a short, non-causal model explanation."""

    summary: str
    top_factors: list[TopFactor] = Field(max_length=5)