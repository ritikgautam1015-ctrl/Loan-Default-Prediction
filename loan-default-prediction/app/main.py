"""FastAPI application for real-time loan-default predictions."""

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.dependencies import get_prediction_service
from app.schemas import ExplanationResponse, LoanApplication, PredictionResponse
from app.services.prediction_service import PredictionService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LOCAL_DEVELOPMENT_ORIGINS = ["http://localhost", "http://127.0.0.1"]


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load model, metadata, and SHAP background once; fail startup on errors."""
    application.state.prediction_service = PredictionService()
    logger.info(
        "Loan Default Prediction API started. Model version=%s",
        application.state.prediction_service.model_version,
    )
    yield
    application.state.prediction_service = None
    logger.info("Loan Default Prediction API stopped.")


app = FastAPI(
    title="Loan Default Prediction API",
    description=(
        "Predicts default probability with the Phase 4 model and provides "
        "optional Phase 5 SHAP explanations. Unknown categories are supported."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=LOCAL_DEVELOPMENT_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

ServiceDependency = Annotated[PredictionService, Depends(get_prediction_service)]


@app.get("/", tags=["Service"], summary="API service information")
def root() -> dict[str, str]:
    """Return API name, API version, and running status."""
    return {
        "service": "Loan Default Prediction API",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health", tags=["Service"], summary="Model readiness")
def health(service: ServiceDependency) -> dict[str, str | bool]:
    """Report readiness and the loaded model version."""
    return {
        "status": "healthy",
        "model_loaded": service.model is not None,
        "model_version": service.model_version,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict loan default risk",
    description=(
        "Returns Default=1 probability, the Phase 4 threshold-based class, and "
        "a probability risk band. Risk bands are independent of the decision threshold."
    ),
    responses={500: {"description": "Prediction service failed."}},
)
def predict(application: LoanApplication, service: ServiceDependency) -> PredictionResponse:
    """Predict one loan application using the saved model pipeline."""
    try:
        return PredictionResponse(**service.predict(application.model_dump()))
    except Exception as error:
        logger.exception("Prediction failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction service failed.",
        ) from error


@app.post(
    "/explain",
    response_model=ExplanationResponse,
    tags=["Explainability"],
    summary="Explain a loan default prediction",
    description=(
        "Adds up to five signed Phase 5 SHAP contributions. Contributions describe "
        "model behavior and are not causal effects."
    ),
    responses={500: {"description": "Explanation service failed."}},
)
def explain(application: LoanApplication, service: ServiceDependency) -> ExplanationResponse:
    """Return prediction and the top signed SHAP factors."""
    try:
        return ExplanationResponse(**service.explain(application.model_dump()))
    except Exception as error:
        logger.exception("Explanation failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Explanation service failed.",
        ) from error