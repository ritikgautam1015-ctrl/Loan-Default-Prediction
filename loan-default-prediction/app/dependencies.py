"""FastAPI dependency providers."""

from fastapi import HTTPException, Request, status

from app.services.prediction_service import PredictionService


def get_prediction_service(request: Request) -> PredictionService:
    """Return the service initialized by the application lifespan."""
    service = getattr(request.app.state, "prediction_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service is unavailable.",
        )
    return service