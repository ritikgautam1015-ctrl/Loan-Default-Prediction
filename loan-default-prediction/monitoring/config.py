"""Environment-backed paths and thresholds for local monitoring jobs."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _path_from_env(name: str, default: str) -> Path:
    path = Path(os.getenv(name, default)).expanduser()
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


REFERENCE_DATA_PATH = _path_from_env(
    "REFERENCE_DATA_PATH", "data/processed/train.csv"
)
CURRENT_DATA_PATH = _path_from_env(
    "CURRENT_DATA_PATH", "data/processed/validation.csv"
)
MODEL_PATH = _path_from_env("MODEL_PATH", "models/best_model.joblib")
MODEL_METADATA_PATH = _path_from_env(
    "MODEL_METADATA_PATH", "models/best_model_metadata.json"
)
PREPROCESSOR_PATH = _path_from_env(
    "PREPROCESSOR_PATH", "data/processed/preprocessor.joblib"
)
FEATURE_METADATA_PATH = _path_from_env(
    "FEATURE_METADATA_PATH", "data/processed/feature_metadata.json"
)
MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME", "loan-default-prediction"
)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
DRIFT_SHARE_THRESHOLD = float(os.getenv("DRIFT_SHARE_THRESHOLD", "0.30"))
DRIFT_TEST_THRESHOLD = float(os.getenv("DRIFT_TEST_THRESHOLD", "0.05"))

REPORT_MONITORING_PATH = PROJECT_ROOT / "reports" / "monitoring"
REPORT_MLFLOW_PATH = PROJECT_ROOT / "reports" / "mlflow"