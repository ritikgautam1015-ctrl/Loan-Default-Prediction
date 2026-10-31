"""Validate local data, model artifacts, reports, and required imports."""

import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sqlalchemy.engine import make_url

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import EXPECTED_COLUMNS  # noqa: E402
from src.explainability.shap_explainer import LoanSHAPExplainer  # noqa: E402


def check_project() -> tuple[list[tuple[str, bool, str]], dict[str, object]]:
    checks: list[tuple[str, bool, str]] = []
    results: dict[str, object] = {}

    dataset_path = PROJECT_ROOT / "data" / "raw" / "Loan_default.csv"
    try:
        sample = pd.read_csv(dataset_path, nrows=5)
        dataset = pd.read_csv(dataset_path)
        valid = (
            list(sample.columns) == EXPECTED_COLUMNS
            and "Default" in dataset
            and set(dataset["Default"].dropna().unique()).issubset({0, 1})
            and not dataset["Default"].isna().any()
        )
        numeric = [
            "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
            "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio",
        ]
        valid = valid and all(pd.api.types.is_numeric_dtype(dataset[column]) for column in numeric)
        checks.append(("Dataset", valid, f"{len(dataset):,} rows x {len(dataset.columns)} columns"))
        results["dataset"] = {"rows": len(dataset), "columns": len(dataset.columns)}
    except Exception as error:
        checks.append(("Dataset", False, str(error)))

    artifact_paths = {
        "Final model": PROJECT_ROOT / "models" / "best_model.joblib",
        "Model metadata": PROJECT_ROOT / "models" / "best_model_metadata.json",
        "Preprocessor": PROJECT_ROOT / "data" / "processed" / "preprocessor.joblib",
        "SHAP importance": PROJECT_ROOT / "reports" / "shap_feature_importance.csv",
        "SHAP summary bar": PROJECT_ROOT / "reports" / "figures" / "shap_summary_bar.png",
        "SHAP beeswarm": PROJECT_ROOT / "reports" / "figures" / "shap_summary_beeswarm.png",
        "SHAP local plot": PROJECT_ROOT / "reports" / "figures" / "shap_local_example.png",
        "Model comparison report": PROJECT_ROOT / "reports" / "model_comparison.csv",
        "Drift summary": PROJECT_ROOT / "reports" / "monitoring" / "drift_summary.json",
        "MLflow run summary": PROJECT_ROOT / "reports" / "mlflow" / "mlflow_run_summary.json",
    }
    for name, path in artifact_paths.items():
        exists = path.is_file() and path.stat().st_size > 0
        checks.append((name, exists, "" if exists else f"Missing: {path.relative_to(PROJECT_ROOT)}"))

    try:
        model = joblib.load(artifact_paths["Final model"])
        metadata = json.loads(artifact_paths["Model metadata"].read_text(encoding="utf-8"))
        preprocessor = joblib.load(artifact_paths["Preprocessor"])
        if not hasattr(model, "predict_proba") or not metadata.get("model_version"):
            raise ValueError("Model probability method or model version is missing.")
        results["model"] = metadata.get("model_name")
        results["model_version"] = metadata.get("model_version")
        results["threshold"] = metadata.get("classification_threshold")
        results["preprocessor_features"] = len(preprocessor.get_feature_names_out())
        checks.append(("Prediction model", True, str(metadata.get("model_name"))))
        checks.append(("Preprocessing artifact", True, f"{results['preprocessor_features']} transformed features"))
    except Exception as error:
        checks.append(("Prediction model", False, str(error)))
        checks.append(("Preprocessing artifact", False, str(error)))

    try:
        loan = pd.read_csv(dataset_path, nrows=1).drop(columns=["LoanID", "Default"])
        probability = float(model.predict_proba(loan)[0, 1])
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Model returned an invalid probability.")
        checks.append(("Sample inference", True, f"Default probability {probability:.4f}"))
    except Exception as error:
        checks.append(("Sample inference", False, str(error)))

    try:
        loan = pd.read_csv(dataset_path, nrows=1).drop(columns=["LoanID", "Default"]).iloc[0].to_dict()
        explanation = LoanSHAPExplainer().explain_prediction(loan)
        json.dumps(explanation)
        expected = {"default_probability", "predicted_class", "threshold", "top_factors"}
        if not expected.issubset(explanation):
            raise ValueError("SHAP response is missing required fields.")
        checks.append(("SHAP explanation", True, f"{len(explanation['top_factors'])} factors"))
    except Exception as error:
        checks.append(("SHAP explanation", False, str(error)))

    modules = {
        "FastAPI": "fastapi",
        "Streamlit": "streamlit",
        "Requests": "requests",
        "MLflow": "mlflow",
        "Evidently": "evidently",
    }
    for name, module in modules.items():
        available = importlib.util.find_spec(module) is not None
        checks.append((name, available, "" if available else f"Missing Python package: {module}"))
    api_modules = all(
        importlib.util.find_spec(module) is not None
        for module in ("app.main", "app.services.prediction_service")
    )
    checks.append(("FastAPI modules", api_modules, "" if api_modules else "API package import failed"))

    try:
        from src.config import DATABASE_URL

        parsed = make_url(DATABASE_URL)
        valid_database_config = parsed.drivername.startswith("postgresql")
        checks.append(
            (
                "PostgreSQL configuration",
                valid_database_config,
                "Valid URL format; live database is not required for inference",
            )
        )
    except Exception as error:
        checks.append(("PostgreSQL configuration", False, str(error)))

    return checks, results


def main() -> int:
    print("Loan Default Prediction - Project Check\n")
    checks, _ = check_project()
    for name, passed, detail in checks:
        label = "OK" if passed else "FAIL"
        suffix = f" - {detail}" if detail else ""
        print(f"[{label}] {name}{suffix}")
    if all(passed for _, passed, _ in checks):
        print("\nProject check completed successfully.")
        return 0
    print("\nProject check found missing or invalid components.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())