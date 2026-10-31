# Loan Default Prediction System

## Run the complete local application

The one-command launcher starts or reuses FastAPI and starts Streamlit. It does
not require PostgreSQL for prediction; PostgreSQL remains available for the
separate Phase 1 data layer.

1. Create a Python 3.11 environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Verify project artifacts and imports:

```powershell
python scripts/check_project.py
```

4. Start the complete local application:

```powershell
python run_app.py
```

5. Open the browser:

- Streamlit UI: `http://localhost:8501`
- FastAPI: `http://127.0.0.1:8000`
- FastAPI Swagger: `http://127.0.0.1:8000/docs`

The UI calls the FastAPI backend for prediction and SHAP explanations. The
launcher checks health endpoints, reuses an already-running API, opens the UI
when a browser is available, and stops only the child services it started when
you press Ctrl+C. Set `API_BASE_URL` in `.env` to use another local API address.

## Project status

**Phase 2 completed:** EDA, data quality checks, feature engineering, preprocessing,
and reproducible train/validation/test splits.

## Dataset

The unmodified source is `data/raw/Loan_default.csv` (255,347 rows, 18 columns).
The target is `Default`: `0` means no default and `1` means default. `LoanID` is
excluded from model features because it is an identifier rather than a meaningful
predictive variable. The raw CSV remains unchanged. Avoid committing real, private,
or large raw data where appropriate.

## EDA and data quality

Open and run `notebooks/01_eda.ipynb` from this directory. It reports shape,
columns, data types, sample rows, descriptive statistics, missingness, exact
duplicate rows, duplicate IDs, categorical frequencies, domain-range flags, IQR
outliers, target imbalance, default rates by category, and numerical correlations.
Outlier/domain flags are reported and retained for review; no valid extremes are
automatically removed. Exact duplicate rows, if any, are omitted from prepared
splits. Category values have leading/trailing whitespace trimmed; capitalization
is reported and preserved. Missing values are handled by the fitted imputers.

For the current supplied dataset there are no missing values or exact duplicate
rows, so no imputation is required for this run. `LoanID` values are checked for
duplicates and kept in the original data, then excluded from ML features.

## Engineered features

Features are computed from pre-decision predictors only. The target is removed
before feature engineering.

- `DTI_Category`: Low at `DTIRatio <= 0.20`, Medium through `0.40`, High above `0.40`.
- `IncomeLoanRatio`: `Income / LoanAmount`; non-positive or missing loan amount
  yields a missing ratio for the numerical imputer.
- `LoanIncomeRatio`: `LoanAmount / Income`; non-positive or missing income yields
  a missing ratio for the numerical imputer.
- `EmploymentCategory`: New below 24 months, Moderate from 24 to under 60 months,
  Experienced at 60 months or more.
- `CreditRiskCategory`: Poor below 580, Fair 580–669, Good 670–739, VeryGood 740+.

## Preprocessing and splits

- Target: `Default` (kept as 0/1 and never scaled)
- Dropped feature: `LoanID`
- Numerical: median imputation followed by `StandardScaler`
- Categorical: most-frequent imputation followed by `OneHotEncoder` with
  `handle_unknown="ignore"`
- Split: 70% train / 15% validation / 15% test, stratified by `Default`
- Random state: 42

The split occurs before fitting the preprocessing pipeline. Imputers, scaler, and
encoder are fitted on training features only; validation and test rows are only
transformed. No resampling is performed. All processed CSV files use the same
encoded feature columns and include the unscaled `Default` target.

## Setup

Use Python 3.11:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

## Prepare processed data

```bash
python scripts/prepare_data.py
```

This writes `data/processed/train.csv`, `validation.csv`, `test.csv`,
`preprocessor.joblib`, and `feature_metadata.json`. The metadata includes feature
lists, split sizes, random seed, and quality findings.

Run tests with:

```bash
pytest
```

## Phase 3: Model training and selection

Run `python scripts/train_models.py` to train Logistic Regression, Random Forest,
and XGBoost on the processed training split. Logistic Regression is evaluated
both with and without balanced class weights. The primary selection metric is
validation ROC-AUC; PR-AUC, precision, recall, F1, accuracy, and confusion
matrices are also reported at a fixed 0.50 threshold.

The 70% training split is used for fitting and three-fold randomized tuning. To
keep tuning practical on this dataset, each randomized search uses a reproducible
stratified sample of at most 20,000 training rows, then refits its selected
configuration on all training rows. The 15% validation split is used for model
comparison and selection. The 15% test split is read only after a winner is
selected and is reserved for final evaluation; it does not influence tuning,
thresholds, or selection.

Results are written to `reports/baseline_model_comparison.csv` and
`reports/model_comparison.csv`. Tuned classifiers are saved under `models/`,
along with `best_model.joblib` and `best_model_metadata.json`. The best-model
artifact bundles Phase 2 feature engineering, the already-fitted Phase 2
preprocessor, and the selected classifier, so it accepts raw predictor columns
without `LoanID` or `Default`.

**Models:** Logistic Regression, Random Forest, XGBoost. **Primary metric:**
ROC-AUC. **Secondary metrics:** PR-AUC, precision, recall, F1, and accuracy.
Splits remain 70% train / 15% validation / 15% test, random state 42.

`notebooks/02_model_analysis.ipynb` presents the training workflow, validation
comparisons, ROC and precision-recall curves, confusion matrices, and final test
evaluation. SHAP, API serving, monitoring, and deployment remain out of scope.

## Phase 4: Model optimization and business evaluation

Run `python scripts/optimize_model.py` to recheck the selected Phase 3 model,
compare sigmoid and isotonic calibration using training-only calibration fits,
select a development threshold using validation F1, analyze validation errors,
and evaluate the fixed model and threshold on the test split once at the final
stage. The test split is never used to choose calibration, threshold, or model.

Calibration is adopted only when validation Brier score improves by at least 5%
and ROC-AUC/PR-AUC each decline by no more than 0.01. Otherwise the original
Phase 3 model remains selected. Thresholds within 0.005 of maximum validation
F1 are considered similar; the one with higher default recall is preferred,
then precision. This is a development/portfolio threshold, not a bank credit
policy.

Phase 4 writes `reports/threshold_analysis.csv`, calibration metrics, row-level
and grouped error analyses, stability and risk-band reports, an explicitly
assumed NPA scenario, three validation plots, and `reports/phase4_report.md`.
The assumed intervention success rate is 30%; it is not observed data. Loan
amount is reported as exposure, not loss. Any estimated prevented defaults or
NPA reduction are scenario-based only; real-world reduction requires prospective
intervention and observed repayment outcomes. The final package records the
selected threshold and model version in `models/best_model_metadata.json`.

`notebooks/03_model_optimization.ipynb` reviews these validation analyses and
displays the final test metrics after decisions are fixed. SHAP, APIs, tracking,
monitoring, and deployment remain out of scope.

## Phase 5: SHAP explainability

Run `python scripts/generate_shap_explanations.py` to explain the saved Phase 4
model using a deterministic sample of up to 5,000 validation loans. The reusable
`LoanSHAPExplainer` in `src/explainability/shap_explainer.py` detects the saved
classifier type and reuses the complete model pipeline and fitted Phase 2
preprocessor. One-hot transformed features remain separate and retain the
preprocessor's generated names.

For the current Logistic Regression model, `LinearExplainer` reports Default=1
contributions in log-odds, not probability points. Positive/negative local
contributions describe movement in model output and do not establish real-world
causality. `LoanSHAPExplainer.explain_prediction()` returns JSON-serializable
values for future API integration without changing or retraining the model.
Global/local figures, feature rankings, and the limitations report are saved
under `reports/`; `notebooks/04_shap_explainability.ipynb` provides a reviewable
walkthrough.

## MLflow Tracking

Run `python scripts/run_mlflow_tracking.py` to log the existing Phase 4 model,
parameters, actual validation/test metrics, Phase 2 preprocessor, metadata, and
available reports to the local `mlruns/` file store. No model is retrained.
MLflow 3 marks the local file store as opt-in. The tracking script enables it for
its own process; set the same flag before launching the standalone UI:

```powershell
$env:MLFLOW_ALLOW_FILE_STORE = "true"
mlflow ui --backend-store-uri ./mlruns
```

Then visit `http://127.0.0.1:5000`. `MLFLOW_TRACKING_URI` and
`MLFLOW_EXPERIMENT_NAME` can be changed in `.env`. Generated run data is ignored
by Git, while the concise run summary remains in
`reports/mlflow/mlflow_run_summary.json`.

## Drift Monitoring

Run `python scripts/run_drift_monitoring.py` to compare the encoded Phase 2
training data (`REFERENCE_DATA_PATH`) against a configurable current batch
(`CURRENT_DATA_PATH`, defaulting to validation). The job uses the saved Phase 4
classifier and threshold for prediction distributions; it does not train,
retune, or modify the model. Current batches may omit `Default`; in that case
target drift and current performance are reported as unavailable. When labels
are present, target drift and current-batch classification metrics are included.

Evidently reports, quality checks, schema differences, and drift summaries are
written to `reports/monitoring/`. The configurable `DRIFT_SHARE_THRESHOLD`
(default 0.30) marks overall feature/prediction drift when that share of columns
is flagged; this is an investigation threshold, not a universal statistical
rule. Drift is a monitoring signal and does not by itself establish model
degradation. Production monitoring should compare real incoming batches and
delayed observed outcomes; the current validation file is only a development
simulation.

## Running the API

Install the project dependencies and start the service from the project root:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The service is available at `http://127.0.0.1:8000`, with interactive Swagger
documentation at `http://127.0.0.1:8000/docs` and ReDoc at
`http://127.0.0.1:8000/redoc`.

- `GET /` returns service status and API version.
- `GET /health` returns readiness and the model version loaded from Phase 4 metadata.
- `POST /predict` returns default probability, decision, risk band, threshold, and model version.
- `POST /explain` adds up to five Phase 5 SHAP factors and a generated summary.

Example request:

```json
{
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
  "HasCoSigner": "No"
}
```

Example `POST /predict` response:

```json
{
  "default_probability": 0.1812,
  "predicted_default": 1,
  "decision_threshold": 0.15,
  "risk_band": "Low",
  "model_version": "1.0.0"
}
```

Example `POST /explain` response:

```json
{
  "default_probability": 0.1812,
  "predicted_default": 1,
  "decision_threshold": 0.15,
  "risk_band": "Low",
  "model_version": "1.0.0",
  "summary": "The model's predicted default risk is mainly influenced by Age, InterestRate, and CreditScore.",
  "top_factors": [
    {"feature": "Age", "shap_value": 0.12, "direction": "increases_risk"},
    {"feature": "InterestRate", "shap_value": 0.08, "direction": "increases_risk"},
    {"feature": "CreditScore", "shap_value": -0.04, "direction": "decreases_risk"}
  ]
}
```

The values above are response-shape examples; live values are computed from the
saved model and Phase 4 threshold. `Low`, `Medium`, and `High` risk bands use
probability cutoffs 0.30 and 0.60 and are independent of the 0.15 decision
threshold. Applicant fields are not logged. `/explain` reports model
contributions, not causal effects.
