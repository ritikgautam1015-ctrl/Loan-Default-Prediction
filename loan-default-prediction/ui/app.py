"""Local Streamlit UI that delegates all predictions to the FastAPI service."""

import json
import os
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
NAVIGATION_ITEMS = [
    "Overview",
    "Applications",
    "Risk Assessment",
    "Risk Drivers",
    "Analytics",
    "Monitoring",
    "System",
    "About",
]
MODEL_METADATA_PATH = PROJECT_ROOT / "models" / "best_model_metadata.json"
FEATURE_METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "feature_metadata.json"
MODEL_COMPARISON_PATH = PROJECT_ROOT / "reports" / "model_comparison.csv"
THRESHOLD_ANALYSIS_PATH = PROJECT_ROOT / "reports" / "threshold_analysis.csv"
RISK_BAND_ANALYSIS_PATH = PROJECT_ROOT / "reports" / "risk_band_analysis.csv"
DRIFT_SUMMARY_PATH = PROJECT_ROOT / "reports" / "monitoring" / "drift_summary.json"
DRIFT_REPORT_PATH = PROJECT_ROOT / "reports" / "monitoring" / "drift_report.html"
QUALITY_REPORT_PATH = PROJECT_ROOT / "reports" / "monitoring" / "data_quality_report.html"
MLFLOW_SUMMARY_PATH = PROJECT_ROOT / "reports" / "mlflow" / "mlflow_run_summary.json"

st.set_page_config(
    page_title="LoanRisk | Credit Risk Assessment Platform",
    page_icon="L",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_visual_system() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #0f172a;
            --muted: #64748b;
            --paper: #f8fafc;
            --card-bg: #ffffff;
            --line: #e2e8f0;
            --navy: #0f2038;
            --navy-dark: #0a1626;
            --blue: #2563eb;
            --blue-dark: #1d4ed8;
            --green: #15803d;
            --green-bg: #dcfce7;
            --green-border: #bbf7d0;
            --amber: #b45309;
            --amber-bg: #fef3c7;
            --amber-border: #fde68a;
            --red: #b91c1c;
            --red-bg: #fee2e2;
            --red-border: #fecaca;
        }
        .stApp, [data-testid="stAppViewContainer"] {
            background: var(--paper);
            color: var(--ink);
            font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, "Inter", Roboto, sans-serif;
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stMainBlockContainer"] {
            width: 100%;
            max-width: 1400px;
            box-sizing: border-box;
            padding-top: 1.8rem;
            padding-bottom: 4rem;
        }
        [data-testid="stSidebar"] {
            background: #0f1f33;
            border-right: 1px solid #1e324d;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span { color: #f1f5f9; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            border-radius: 8px;
            padding: 0.52rem 0.75rem;
            font-size: 0.88rem;
            font-weight: 500;
            color: #cbd5e1;
            transition: all 120ms ease;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            background: #1a3150;
            color: #ffffff;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
            background: #2563eb;
            color: #ffffff;
            font-weight: 600;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 10px;
            padding: 0.9rem 1.1rem;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        }
        [data-testid="stMetricLabel"] {
            color: var(--muted);
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        [data-testid="stMetricValue"] {
            color: var(--ink);
            font-size: 1.65rem;
            font-weight: 750;
        }
        [data-testid="stBaseButton-primary"] {
            background: #0f2038;
            border: 1px solid #0f2038;
            border-radius: 7px;
            color: #ffffff;
            font-weight: 600;
            min-height: 2.7rem;
            transition: background 150ms ease;
        }
        [data-testid="stBaseButton-primary"]:hover {
            background: #1d4ed8;
            border-color: #1d4ed8;
            color: #ffffff;
        }
        [data-testid="stBaseButton-secondary"] {
            background: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 7px;
            color: #0f172a;
            font-weight: 600;
            min-height: 2.7rem;
        }
        [data-testid="stBaseButton-secondary"]:hover {
            border-color: #94a3b8;
            background: #f8fafc;
        }
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stSelectbox"] [role="combobox"] {
            border-radius: 7px;
            border-color: #cbd5e1;
            background: #ffffff;
            color: var(--ink);
        }
        [data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 9px;
            overflow: hidden;
            background: #ffffff;
        }
        [data-testid="stAlert"] { border-radius: 8px; }
        hr { border-color: var(--line); }

        /* Brand lockup */
        .brand-lockup {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.4rem 0 1.4rem;
        }
        .brand-mark {
            width: 36px;
            height: 36px;
            display: grid;
            place-items: center;
            border-radius: 8px;
            color: #ffffff;
            background: #2563eb;
            font-weight: 800;
            font-size: 1.05rem;
            letter-spacing: -0.02em;
        }
        .brand-name {
            color: #ffffff;
            font-weight: 750;
            font-size: 1rem;
            letter-spacing: 0.02em;
        }
        .brand-caption {
            color: #94a3b8;
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            margin-top: 1px;
            text-transform: uppercase;
        }

        /* Top Bar */
        .top-brand-lockup {
            display: flex;
            flex-direction: column;
        }
        .top-brand-name {
            font-size: 1.4rem;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: -0.02em;
            line-height: 1.1;
        }
        .top-brand-sub {
            font-size: 0.82rem;
            font-weight: 600;
            color: #64748b;
            margin-top: 2px;
        }
        .top-status-indicator {
            display: flex;
            align-items: center;
            justify-content: flex-end;
            gap: 6px;
            font-size: 0.84rem;
            font-weight: 600;
            color: #0f172a;
        }
        .status-dot {
            font-size: 0.75rem;
        }
        .status-ready { color: #16a34a; }
        .status-warn { color: #d97706; }
        .header-divider {
            border: none;
            border-top: 1px solid #e2e8f0;
            margin: 0.6rem 0 1.25rem 0;
        }

        /* Page Headers */
        .page-kicker {
            color: var(--blue);
            font-size: 0.72rem;
            font-weight: 750;
            text-transform: uppercase;
            letter-spacing: 0.11em;
            margin-bottom: 0.3rem;
        }
        .page-title {
            color: var(--ink);
            font-weight: 800;
            font-size: 2.1rem;
            line-height: 1.15;
            letter-spacing: -0.025em;
            margin: 0;
        }
        .page-description {
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.5;
            margin-top: 0.45rem;
            margin-bottom: 1.25rem;
        }

        /* Section Headings */
        .section-label {
            color: var(--ink);
            font-size: 1.12rem;
            font-weight: 750;
            letter-spacing: -0.01em;
            margin: 1.6rem 0 0.75rem;
        }

        /* Hero Right Visual Card */
        .hero-assessment-card {
            width: 100%;
            max-width: 100%;
            min-width: 0;
            box-sizing: border-box;
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 11px;
            padding: 1.4rem 1.5rem;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.035);
        }
        .card-eyebrow {
            font-size: 0.72rem;
            font-weight: 750;
            letter-spacing: 0.1em;
            color: var(--muted);
            text-transform: uppercase;
            margin-bottom: 0.3rem;
        }
        .card-subhead {
            font-size: 0.95rem;
            font-weight: 700;
            color: var(--ink);
            margin-bottom: 1rem;
        }
        .metric-block {
            background: #f8fafc;
            border: 1px solid #edf2f7;
            border-radius: 8px;
            padding: 1.1rem 1.2rem;
            text-align: center;
            margin-bottom: 0.9rem;
        }
        .metric-caption {
            font-size: 0.74rem;
            font-weight: 600;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.35rem;
        }
        .metric-large {
            font-size: 2.7rem;
            font-weight: 800;
            color: var(--ink);
            line-height: 1.05;
            margin-bottom: 0.55rem;
            letter-spacing: -0.03em;
        }
        .risk-badge {
            display: inline-block;
            padding: 0.3rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 750;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .chip-low {
            color: var(--green);
            background: var(--green-bg);
            border: 1px solid var(--green-border);
        }
        .chip-medium {
            color: var(--amber);
            background: var(--amber-bg);
            border: 1px solid var(--amber-border);
        }
        .chip-high {
            color: var(--red);
            background: var(--red-bg);
            border: 1px solid var(--red-border);
        }
        .card-footer-text {
            font-size: 0.78rem;
            font-weight: 600;
            color: var(--muted);
            text-align: center;
            margin-bottom: 0.75rem;
        }

        /* Empty State inside Hero */
        .empty-state-visual {
            text-align: center;
            padding: 1.3rem 0.5rem 0.8rem;
        }
        .empty-icon-shield {
            margin-bottom: 0.75rem;
            display: flex;
            justify-content: center;
        }
        .empty-card-title {
            font-size: 0.88rem;
            font-weight: 750;
            color: var(--ink);
            letter-spacing: 0.06em;
            margin-bottom: 0.4rem;
        }
        .empty-card-desc {
            font-size: 0.83rem;
            color: var(--muted);
            line-height: 1.45;
            max-width: 250px;
            margin: 0 auto;
        }

        /* Action Cards */
        .action-card-title {
            font-size: 1rem;
            font-weight: 700;
            color: var(--ink);
            margin-bottom: 0.3rem;
        }
        .action-card-desc {
            font-size: 0.84rem;
            color: var(--muted);
            min-height: 2.3rem;
            line-height: 1.4;
            margin-bottom: 0.85rem;
        }

        /* Recent Assessments List */
        .recent-id {
            font-size: 0.95rem;
            font-weight: 750;
            color: var(--ink);
        }
        .recent-meta {
            font-size: 0.77rem;
            color: var(--muted);
            margin-top: 2px;
        }
        .recent-prob {
            font-size: 1.25rem;
            font-weight: 800;
            color: var(--ink);
            letter-spacing: -0.02em;
        }
        .risk-chip {
            display: inline-block;
            padding: 0.28rem 0.65rem;
            border-radius: 6px;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }
        .risk-chip-low { color: #15803d; background: #dcfce7; }
        .risk-chip-medium { color: #9a4c06; background: #fef3c7; }
        .risk-chip-high { color: #b91c1c; background: #fee2e2; }

        .empty-recent-block {
            text-align: center;
            padding: 1.8rem 1rem;
        }
        .empty-recent-title {
            font-size: 1.02rem;
            font-weight: 700;
            color: var(--ink);
            margin-bottom: 0.3rem;
        }
        .empty-recent-subtitle {
            font-size: 0.86rem;
            color: var(--muted);
            margin-bottom: 1.1rem;
        }

        /* Guidance Steps */
        .step-num {
            font-size: 0.85rem;
            font-weight: 800;
            color: var(--blue);
            margin-bottom: 0.4rem;
            letter-spacing: 0.05em;
        }
        .step-title {
            font-size: 0.96rem;
            font-weight: 700;
            color: var(--ink);
            margin-bottom: 0.35rem;
            line-height: 1.25;
        }
        .step-desc {
            font-size: 0.82rem;
            color: var(--muted);
            line-height: 1.45;
        }

        /* Assessment Result Panel */
        .result-panel {
            background: #ffffff;
            border: 1px solid var(--line);
            border-left: 4px solid var(--blue);
            border-radius: 9px;
            padding: 1.2rem 1.4rem;
            margin: 0.9rem 0 0.8rem;
        }
        .result-eyebrow {
            color: var(--muted);
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        .result-grid {
            display: grid;
            grid-template-columns: 1.4fr 1.1fr 1fr 1fr;
            gap: 1.1rem;
            margin-top: 0.65rem;
        }
        .result-value {
            color: var(--ink);
            font-size: 1.15rem;
            font-weight: 750;
        }
        .result-value.primary {
            color: var(--blue);
            font-size: 1.85rem;
            letter-spacing: -0.02em;
        }
        .result-label {
            color: var(--muted);
            font-size: 0.74rem;
            font-weight: 600;
            margin-bottom: 0.2rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .risk-low { border-left-color: var(--green); }
        .risk-medium { border-left-color: var(--amber); }
        .risk-high { border-left-color: var(--red); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(kicker: str, title: str, description: str) -> None:
    st.markdown(
        f"<div class='page-kicker'>{escape(kicker)}</div>"
        f"<h1 class='page-title'>{escape(title)}</h1>"
        f"<div class='page-description'>{escape(description)}</div>",
        unsafe_allow_html=True,
    )


def navigate_to(page: str) -> None:
    st.session_state["loanrisk_navigation_target"] = page
    st.rerun()


def get_assessment_history() -> list[dict[str, Any]]:
    history = st.session_state.setdefault("loanrisk_assessment_history", [])
    previous_result = st.session_state.get("loan_default_last_prediction")
    previous_payload = st.session_state.get("loan_default_last_payload")
    if previous_result and previous_payload and not history:
        history.append(
            {
                "assessment_id": "Assessment #001",
                "created_at": datetime.now().astimezone().isoformat(),
                "payload": previous_payload,
                "prediction": previous_result,
                "explanation": st.session_state.get("loan_default_last_explanation"),
            }
        )
    return history


def record_assessment(payload: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    history = get_assessment_history()
    entry = {
        "assessment_id": f"Assessment #{len(history) + 1:03d}",
        "created_at": datetime.now().astimezone().isoformat(),
        "payload": payload,
        "prediction": prediction,
        "explanation": None,
    }
    history.append(entry)
    st.session_state["loanrisk_active_assessment_id"] = entry["assessment_id"]
    return entry


def get_active_assessment() -> dict[str, Any] | None:
    history = get_assessment_history()
    active_id = st.session_state.get("loanrisk_active_assessment_id")
    for assessment in history:
        if assessment["assessment_id"] == active_id:
            return assessment
    return history[-1] if history else None


def format_assessment_time(iso_str: str) -> str:
    try:
        dt = pd.to_datetime(iso_str).astimezone()
        now = datetime.now().astimezone()
        if dt.date() == now.date():
            return f"Today · {dt.strftime('%I:%M %p')}"
        elif (now.date() - dt.date()).days == 1:
            return f"Yesterday · {dt.strftime('%I:%M %p')}"
        else:
            return dt.strftime("%b %d · %I:%M %p")
    except Exception:
        return "Recent"


@st.cache_data(show_spinner=False)
def load_project_data() -> dict[str, Any]:
    """Load small metadata and reports, never the raw/training dataset."""
    result: dict[str, Any] = {
        "model": {},
        "features": {},
        "comparison": pd.DataFrame(),
        "drift": {},
    }
    if MODEL_METADATA_PATH.is_file():
        result["model"] = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    if FEATURE_METADATA_PATH.is_file():
        result["features"] = json.loads(FEATURE_METADATA_PATH.read_text(encoding="utf-8"))
    if MODEL_COMPARISON_PATH.is_file():
        result["comparison"] = pd.read_csv(MODEL_COMPARISON_PATH)
    if DRIFT_SUMMARY_PATH.is_file():
        result["drift"] = json.loads(DRIFT_SUMMARY_PATH.read_text(encoding="utf-8"))
    return result


@st.cache_data(ttl=10, show_spinner=False)
def check_api_health(api_base_url: str) -> tuple[bool, str]:
    try:
        response = requests.get(f"{api_base_url}/health", timeout=2)
        response.raise_for_status()
        payload = response.json()
        return bool(payload.get("model_loaded")), str(payload.get("model_version", "unknown"))
    except (requests.RequestException, ValueError):
        return False, "unavailable"


def api_post(api_base_url: str, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        response = requests.post(
            f"{api_base_url}{endpoint}", json=payload, timeout=45
        )
    except requests.RequestException as error:
        raise RuntimeError(
            "Unable to connect to the prediction service. Please make sure the FastAPI backend is running."
        ) from error
    try:
        body = response.json()
    except ValueError as error:
        raise RuntimeError("The prediction service returned an unreadable response.") from error
    if not response.ok:
        detail = body.get("detail") if isinstance(body, dict) else None
        if response.status_code == 422:
            raise ValueError(f"Please review the application fields. {detail or ''}".strip())
        raise RuntimeError("The prediction service could not complete this request.")
    if not isinstance(body, dict):
        raise RuntimeError("The prediction service returned an unexpected response format.")
    required = {
        "default_probability", "predicted_default", "decision_threshold",
        "risk_band", "model_version",
    }
    if endpoint == "/explain":
        required.update({"summary", "top_factors"})
    if not required.issubset(body):
        raise RuntimeError("The prediction service returned an unexpected response format.")
    try:
        probability = float(body["default_probability"])
        threshold = float(body["decision_threshold"])
        predicted = int(body["predicted_default"])
        if not (0 <= probability <= 1 and 0 <= threshold <= 1 and predicted in {0, 1}):
            raise ValueError
        if endpoint == "/explain":
            factors = body["top_factors"]
            if not isinstance(factors, list) or len(factors) > 5:
                raise ValueError
            for factor in factors:
                if (
                    not isinstance(factor, dict)
                    or not isinstance(factor.get("feature"), str)
                    or factor.get("direction") not in {"increases_risk", "decreases_risk"}
                    or not isinstance(float(factor.get("shap_value")), float)
                ):
                    raise ValueError
    except (TypeError, ValueError):
        raise RuntimeError("The prediction service returned an invalid response format.")
    return body


def category_options(feature_metadata: dict[str, Any], feature: str) -> list[str]:
    counts = (
        feature_metadata.get("quality_report", {})
        .get("categorical_value_counts", {})
        .get(feature, {})
    )
    return sorted(counts, key=lambda value: (-counts[value], value))


def show_prediction(result: dict[str, Any], title: str = "Assessment Result") -> None:
    probability = float(result["default_probability"])
    risk_band = str(result["risk_band"])
    predicted = int(result["predicted_default"])
    threshold = float(result["decision_threshold"])
    risk_class = {"Low": "risk-low", "Medium": "risk-medium", "High": "risk-high"}.get(
        risk_band, ""
    )
    risk_name = {"Low": "Lower Risk", "Medium": "Moderate Risk", "High": "Higher Risk"}.get(
        risk_band, f"{risk_band} Risk"
    )
    st.markdown(
        f"""<div class="result-panel {risk_class}">
        <div class="result-eyebrow">{escape(title)} · Risk Assessment Output</div>
        <div class="result-grid">
          <div><div class="result-label">Default Probability</div><div class="result-value primary">{probability:.1%}</div></div>
          <div><div class="result-label">Risk Classification</div><div class="result-value">{escape(risk_name)}</div></div>
          <div><div class="result-label">Decision Threshold</div><div class="result-value">{threshold:.2f}</div></div>
          <div><div class="result-label">Assessment Status</div><div class="result-value">Completed</div></div>
        </div></div>""",
        unsafe_allow_html=True,
    )
    st.progress(min(max(probability, 0.0), 1.0), text="Estimated Risk Scale")
    if predicted == 1:
        st.warning("Assessment notice: Estimated risk exceeds the standard operational threshold.")
    else:
        st.success("Assessment notice: Estimated risk is within normal acceptable thresholds.")


def show_dashboard(data: dict[str, Any], api_base_url: str) -> None:
    # Desired top bar
    top_left, top_right = st.columns([2.3, 1], vertical_alignment="center")
    with top_left:
        st.markdown(
            """<div class="top-brand-lockup">
                <div class="top-brand-name">LoanRisk</div>
                <div class="top-brand-sub">Credit Risk Assessment Platform</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with top_right:
        is_ready, _ = check_api_health(api_base_url)
        status_label = "System Ready" if is_ready else "Connecting..."
        status_dot_class = "status-ready" if is_ready else "status-warn"
        st.markdown(
            f"""<div class="top-status-indicator">
                <span class="status-dot {status_dot_class}">●</span> {status_label}
            </div>""",
            unsafe_allow_html=True,
        )
    st.markdown("<hr class='header-divider'/>", unsafe_allow_html=True)

    history = get_assessment_history()
    current = get_active_assessment()

    # Hero section
    with st.container(border=True):
        hero_left, hero_right = st.columns([1.2, 1], gap="large", vertical_alignment="center")
        with hero_left:
            st.markdown(
                """<div class="hero-kicker">LOANRISK</div>
                <h1 class="page-title">Credit risk assessment,<br>made simple.</h1>
                <div class="page-description">Evaluate loan applications, understand estimated default risk, and review the factors influencing each assessment.</div>""",
                unsafe_allow_html=True,
            )
            action_left, action_right = st.columns([1, 1.15], gap="small")
            with action_left:
                if st.button("Start Risk Assessment", key="hero-start", type="primary", use_container_width=True):
                    navigate_to("Risk Assessment")
            with action_right:
                if st.button("View Recent Assessments", key="hero-recent", use_container_width=True):
                    navigate_to("Applications")

        with hero_right:
            if current:
                result = current["prediction"]
                probability = float(result["default_probability"])
                band = result.get("risk_band", "Low")
                risk_label = {"Low": "LOWER RISK", "Medium": "MODERATE RISK", "High": "HIGHER RISK"}.get(
                    band, f"{band.upper()} RISK"
                )
                risk_chip_class = {"Low": "chip-low", "Medium": "chip-medium", "High": "chip-high"}.get(
                    band, "chip-low"
                )
                st.markdown(
                    f"""<div class="hero-assessment-card">
                        <div class="card-eyebrow">RISK ASSESSMENT</div>
                        <div class="card-subhead">Current Assessment</div>
                        <div class="metric-block">
                            <div class="metric-caption">Estimated Risk</div>
                            <div class="metric-large">{probability:.1%}</div>
                            <div class="risk-badge {risk_chip_class}">{escape(risk_label)}</div>
                        </div>
                        <div class="card-footer-text">Assessment ready</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                if st.button("Review Risk Drivers", key="hero-review-drivers", use_container_width=True):
                    st.session_state["loanrisk_active_assessment_id"] = current["assessment_id"]
                    navigate_to("Risk Drivers")
            else:
                st.markdown(
                    """<div class="hero-assessment-card">
                        <div class="card-eyebrow">RISK ASSESSMENT</div>
                        <div class="card-subhead">Current Assessment</div>
                        <div class="empty-state-visual">
                            <div class="empty-icon-shield">
                                <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
                                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                                    <path d="m9 12 2 2 4-4"/>
                                </svg>
                            </div>
                            <div class="empty-card-title">READY FOR ASSESSMENT</div>
                            <div class="empty-card-desc">Enter an application to generate a risk assessment.</div>
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )

    # Recent Assessments Section
    st.markdown('<div class="section-label">Recent Assessments</div>', unsafe_allow_html=True)
    if history:
        for assessment in reversed(history[-5:]):
            result = assessment["prediction"]
            band = result.get("risk_band", "Low")
            risk_name = {"Low": "Lower Risk", "Medium": "Moderate Risk", "High": "Higher Risk"}.get(
                band, f"{band} Risk"
            )
            time_str = format_assessment_time(assessment["created_at"])
            prob_str = f"{float(result['default_probability']):.1%}"
            risk_slug = {"Low": "low", "Medium": "medium", "High": "high"}.get(band, "low")

            with st.container(border=True):
                r_col1, r_col2, r_col3, r_col4 = st.columns([1.5, 1.2, 1.2, 0.9], vertical_alignment="center")
                with r_col1:
                    st.markdown(
                        f"<div class='recent-id'>{escape(assessment['assessment_id'])}</div>"
                        f"<div class='recent-meta'>{escape(time_str)}</div>",
                        unsafe_allow_html=True,
                    )
                with r_col2:
                    st.markdown(
                        f"<span class='risk-chip risk-chip-{risk_slug}'>{escape(risk_name)}</span>",
                        unsafe_allow_html=True,
                    )
                with r_col3:
                    st.markdown(
                        f"<div class='recent-prob'>{escape(prob_str)}</div>"
                        f"<div class='recent-meta'>Estimated risk</div>",
                        unsafe_allow_html=True,
                    )
                with r_col4:
                    if st.button("Review", key=f"open-{assessment['assessment_id']}", use_container_width=True):
                        st.session_state["loanrisk_active_assessment_id"] = assessment["assessment_id"]
                        navigate_to("Risk Drivers")
    else:
        with st.container(border=True):
            st.markdown(
                """<div class="empty-recent-block">
                    <div class="empty-recent-title">No assessments yet.</div>
                    <div class="empty-recent-subtitle">Start your first risk assessment to see results here.</div>
                </div>""",
                unsafe_allow_html=True,
            )
            empty_btn_left, empty_btn_mid, empty_btn_right = st.columns([1, 1, 1])
            with empty_btn_mid:
                if st.button("Start Assessment", key="empty-start-assessment", type="primary", use_container_width=True):
                    navigate_to("Risk Assessment")

    # Quick Actions Section
    st.markdown('<div class="section-label">Quick Actions</div>', unsafe_allow_html=True)
    qa_col1, qa_col2, qa_col3 = st.columns(3, gap="medium")
    with qa_col1:
        with st.container(border=True):
            st.markdown(
                """<div class="action-card-title">New Risk Assessment</div>
                <div class="action-card-desc">Evaluate a loan application.</div>""",
                unsafe_allow_html=True,
            )
            if st.button("Start Assessment", key="qa-new", use_container_width=True, type="primary"):
                navigate_to("Risk Assessment")

    with qa_col2:
        with st.container(border=True):
            st.markdown(
                """<div class="action-card-title">Recent Assessments</div>
                <div class="action-card-desc">Review your latest risk assessments.</div>""",
                unsafe_allow_html=True,
            )
            if st.button("View Assessments", key="qa-recent", use_container_width=True):
                navigate_to("Applications")

    with qa_col3:
        with st.container(border=True):
            st.markdown(
                """<div class="action-card-title">Risk Insights</div>
                <div class="action-card-desc">Understand the factors influencing an assessment.</div>""",
                unsafe_allow_html=True,
            )
            if st.button("View Risk Drivers", key="qa-drivers", use_container_width=True):
                navigate_to("Risk Drivers")

    # Risk Activity (Only show if actual history exists today)
    today = datetime.now().astimezone().date()
    todays_history = [
        item for item in history
        if datetime.fromisoformat(item["created_at"]).astimezone().date() == today
    ]
    if todays_history:
        lower = sum(item["prediction"].get("risk_band") == "Low" for item in todays_history)
        moderate = sum(item["prediction"].get("risk_band") == "Medium" for item in todays_history)
        higher = sum(item["prediction"].get("risk_band") == "High" for item in todays_history)
        st.markdown("<div class='section-label'>Today's Activity</div>", unsafe_allow_html=True)
        act_col1, act_col2, act_col3, act_col4 = st.columns(4, gap="medium")
        with act_col1:
            st.metric("Total Assessments", len(todays_history))
        with act_col2:
            st.metric("Higher Risk", higher)
        with act_col3:
            st.metric("Moderate Risk", moderate)
        with act_col4:
            st.metric("Lower Risk", lower)

    # Product Guidance (How it works)
    st.markdown('<div class="section-label">How it works</div>', unsafe_allow_html=True)
    g_col1, g_col2, g_col3, g_col4 = st.columns(4, gap="medium")
    steps_data = [
        ("01", "Enter application details", "Provide borrower income, credit history, and loan facility parameters."),
        ("02", "Receive risk assessment", "Obtain a model-estimated default risk assessment and classification."),
        ("03", "Review risk drivers", "Examine the main applicant factors that contributed to the assessment."),
        ("04", "Make informed review", "Use assessment findings as structured decision-support for lending decisions."),
    ]
    for col, (num, title, desc) in zip([g_col1, g_col2, g_col3, g_col4], steps_data):
        with col:
            with st.container(border=True):
                st.markdown(
                    f"""<div class="step-num">{num}</div>
                    <div class="step-title">{escape(title)}</div>
                    <div class="step-desc">{escape(desc)}</div>""",
                    unsafe_allow_html=True,
                )


def show_prediction_page(data: dict[str, Any], api_base_url: str) -> None:
    page_header(
        "Application Evaluation",
        "Risk Assessment",
        "Enter application details to generate a model-based credit risk assessment.",
    )
    category_metadata = data["features"]
    categorical_fields = [
        "Education", "EmploymentType", "MaritalStatus", "HasMortgage",
        "HasDependents", "LoanPurpose", "HasCoSigner",
    ]
    options = {field: category_options(category_metadata, field) for field in categorical_fields}
    missing_categories = [field for field, values in options.items() if not values]
    if missing_categories:
        st.error("Category metadata is unavailable for: " + ", ".join(missing_categories))
        return

    with st.form("loan_application_form"):
        st.markdown('<div class="section-label">01 · Applicant Profile</div>', unsafe_allow_html=True)
        applicant = st.columns(3)
        age = applicant[0].number_input("Age", min_value=1.0, value=35.0, step=1.0)
        income = applicant[1].number_input("Income", min_value=0.0, value=60000.0, step=1000.0)
        education = applicant[2].selectbox("Education", options["Education"])
        employment_col = st.columns(3)
        employment = employment_col[0].selectbox("Employment Type", options["EmploymentType"])
        marital_status = employment_col[1].selectbox("Marital Status", options["MaritalStatus"])
        months_employed = employment_col[2].number_input(
            "Months Employed", min_value=0.0, value=60.0, step=1.0
        )

        st.markdown('<div class="section-label">02 · Credit Profile</div>', unsafe_allow_html=True)
        credit = st.columns(3)
        credit_score = credit[0].number_input("Credit Score", min_value=0.0, value=680.0, step=1.0)
        credit_lines = credit[1].number_input("Number of Credit Lines", min_value=0.0, value=4.0, step=1.0)
        dti_ratio = credit[2].number_input(
            "Debt-to-Income Ratio", min_value=0.0, value=0.32, step=0.01, format="%.2f"
        )

        st.markdown('<div class="section-label">03 · Facility Terms</div>', unsafe_allow_html=True)
        loan = st.columns(3)
        loan_amount = loan[0].number_input("Loan Amount", min_value=0.0, value=25000.0, step=1000.0)
        interest_rate = loan[1].number_input(
            "Interest Rate", min_value=0.0, value=8.5, step=0.1, format="%.2f"
        )
        loan_term = loan[2].number_input("Loan Term (months)", min_value=1.0, value=36.0, step=1.0)
        purpose = st.selectbox("Loan Purpose", options["LoanPurpose"])

        st.markdown('<div class="section-label">04 · Additional Circumstances</div>', unsafe_allow_html=True)
        additional = st.columns(3)
        mortgage = additional[0].selectbox("Has Mortgage", options["HasMortgage"])
        dependents = additional[1].selectbox("Has Dependents", options["HasDependents"])
        cosigner = additional[2].selectbox("Has Co-Signer", options["HasCoSigner"])
        submitted = st.form_submit_button("Assess Risk", type="primary", use_container_width=True)

    if submitted:
        payload = {
            "Age": age,
            "Income": income,
            "LoanAmount": loan_amount,
            "CreditScore": credit_score,
            "MonthsEmployed": months_employed,
            "NumCreditLines": credit_lines,
            "InterestRate": interest_rate,
            "LoanTerm": loan_term,
            "DTIRatio": dti_ratio,
            "Education": education,
            "EmploymentType": employment,
            "MaritalStatus": marital_status,
            "HasMortgage": mortgage,
            "HasDependents": dependents,
            "LoanPurpose": purpose,
            "HasCoSigner": cosigner,
        }
        try:
            result = api_post(api_base_url, "/predict", payload)
            assessment = record_assessment(payload, result)
            st.session_state["loan_default_last_payload"] = payload
            st.session_state["loan_default_last_prediction"] = result
            st.session_state["loan_default_last_assessment_id"] = assessment["assessment_id"]
            st.session_state.pop("loan_default_last_explanation", None)
        except (RuntimeError, ValueError) as error:
            st.session_state.pop("loan_default_last_prediction", None)
            st.session_state.pop("loan_default_last_explanation", None)
            st.error(str(error))

    result = st.session_state.get("loan_default_last_prediction")
    if result:
        st.divider()
        st.markdown('<div class="section-label">Assessment Result</div>', unsafe_allow_html=True)
        show_prediction(result, title=st.session_state.get("loan_default_last_assessment_id", "Current Assessment"))
        btn_col1, btn_col2 = st.columns([1, 2])
        with btn_col1:
            if st.button("Review Risk Drivers", type="primary", use_container_width=True):
                navigate_to("Risk Drivers")


def show_explanation(explanation: dict[str, Any]) -> None:
    st.markdown('<div class="section-label">Risk Rationale & Key Factors</div>', unsafe_allow_html=True)
    summary_text = explanation.get("summary", "No summary was returned.")
    st.info(summary_text)

    factors = explanation.get("top_factors", [])
    if not factors:
        st.info("No explanatory factors were returned for this application.")
        return
    frame = pd.DataFrame(factors)
    frame["shap_value"] = pd.to_numeric(frame["shap_value"], errors="coerce")
    frame = frame.dropna(subset=["shap_value"]).sort_values(
        "shap_value", key=lambda values: values.abs(), ascending=False
    )
    if frame.empty:
        st.warning("The explanation response did not contain valid numeric factors.")
        return

    increasing = frame.loc[frame["direction"] == "increases_risk"]
    decreasing = frame.loc[frame["direction"] == "decreases_risk"]

    left, right = st.columns(2, gap="medium")
    with left:
        with st.container(border=True):
            st.markdown("**Factors Increasing Risk**")
            if increasing.empty:
                st.caption("No significant positive risk drivers identified.")
            else:
                for row in increasing.itertuples(index=False):
                    st.write(f"• **{row.feature}** (Contribution: +{row.shap_value:.4f})")
    with right:
        with st.container(border=True):
            st.markdown("**Factors Decreasing Risk**")
            if decreasing.empty:
                st.caption("No significant risk-mitigating factors identified.")
            else:
                for row in decreasing.itertuples(index=False):
                    st.write(f"• **{row.feature}** (Contribution: {row.shap_value:.4f})")

    st.markdown('<div class="section-label">Contribution Impact Chart</div>', unsafe_allow_html=True)
    chart = frame.assign(
        increases_risk=frame["shap_value"].clip(lower=0),
        decreases_risk=frame["shap_value"].clip(upper=0),
    ).set_index("feature")[["increases_risk", "decreases_risk"]]
    st.bar_chart(
        chart,
        horizontal=True,
        color=["#b91c1c", "#15803d"],
        stack=False,
    )
    st.caption(
        "Positive values push estimated risk higher; negative values indicate risk-reducing borrower factors."
    )


def show_risk_drivers(api_base_url: str) -> None:
    page_header(
        "Assessment Rationale",
        "Risk Drivers",
        "Review the key factors and contributions influencing the risk assessment.",
    )
    assessment = get_active_assessment()
    if not assessment:
        st.info("Complete a risk assessment to review its drivers.")
        if st.button("Start Risk Assessment", type="primary"):
            navigate_to("Risk Assessment")
        return

    prediction = assessment["prediction"]
    show_prediction(prediction, title=assessment["assessment_id"])

    explanation = assessment.get("explanation")
    if not explanation:
        try:
            explanation = api_post(api_base_url, "/explain", assessment["payload"])
            assessment["explanation"] = explanation
        except (RuntimeError, ValueError) as error:
            st.error(str(error))
            return

    show_explanation(explanation)

    with st.expander("Technical SHAP Details"):
        model_name = load_project_data()["model"].get("model_name", "")
        units = "log-odds for Default=1" if model_name == "Logistic Regression" else "model output units"
        st.caption(
            f"SHAP contribution units: {units}. These values quantify attribution weights for the current "
            "prediction model and are provided for internal audit and technical explainability."
        )
        st.json(explanation.get("top_factors", []))


def show_model_performance(data: dict[str, Any]) -> None:
    page_header(
        "Model Analytics & Evaluation",
        "Analytics",
        "Technical performance benchmarks, operating threshold analysis, and risk-band distributions.",
    )
    metadata = data["model"]
    st.write(
        f"Production Model: **{metadata.get('model_name', 'Unavailable')}** "
        f"(Version: {metadata.get('model_version', 'unknown')})."
    )
    metric_keys = ["ROC_AUC", "PR_AUC", "Accuracy", "Precision", "Recall", "F1", "Brier_Score"]
    validation = metadata.get("validation_metrics", {})
    test = metadata.get("test_metrics", {})
    rows = []
    for metric in metric_keys:
        rows.append(
            {
                "Metric": metric.replace("_", " "),
                "Validation": validation.get(metric),
                "Test Holdout": test.get(metric),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(
        "Validation metrics reflect optimal decision threshold selection; Test holdout figures reflect final generalization."
    )

    comparison = data["comparison"]
    if not comparison.empty:
        st.markdown('<div class="section-label">Model Comparison</div>', unsafe_allow_html=True)
        st.dataframe(comparison, hide_index=True, use_container_width=True)
        st.caption("Active model evaluated against baseline algorithms on identical test partitions.")

    if THRESHOLD_ANALYSIS_PATH.is_file():
        st.markdown('<div class="section-label">Operating Threshold Analysis</div>', unsafe_allow_html=True)
        thresholds = pd.read_csv(THRESHOLD_ANALYSIS_PATH)
        st.dataframe(thresholds, hide_index=True, use_container_width=True)

    if RISK_BAND_ANALYSIS_PATH.is_file():
        st.markdown('<div class="section-label">Risk-Band Distribution</div>', unsafe_allow_html=True)
        st.dataframe(pd.read_csv(RISK_BAND_ANALYSIS_PATH), hide_index=True, use_container_width=True)


def _show_html_download(path: Path, label: str, key: str) -> None:
    if path.is_file():
        st.download_button(
            label,
            data=path.read_bytes(),
            file_name=path.name,
            mime="text/html",
            key=key,
        )
    else:
        st.info(f"Report unavailable: {path.relative_to(PROJECT_ROOT)}")


def show_drift_monitoring(data: dict[str, Any]) -> None:
    page_header(
        "Portfolio & Data Quality Surveillance",
        "Monitoring",
        "Reference-to-current distribution checks, feature stability, and prediction drift monitoring.",
    )
    drift = data["drift"]
    if not drift:
        st.warning("No monitoring summary is available. Run `python scripts/run_drift_monitoring.py`.")
        return
    status = st.columns(5)
    status[0].metric("Reference Rows", f"{drift.get('reference_rows', 0):,}")
    status[1].metric("Current Rows", f"{drift.get('current_rows', 0):,}")
    status[2].metric("Feature Drift", "Detected" if drift.get("feature_drift_detected") else "Not detected")
    status[3].metric("Prediction Drift", "Detected" if drift.get("prediction_drift_detected") else "Not detected")
    target_available = bool(drift.get("target_drift_available"))
    target_text = (
        "Detected" if drift.get("target_drift_detected") else "Not detected"
    ) if target_available else "Unavailable"
    status[4].metric("Target Drift", target_text)
    st.caption(
        f"Reference: `{drift.get('reference_data_path', 'unknown')}` · "
        f"Current: `{drift.get('current_data_path', 'unknown')}` · "
        f"Drift Threshold: {drift.get('feature_drift_share_threshold', 'unknown')}"
    )
    st.info(drift.get("interpretation", "Drift is an analytical signal for model governance and portfolio health."))

    features = drift.get("drifted_features", [])
    st.markdown('<div class="section-label">Drifted Features</div>', unsafe_allow_html=True)
    if features:
        st.dataframe(pd.DataFrame({"Feature": features}), hide_index=True, use_container_width=True)
    else:
        st.write("No feature columns were flagged with significant statistical drift.")

    st.markdown('<div class="section-label">Data Quality</div>', unsafe_allow_html=True)
    quality = drift.get("data_quality", {})
    current_quality = quality.get("current", {})
    schema = quality.get("schema_differences", {})
    quality_columns = st.columns(4)
    quality_columns[0].metric("Missing Values", current_quality.get("missing_value_count", "N/A"))
    quality_columns[1].metric("Duplicate Rows", current_quality.get("duplicate_rows", "N/A"))
    quality_columns[2].metric("Schema Issues", "Yes" if quality.get("schema_mismatch") else "No")
    quality_columns[3].metric("Invalid Values", sum(current_quality.get("invalid_numeric_values_by_column", {}).values()))
    if schema.get("missing_from_current") or schema.get("unexpected_in_current") or schema.get("dtype_mismatches"):
        st.json(schema)

    if drift.get("current_performance_available"):
        st.markdown('<div class="section-label">Current Labeled Performance</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([drift["current_performance"]]), use_container_width=True)

    st.markdown('<div class="section-label">Audit Reports</div>', unsafe_allow_html=True)
    _show_html_download(DRIFT_REPORT_PATH, "Download Feature Drift Report (HTML)", "drift-report")
    _show_html_download(QUALITY_REPORT_PATH, "Download Data Quality Report (HTML)", "quality-report")


def show_applications() -> None:
    page_header(
        "Assessment Register",
        "Applications",
        "Review completed credit risk assessments from this browser session.",
    )
    history = get_assessment_history()
    if not history:
        with st.container(border=True):
            st.markdown(
                """<div class="empty-recent-block">
                    <div class="empty-recent-title">No assessments yet.</div>
                    <div class="empty-recent-subtitle">Completed credit risk assessments will appear here.</div>
                </div>""",
                unsafe_allow_html=True,
            )
            empty_btn_left, empty_btn_mid, empty_btn_right = st.columns([1, 1, 1])
            with empty_btn_mid:
                if st.button("Start Risk Assessment", key="app-start", type="primary", use_container_width=True):
                    navigate_to("Risk Assessment")
        return

    rows = []
    for item in reversed(history):
        prediction = item["prediction"]
        band = prediction.get("risk_band", "Low")
        rows.append(
            {
                "Assessment ID": item["assessment_id"],
                "Created At": format_assessment_time(item["created_at"]),
                "Risk Classification": {"Low": "Lower Risk", "Medium": "Moderate Risk", "High": "Higher Risk"}.get(
                    band, f"{band} Risk"
                ),
                "Estimated Probability": f"{float(prediction['default_probability']):.1%}",
                "Decision Threshold": f"{float(prediction['decision_threshold']):.2f}",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown('<div class="section-label">Open Assessment Details</div>', unsafe_allow_html=True)
    select_col, btn_col = st.columns([3, 1], vertical_alignment="bottom")
    with select_col:
        selected = st.selectbox(
            "Select Assessment",
            options=[item["assessment_id"] for item in reversed(history)],
            label_visibility="collapsed",
        )
    with btn_col:
        if st.button("Review Risk Drivers", type="primary", use_container_width=True):
            st.session_state["loanrisk_active_assessment_id"] = selected
            navigate_to("Risk Drivers")


def show_system(data: dict[str, Any], api_base_url: str) -> None:
    import time

    page_header(
        "Service Health & Integrations",
        "System",
        "Local service readiness and artifact status. Prediction requests are served by FastAPI.",
    )
    started = time.perf_counter()
    try:
        response = requests.get(f"{api_base_url}/health", timeout=3)
        response.raise_for_status()
        health = response.json()
        api_ready = bool(health.get("model_loaded"))
        latency_ms = (time.perf_counter() - started) * 1000
    except (requests.RequestException, ValueError):
        health = {}
        api_ready = False
        latency_ms = None

    model = data["model"]
    integrations = [
        ("FastAPI Backend", api_ready, f"Version: {health.get('model_version', 'Unavailable')}"),
        ("Active Model", (PROJECT_ROOT / "models" / "best_model.joblib").is_file(), str(model.get("model_name", "Unavailable"))),
        ("Feature Preprocessor", (PROJECT_ROOT / "data" / "processed" / "preprocessor.joblib").is_file(), "Pipeline artifact loaded"),
        ("SHAP Explainer", (PROJECT_ROOT / "src" / "explainability" / "shap_explainer.py").is_file(), "Local tree/kernel explainer available"),
        ("MLflow Tracking", MLFLOW_SUMMARY_PATH.is_file(), "Telemetry run history available"),
        ("Evidently Monitoring", DRIFT_SUMMARY_PATH.is_file(), "Latest drift summary loaded"),
        ("Credit Dataset", (PROJECT_ROOT / "data" / "raw" / "Loan_default.csv").is_file(), "Raw dataset available"),
    ]
    for label, ready, detail in integrations:
        left, right = st.columns([1.2, 3])
        left.markdown(f"**{label}**")
        status_text = "● Ready" if ready else "○ Unavailable"
        right.write(f"{status_text} · {detail}")

    if api_ready and latency_ms is not None:
        st.metric("API Response Latency", f"{latency_ms:.1f} ms")
    st.caption(f"Backend URL: `{api_base_url}` · Service status monitored in real time.")

    with st.expander("Configure API Endpoint"):
        configured_url = st.text_input("API Base URL", value=api_base_url, key="system_api_url")
        if st.button("Save API URL", key="save-api-url"):
            st.session_state["loanrisk_api_base_url"] = configured_url.rstrip("/")
            check_api_health.clear()
            st.rerun()

    mlflow_summary = {}
    if MLFLOW_SUMMARY_PATH.is_file():
        mlflow_summary = json.loads(MLFLOW_SUMMARY_PATH.read_text(encoding="utf-8"))
    if mlflow_summary:
        st.caption(f"Latest MLflow Run ID: `{mlflow_summary.get('run_id', 'unknown')}`")


def show_about() -> None:
    page_header("Product Overview", "About LoanRisk", "")
    st.write(
        "**LoanRisk** is a credit-risk assessment platform designed to help underwriters and reviewers "
        "evaluate loan applications using transparent risk estimates and interpretable risk drivers."
    )
    st.write(
        "Each assessment provides estimated default risk and highlights the primary factors influencing "
        "the rating. The platform operates as decision-support technology to be used alongside institutional "
        "policy, compliance guidelines, and human review."
    )


def main() -> None:
    inject_visual_system()
    data = load_project_data()
    target_page = st.session_state.pop("loanrisk_navigation_target", None)
    if target_page in NAVIGATION_ITEMS:
        st.session_state["loanrisk_navigation"] = target_page
    api_base_url = st.session_state.get("loanrisk_api_base_url", API_BASE_URL).rstrip("/")

    # Sidebar
    st.sidebar.markdown(
        """<div class="brand-lockup">
            <div class="brand-mark">L</div>
            <div>
                <div class="brand-name">LoanRisk</div>
                <div class="brand-caption">Credit Risk Platform</div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("<div style='height:.25rem'></div>", unsafe_allow_html=True)
    page = st.sidebar.radio(
        "Navigation",
        NAVIGATION_ITEMS,
        key="loanrisk_navigation",
        label_visibility="collapsed",
    )
    st.sidebar.markdown(
        """<div style="border-top:1px solid #1e324d;margin-top:1.5rem;padding-top:1.1rem">
        <div style="color:#4ade80;font-size:.8rem;font-weight:700">● System Ready</div>
        <div style="color:#94a3b8;font-size:.74rem;margin-top:.25rem">Local Workspace</div></div>""",
        unsafe_allow_html=True,
    )

    if page == "Overview":
        show_dashboard(data, api_base_url)
    elif page == "Applications":
        show_applications()
    elif page == "Risk Assessment":
        show_prediction_page(data, api_base_url)
    elif page == "Risk Drivers":
        show_risk_drivers(api_base_url)
    elif page == "Analytics":
        show_model_performance(data)
    elif page == "Monitoring":
        show_drift_monitoring(data)
    elif page == "System":
        show_system(data, api_base_url)
    else:
        show_about()


if __name__ == "__main__":
    main()
