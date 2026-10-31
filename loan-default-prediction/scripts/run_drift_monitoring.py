"""Run a local Evidently comparison of training reference and current data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitoring.config import CURRENT_DATA_PATH, REFERENCE_DATA_PATH  # noqa: E402
from monitoring.evidently_monitor import run_monitoring  # noqa: E402


def main() -> int:
    summary = run_monitoring()
    print("Loan Default Drift Monitoring")
    print("-----------------------------")
    print(f"Reference rows: {summary['reference_rows']:,}")
    print(f"Current rows: {summary['current_rows']:,}")
    print(f"Reference: {REFERENCE_DATA_PATH}")
    print(f"Current: {CURRENT_DATA_PATH}")
    print(f"Feature drift detected: {'Yes' if summary['feature_drift_detected'] else 'No'}")
    print(f"Prediction drift detected: {'Yes' if summary['prediction_drift_detected'] else 'No'}")
    target_state = (
        "Yes" if summary["target_drift_detected"] else "No"
    ) if summary["target_drift_available"] else "Unavailable (current labels absent)"
    print(f"Target drift detected: {target_state}")
    print("Drifted features:")
    if summary["drifted_features"]:
        for feature in summary["drifted_features"]:
            print(f"- {feature}")
    else:
        print("- None")
    print("Reports:")
    print("reports/monitoring/drift_report.html")
    print("reports/monitoring/data_quality_report.html")
    print("reports/monitoring/drift_summary.json")
    if summary["target_drift_available"]:
        print("reports/monitoring/performance_summary.json")
    print("Drift is a monitoring signal; it does not prove model performance degradation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())