"""Generate sampled global and local SHAP artifacts for the final model."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.optimize_model import load_original_split  # noqa: E402
from src.explainability.shap_explainer import LoanSHAPExplainer  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "reports"
FIGURE_PATH = REPORT_PATH / "figures"
SAMPLE_SIZE = 5_000


def main() -> int:
    print("Loading final model and metadata...")
    explainer = LoanSHAPExplainer()
    validation = load_original_split("validation").drop(columns="Default")
    sample = validation.sample(
        n=min(SAMPLE_SIZE, len(validation)), random_state=42
    ).reset_index(drop=True)
    background = sample.sample(n=min(explainer.background_size, len(sample)), random_state=42)

    print(f"Creating {explainer.explainer_type} for {explainer.model_type}...")
    explainer.create_explainer(background)
    print(f"Calculating SHAP values for {len(sample):,} validation rows...")
    shap_values = explainer.calculate_shap_values(sample, background_data=background)
    importance = explainer.get_global_importance(shap_values)

    REPORT_PATH.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.mkdir(parents=True, exist_ok=True)
    importance.to_csv(REPORT_PATH / "shap_feature_importance.csv", index=False)

    import shap

    shap.plots.bar(shap_values, max_display=20, show=False)
    figure = plt.gcf()
    figure.set_size_inches(9, 8)
    plt.title("SHAP Global Feature Importance")
    figure.tight_layout()
    figure.savefig(FIGURE_PATH / "shap_summary_bar.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    figure = plt.gcf()
    figure.set_size_inches(10, 8)
    plt.title("SHAP Feature Impact on Loan Default Predictions")
    figure.tight_layout()
    figure.savefig(FIGURE_PATH / "shap_summary_beeswarm.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    local_loan = sample.iloc[[0]]
    local_explanation = explainer.calculate_shap_values(
        local_loan, background_data=background
    )
    local_detail = explainer.explain_single_prediction(local_loan)
    local_result = explainer.explain_prediction(local_loan)
    shap.plots.waterfall(local_explanation[0], max_display=15, show=False)
    figure = plt.gcf()
    figure.set_size_inches(10, 7)
    plt.title("Local SHAP Explanation - Validation Loan")
    figure.tight_layout()
    figure.savefig(FIGURE_PATH / "shap_local_example.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    (REPORT_PATH / "shap_local_example.json").write_text(
        json.dumps(local_result, indent=2), encoding="utf-8"
    )

    top_ten = importance.head(10)
    top_ten_markdown = "\n".join(
        [
            "| feature | mean_abs_shap | rank |",
            "| --- | ---: | ---: |",
            *[
                f"| {row.feature} | {row.mean_abs_shap:.6f} | {row.rank} |"
                for row in top_ten.itertuples(index=False)
            ],
        ]
    )
    report_lines = [
        "# Phase 5 SHAP Explainability Report",
        "",
        "## 1. Objective",
        "SHAP was added to describe how the selected model uses transformed input features globally and for one validation loan. Explanations do not change or select the model.",
        "",
        "## 2. Model explained",
        f"Phase 4 selected **{explainer.model_type}**. The saved `models/best_model.joblib` remains the prediction model; no retraining or parameter changes were made.",
        "",
        "## 3. Method",
        f"The explainer is **{explainer.explainer_type}**. It explains {explainer.output_units}; SHAP values are not probability-point changes. The existing Phase 4 pipeline and fitted Phase 2 preprocessing were reused. A deterministic sample of {len(sample):,} validation records was explained, with a {len(background):,}-row validation background. The target class is Default=1.",
        "",
        "## 4. Global feature importance",
        "The table ranks mean absolute SHAP contributions over the validation sample. Higher mean absolute SHAP values indicate that a feature has a greater influence on the model's predictions within the analyzed sample. One-hot columns remain separate. These values are not causal effects.",
        "",
        top_ten_markdown,
        "",
        "## 5. Local explanation",
        "For an individual loan, signed SHAP values identify transformed features that moved the model output higher or lower relative to the explainer's baseline. The reusable method returns JSON-safe probability, threshold, class, and top-factor details.",
        "",
        "## 6. Example interpretation",
        f"The sampled validation loan has predicted default probability {local_result['default_probability']:.4f} and predicted class {local_result['predicted_class']} at threshold {local_result['threshold']:.2f}. Top model contributions are listed below; each describes contribution to the model output, not a cause of default.",
        "",
    ]
    for factor in local_detail["top_factors"][:5]:
        report_lines.append(f"- {factor['explanation']}")
    report_lines.extend(
        [
            "",
            "## 7. Limitations",
            "- SHAP explains model behavior, not real-world causality.",
            "- One-hot encoded variables appear as separate transformed features.",
            "- Explanations depend on the trained model and sampled validation/background distribution.",
            "- A SHAP contribution does not prove that changing a feature would change a real-world default outcome.",
            "- Linear model SHAP values here are log-odds contributions, not probability changes.",
            "",
            "## 8. Phase 5 conclusion",
            "The final model now has a reusable `LoanSHAPExplainer.explain_prediction()` method suitable for later API integration. SHAP was used only for explanation.",
            "",
        ]
    )
    (REPORT_PATH / "phase5_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print("\nSHAP explainability completed.")
    print(f"Model: {explainer.model_type}")
    print(f"Validation samples explained: {len(sample):,}")
    print("Top features:")
    for rank, feature in enumerate(importance["feature"].head(5), start=1):
        print(f"{rank}. {feature}")
    print("Outputs:")
    for output in (
        "reports/shap_feature_importance.csv",
        "reports/figures/shap_summary_bar.png",
        "reports/figures/shap_summary_beeswarm.png",
        "reports/figures/shap_local_example.png",
        "reports/phase5_report.md",
    ):
        print(f"- {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())