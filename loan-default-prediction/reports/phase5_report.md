# Phase 5 SHAP Explainability Report

## 1. Objective
SHAP was added to describe how the selected model uses transformed input features globally and for one validation loan. Explanations do not change or select the model.

## 2. Model explained
Phase 4 selected **Logistic Regression**. The saved `models/best_model.joblib` remains the prediction model; no retraining or parameter changes were made.

## 3. Method
The explainer is **LinearExplainer**. It explains log-odds for Default=1; SHAP values are not probability-point changes. The existing Phase 4 pipeline and fitted Phase 2 preprocessing were reused. A deterministic sample of 5,000 validation records was explained, with a 100-row validation background. The target class is Default=1.

## 4. Global feature importance
The table ranks mean absolute SHAP contributions over the validation sample. Higher mean absolute SHAP values indicate that a feature has a greater influence on the model's predictions within the analyzed sample. One-hot columns remain separate. These values are not causal effects.

| feature | mean_abs_shap | rank |
| --- | ---: | ---: |
| Age | 0.507031 | 1 |
| InterestRate | 0.399051 | 2 |
| MonthsEmployed | 0.309363 | 3 |
| LoanIncomeRatio | 0.293204 | 4 |
| CreditScore | 0.105550 | 5 |
| NumCreditLines | 0.093095 | 6 |
| EmploymentType_Full-time | 0.090812 | 7 |
| EmploymentType_Unemployed | 0.072615 | 8 |
| HasCoSigner_Yes | 0.065635 | 9 |
| HasCoSigner_No | 0.064375 | 10 |

## 5. Local explanation
For an individual loan, signed SHAP values identify transformed features that moved the model output higher or lower relative to the explainer's baseline. The reusable method returns JSON-safe probability, threshold, class, and top-factor details.

## 6. Example interpretation
The sampled validation loan has predicted default probability 0.0828 and predicted class 0 at threshold 0.15. Top model contributions are listed below; each describes contribution to the model output, not a cause of default.

- Age (input=60) contributed to the model's lower predicted default risk (-0.6712 log-odds for Default=1).
- LoanIncomeRatio contributed to the model's higher predicted default risk (+0.4815 log-odds for Default=1).
- InterestRate (input=7.83) contributed to the model's lower predicted default risk (-0.3808 log-odds for Default=1).
- MonthsEmployed (input=35) contributed to the model's higher predicted default risk (+0.2244 log-odds for Default=1).
- EmploymentType=Unemployed one-hot indicator (input category=Unemployed) contributed to the model's higher predicted default risk (+0.1559 log-odds for Default=1).

## 7. Limitations
- SHAP explains model behavior, not real-world causality.
- One-hot encoded variables appear as separate transformed features.
- Explanations depend on the trained model and sampled validation/background distribution.
- A SHAP contribution does not prove that changing a feature would change a real-world default outcome.
- Linear model SHAP values here are log-odds contributions, not probability changes.

## 8. Phase 5 conclusion
The final model now has a reusable `LoanSHAPExplainer.explain_prediction()` method suitable for later API integration. SHAP was used only for explanation.
