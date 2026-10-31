"""Small, reproducible training-only randomized searches."""

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from src.models.train import RANDOM_STATE

TUNING_SAMPLE_SIZE = 20_000
CV_SPLITS = 3


def _search_space(model_name: str) -> tuple[object, dict[str, list[object]], int]:
    if model_name == "Logistic Regression":
        estimator = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        parameters = {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "class_weight": [None, "balanced"],
            "solver": ["liblinear", "lbfgs"],
        }
        return estimator, parameters, 8
    if model_name == "Random Forest":
        estimator = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)
        parameters = {
            "n_estimators": [200, 300, 400],
            "max_depth": [None, 5, 10, 15],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
            "class_weight": [None, "balanced"],
        }
        return estimator, parameters, 5
    if model_name == "XGBoost":
        estimator = XGBClassifier(
            random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=1
        )
        parameters = {
            "n_estimators": [200, 300, 500],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [3, 4, 5, 6],
            "min_child_weight": [1, 3, 5],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
        }
        return estimator, parameters, 5
    raise ValueError(f"Unsupported model for tuning: {model_name!r}")


def tune_model(model_name: str, features, target) -> tuple[object, dict[str, object], float]:
    """Tune on a stratified training-only subset, then refit on all training rows."""
    estimator, parameters, n_iter = _search_space(model_name)
    if len(features) > TUNING_SAMPLE_SIZE:
        sampled_features, _, sampled_target, _ = train_test_split(
            features,
            target,
            train_size=TUNING_SAMPLE_SIZE,
            stratify=target,
            random_state=RANDOM_STATE,
        )
    else:
        sampled_features, sampled_target = features, target

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=parameters,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        pre_dispatch="2*n_jobs",
        refit=False,
        error_score="raise",
    )
    search.fit(sampled_features, sampled_target)
    best_model = clone(estimator).set_params(**search.best_params_)
    if hasattr(best_model, "n_jobs"):
        best_model.set_params(n_jobs=-1)
    best_model.fit(features, target)
    return best_model, search.best_params_, float(search.best_score_)