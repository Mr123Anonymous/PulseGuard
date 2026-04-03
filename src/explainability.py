from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import ARTIFACTS_DIR, DATA_PROCESSED
from .train_model import _build_preprocessor


@dataclass
class ExplainabilityResult:
    logistic_feature_rows: int
    forest_feature_rows: int


def run_explainability() -> ExplainabilityResult:
    if not DATA_PROCESSED.exists():
        raise FileNotFoundError(
            f"Missing processed data at {DATA_PROCESSED}. Run data prep first."
        )

    df = pd.read_csv(DATA_PROCESSED)
    y = df["target_readmit_30d"]
    X = df.drop(columns=["target_readmit_30d"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = _build_preprocessor(X)

    logistic = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    forest = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    min_samples_leaf=4,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )

    logistic.fit(X_train, y_train)
    forest.fit(X_train, y_train)

    feature_names = logistic.named_steps["preprocessor"].get_feature_names_out()
    coefficients = logistic.named_steps["model"].coef_[0]

    logistic_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "abs_coefficient": abs(coefficients),
        }
    ).sort_values("abs_coefficient", ascending=False)

    # Bound runtime for portfolio/demo usage.
    sample_n = min(3000, len(X_test))
    X_perm = X_test.sample(n=sample_n, random_state=42)
    y_perm = y_test.loc[X_perm.index]

    perm = cast(
        Any,
        permutation_importance(
            forest,
            X_perm,
            y_perm,
            n_repeats=2,
            random_state=42,
            scoring="roc_auc",
        ),
    )

    forest_importance = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    logistic_importance.to_csv(ARTIFACTS_DIR / "logistic_top_features.csv", index=False)
    forest_importance.to_csv(
        ARTIFACTS_DIR / "forest_permutation_importance.csv", index=False
    )

    return ExplainabilityResult(
        logistic_feature_rows=int(logistic_importance.shape[0]),
        forest_feature_rows=int(forest_importance.shape[0]),
    )


if __name__ == "__main__":
    print(run_explainability())
