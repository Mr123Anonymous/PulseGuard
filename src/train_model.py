from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ARTIFACTS_DIR, DATA_PROCESSED


@dataclass
class ModelResult:
    logistic_roc_auc: float
    logistic_pr_auc: float
    forest_roc_auc: float
    forest_pr_auc: float


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ]
    )


def train_and_evaluate() -> ModelResult:
    if not DATA_PROCESSED.exists():
        raise FileNotFoundError(
            f"Missing processed data at {DATA_PROCESSED}. Run data prep first."
        )

    df = pd.read_csv(DATA_PROCESSED)
    y = pd.to_numeric(df["target_readmit_30d"], errors="coerce")
    X = df.drop(columns=["target_readmit_30d"])

    valid_mask = y.notna()
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].astype(int).reset_index(drop=True)

    if y.nunique() < 2:
        raise ValueError("Target must contain at least two classes after cleaning.")

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

    prob_log = logistic.predict_proba(X_test)[:, 1]
    prob_for = forest.predict_proba(X_test)[:, 1]

    result = ModelResult(
        logistic_roc_auc=float(roc_auc_score(y_test, prob_log)),
        logistic_pr_auc=float(average_precision_score(y_test, prob_log)),
        forest_roc_auc=float(roc_auc_score(y_test, prob_for)),
        forest_pr_auc=float(average_precision_score(y_test, prob_for)),
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(logistic, ARTIFACTS_DIR / "logistic_model.pkl")
    joblib.dump(forest, ARTIFACTS_DIR / "forest_model.pkl")
    with (ARTIFACTS_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result.__dict__, f, indent=2)

    return result


if __name__ == "__main__":
    print(train_and_evaluate())
