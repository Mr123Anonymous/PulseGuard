from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import ARTIFACTS_DIR, DATA_PROCESSED


@dataclass
class MonitoringSummary:
    drift_alerts: int
    min_monthly_roc_auc: float
    max_monthly_roc_auc: float


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected_perc, bin_edges = np.histogram(expected, bins=bins)
    actual_perc, _ = np.histogram(actual, bins=bin_edges)

    expected_perc = expected_perc / max(expected_perc.sum(), 1)
    actual_perc = actual_perc / max(actual_perc.sum(), 1)

    expected_perc = np.where(expected_perc == 0, 1e-6, expected_perc)
    actual_perc = np.where(actual_perc == 0, 1e-6, actual_perc)

    return float(np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc)))


def run_monitoring_simulation() -> MonitoringSummary:
    if not DATA_PROCESSED.exists():
        raise FileNotFoundError("Processed data missing. Run data prep first.")

    model_path = ARTIFACTS_DIR / "logistic_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError("Model artifact missing. Run training first.")

    model = joblib.load(model_path)
    df = pd.read_csv(DATA_PROCESSED)

    y = df["target_readmit_30d"]
    X = df.drop(columns=["target_readmit_30d"])

    rng = np.random.default_rng(42)
    month_id = rng.integers(1, 7, size=len(df))
    monitoring_rows = []

    baseline_mask = month_id == 1
    baseline_feature = X["time_in_hospital"].to_numpy(dtype=float)[baseline_mask]

    for month in range(1, 7):
        mask = month_id == month
        X_m = X[mask]
        y_m = y[mask]
        if len(X_m) < 20:
            continue

        p = model.predict_proba(X_m)[:, 1]
        roc = float(roc_auc_score(y_m, p))
        pr = float(average_precision_score(y_m, p))
        psi_value = _psi(
            baseline_feature,
            X_m["time_in_hospital"].to_numpy(dtype=float),
        )
        drift_flag = int(psi_value > 0.2)

        monitoring_rows.append(
            {
                "month": month,
                "rows": int(len(X_m)),
                "roc_auc": roc,
                "pr_auc": pr,
                "psi_time_in_hospital": psi_value,
                "drift_alert": drift_flag,
            }
        )

    out = pd.DataFrame(monitoring_rows)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(ARTIFACTS_DIR / "monitoring_report.csv", index=False)

    summary = MonitoringSummary(
        drift_alerts=int(out["drift_alert"].sum()) if not out.empty else 0,
        min_monthly_roc_auc=float(out["roc_auc"].min()) if not out.empty else 0.0,
        max_monthly_roc_auc=float(out["roc_auc"].max()) if not out.empty else 0.0,
    )

    with (ARTIFACTS_DIR / "monitoring_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary.__dict__, f, indent=2)

    return summary


if __name__ == "__main__":
    print(run_monitoring_simulation())
