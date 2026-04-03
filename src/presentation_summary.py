from __future__ import annotations

import json
from pathlib import Path

from .config import ARTIFACTS_DIR, REPORTS_DIR


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_presentation_summary() -> dict:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = _load_json(ARTIFACTS_DIR / "metrics.json")
    monitoring = _load_json(ARTIFACTS_DIR / "monitoring_summary.json")
    eda = _load_json(ARTIFACTS_DIR / "eda_summary.json")

    summary = {
        "project": "Healthcare Readmission Analytics",
        "key_metrics": {
            "logistic_roc_auc": metrics.get("logistic_roc_auc"),
            "forest_roc_auc": metrics.get("forest_roc_auc"),
            "forest_pr_auc": metrics.get("forest_pr_auc"),
        },
        "data_profile": {
            "rows": eda.get("rows"),
            "columns": eda.get("columns"),
            "target_rate": eda.get("target_rate"),
        },
        "monitoring": monitoring,
        "talk_track": [
            "Built an end-to-end healthcare analytics workflow from cleaning to monitoring.",
            "Compared interpretable and non-linear baselines to assess readmission risk.",
            "Added drift checks to demonstrate model lifecycle awareness for production settings.",
        ],
    }

    with (ARTIFACTS_DIR / "presentation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    brief = [
        "# Executive Brief",
        "",
        "## Problem",
        "Estimate 30-day readmission risk and identify high-impact factors.",
        "",
        "## Data snapshot",
        f"- Rows: {summary['data_profile']['rows']}",
        f"- Columns: {summary['data_profile']['columns']}",
        f"- Positive target rate: {summary['data_profile']['target_rate']}",
        "",
        "## Model performance",
        f"- Logistic ROC-AUC: {summary['key_metrics']['logistic_roc_auc']}",
        f"- Forest ROC-AUC: {summary['key_metrics']['forest_roc_auc']}",
        f"- Forest PR-AUC: {summary['key_metrics']['forest_pr_auc']}",
        "",
        "## Monitoring",
        f"- Drift alerts: {monitoring.get('drift_alerts')}",
        f"- Min monthly ROC-AUC: {monitoring.get('min_monthly_roc_auc')}",
        f"- Max monthly ROC-AUC: {monitoring.get('max_monthly_roc_auc')}",
        "",
        "## Interview narrative",
        "- Built a reproducible analytics pipeline in Python with healthcare-oriented KPIs.",
        "- Combined cohort SQL, model baselines, explainability, and monitoring outputs.",
        "- Structured outputs for technical and non-technical communication.",
    ]

    (REPORTS_DIR / "executive_brief.md").write_text("\n".join(brief), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(build_presentation_summary())
