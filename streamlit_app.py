from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
FIGURES = ARTIFACTS / "figures"
DATA_RAW = ROOT / "data" / "raw" / "diabetic_data.csv"
DATA_PROCESSED = ROOT / "data" / "processed" / "model_input.csv"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def warn_missing(path: Path) -> None:
    st.warning(f"Missing file: {path}. Run python -m src.run_pipeline first.")


def section_1_problem() -> None:
    st.header("1. Problem and Business Outcome")
    st.markdown(
        """
This project predicts 30-day hospital readmission risk to support care-management
prioritization. The objective is to identify higher-risk patients earlier so teams
can focus interventions where they are most likely to reduce avoidable readmissions.
"""
    )


def section_2_data_quality() -> None:
    st.header("2. Data Quality Snapshot")

    raw_rows = None
    processed_rows = None
    raw_missing = None
    processed_missing = None

    if DATA_RAW.exists():
        raw_df = pd.read_csv(DATA_RAW)
        raw_rows = int(raw_df.shape[0])
        raw_missing = int(raw_df.isna().sum().sum())
    if DATA_PROCESSED.exists():
        processed_df = pd.read_csv(DATA_PROCESSED)
        processed_rows = int(processed_df.shape[0])
        processed_missing = int(processed_df.isna().sum().sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw rows", raw_rows if raw_rows is not None else "NA")
    c2.metric("Processed rows", processed_rows if processed_rows is not None else "NA")
    c3.metric("Raw missing cells", raw_missing if raw_missing is not None else "NA")
    c4.metric(
        "Processed missing cells",
        processed_missing if processed_missing is not None else "NA",
    )

    st.caption("Cleaning removes or imputes missing values and standardizes model inputs.")


def section_3_eda() -> None:
    st.header("3. EDA Insights")

    first_figures = [
        ("Readmission distribution", FIGURES / "readmission_distribution.png"),
        ("Readmission by age band", FIGURES / "readmission_by_age.png"),
        ("Time in hospital by target", FIGURES / "time_in_hospital_by_target.png"),
    ]

    for title, path in first_figures:
        st.subheader(title)
        if path.exists():
            st.image(str(path), use_container_width=True)
        else:
            warn_missing(path)

    if DATA_PROCESSED.exists():
        eda_df = pd.read_csv(DATA_PROCESSED, usecols=["target_readmit_30d", "time_in_hospital"])
        summary = (
            eda_df.groupby("target_readmit_30d")["time_in_hospital"]
            .agg(
                count="count",
                mean="mean",
                median="median",
                q1=lambda s: s.quantile(0.25),
                q3=lambda s: s.quantile(0.75),
            )
            .reset_index()
        )
        summary["iqr"] = summary["q3"] - summary["q1"]
        summary = summary.rename(columns={"target_readmit_30d": "readmit_30d"})

        st.subheader("Time in Hospital Summary by Readmission Class")
        st.dataframe(summary, use_container_width=True)

    st.subheader("Lab procedures by target")
    violin_path = FIGURES / "num_lab_procedures_by_target.png"
    if violin_path.exists():
        st.image(str(violin_path), use_container_width=True)
    else:
        warn_missing(violin_path)


def section_4_model_comparison() -> None:
    st.header("4. Model Comparison")

    metrics = load_json(ARTIFACTS / "metrics.json")
    if not metrics:
        warn_missing(ARTIFACTS / "metrics.json")
        return

    comparison = pd.DataFrame(
        [
            {
                "model": "Logistic Regression",
                "roc_auc": metrics.get("logistic_roc_auc"),
                "pr_auc": metrics.get("logistic_pr_auc"),
            },
            {
                "model": "Random Forest",
                "roc_auc": metrics.get("forest_roc_auc"),
                "pr_auc": metrics.get("forest_pr_auc"),
            },
        ]
    )

    st.dataframe(comparison, use_container_width=True)
    st.info(
        "Random Forest is selected as stronger on current discrimination, while Logistic "
        "Regression remains useful for coefficient-level interpretation."
    )


def section_5_monitoring() -> None:
    st.header("5. Monitoring View")

    summary_path = ARTIFACTS / "monitoring_summary.json"
    report_path = ARTIFACTS / "monitoring_report.csv"

    summary = load_json(summary_path)
    if summary:
        c1, c2, c3 = st.columns(3)
        c1.metric("Drift alerts", summary.get("drift_alerts", "NA"))
        c2.metric("Min monthly ROC-AUC", summary.get("min_monthly_roc_auc", "NA"))
        c3.metric("Max monthly ROC-AUC", summary.get("max_monthly_roc_auc", "NA"))
    else:
        warn_missing(summary_path)

    if report_path.exists():
        report = pd.read_csv(report_path)
        st.dataframe(report, use_container_width=True)

        if {"month", "roc_auc"}.issubset(report.columns):
            st.line_chart(report.set_index("month")["roc_auc"])
    else:
        warn_missing(report_path)

    alerts = summary.get("drift_alerts") if summary else None
    if alerts is not None and alerts > 0:
        st.error("Action: investigate drifted segments and schedule model recalibration.")
    else:
        st.success("Action: continue monitoring on current cadence.")


def main() -> None:
    st.set_page_config(page_title="Healthcare Readmission Dashboard", layout="wide")
    st.title("Healthcare Readmission Analytics")
    st.caption("GDSA internship-style showcase")

    tab_problem, tab_quality, tab_eda, tab_models, tab_monitor = st.tabs(
        [
            "1. Problem",
            "2. Data Quality",
            "3. EDA",
            "4. Models",
            "5. Monitoring",
        ]
    )

    with tab_problem:
        section_1_problem()

    with tab_quality:
        section_2_data_quality()

    with tab_eda:
        section_3_eda()

    with tab_models:
        section_4_model_comparison()

    with tab_monitor:
        section_5_monitoring()


if __name__ == "__main__":
    main()
