from __future__ import annotations

import json
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import ARTIFACTS_DIR, DATA_PROCESSED, FIGURES_DIR


@dataclass
class EDAResult:
    rows: int
    columns: int
    target_rate: float


def _save_plot(name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / name, dpi=140)
    plt.close()


def run_eda() -> EDAResult:
    if not DATA_PROCESSED.exists():
        raise FileNotFoundError(
            f"Missing processed data at {DATA_PROCESSED}. Run data prep first."
        )

    df = pd.read_csv(DATA_PROCESSED)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(7, 4))
    sns.countplot(data=df, x="target_readmit_30d")
    plt.title("Readmission Target Distribution")
    plt.xlabel("Readmitted within 30 days")
    plt.ylabel("Count")
    _save_plot("readmission_distribution.png")

    if "age" in df.columns:
        plt.figure(figsize=(10, 4))
        age_rate = (
            df.groupby("age")["target_readmit_30d"]
            .mean()
            .reset_index(name="target_readmit_30d")
        )
        sns.barplot(data=age_rate, x="age", y="target_readmit_30d", color="#2a9d8f")
        plt.xticks(rotation=45, ha="right")
        plt.title("Readmission Rate by Age Band")
        plt.xlabel("Age Band")
        plt.ylabel("Readmission rate")
        _save_plot("readmission_by_age.png")

    if "time_in_hospital" in df.columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df, x="target_readmit_30d", y="time_in_hospital", color="#f4a261")
        plt.title("Time in Hospital by Readmission Target")
        plt.xlabel("Readmitted within 30 days")
        plt.ylabel("Days in hospital")
        _save_plot("time_in_hospital_by_target.png")

    if "num_lab_procedures" in df.columns:
        plt.figure(figsize=(8, 4))
        sns.violinplot(
            data=df,
            x="target_readmit_30d",
            y="num_lab_procedures",
            inner="quartile",
            color="#e76f51",
        )
        plt.title("Lab Procedures by Readmission Target")
        plt.xlabel("Readmitted within 30 days")
        plt.ylabel("Number of lab procedures")
        _save_plot("num_lab_procedures_by_target.png")

    result = EDAResult(
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        target_rate=float(df["target_readmit_30d"].mean()),
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with (ARTIFACTS_DIR / "eda_summary.json").open("w", encoding="utf-8") as f:
        json.dump(result.__dict__, f, indent=2)

    return result


if __name__ == "__main__":
    print(run_eda())
