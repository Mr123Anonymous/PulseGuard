from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from urllib.request import urlopen

import pandas as pd

from .config import DATA_PROCESSED, DATA_RAW


UCI_DATASET_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)


@dataclass
class DataPrepResult:
    rows_raw: int
    rows_final: int
    positive_rate: float


def download_raw_dataset() -> None:
    DATA_RAW.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(UCI_DATASET_ZIP_URL) as response:
        zip_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        with archive.open("diabetic_data.csv") as source, DATA_RAW.open("wb") as target:
            target.write(source.read())


def _binary_target(readmitted: pd.Series) -> pd.Series:
    return (readmitted == "<30").astype(int)


def load_and_prepare_data() -> pd.DataFrame:
    if not DATA_RAW.exists():
        download_raw_dataset()

    df = pd.read_csv(DATA_RAW)
    df = df.replace("?", pd.NA)

    if "readmitted" not in df.columns:
        raise ValueError("Expected 'readmitted' column in raw data.")

    df["target_readmit_30d"] = _binary_target(df["readmitted"]).astype(int)

    # Remove leakage-like columns and IDs not useful for baseline modeling.
    drop_cols = [
        "encounter_id",
        "patient_nbr",
        "readmitted",
        "weight",
        "payer_code",
        "medical_specialty",
    ]
    existing_drop = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing_drop)

    # Keep only rows with core fields required for analysis.
    required = ["race", "gender", "age", "time_in_hospital", "num_lab_procedures"]
    existing_required = [c for c in required if c in df.columns]
    df = df.dropna(subset=existing_required)

    # Fill categorical and numeric columns separately to preserve feature types.
    categorical_cols = df.select_dtypes(include=["object", "string", "category"]).columns
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns

    if len(categorical_cols) > 0:
        df[categorical_cols] = df[categorical_cols].fillna("Unknown")

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    DATA_PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PROCESSED, index=False)
    return df


def run_data_prep() -> DataPrepResult:
    if not DATA_RAW.exists():
        download_raw_dataset()

    raw_rows = pd.read_csv(DATA_RAW).shape[0]
    prepared = load_and_prepare_data()
    return DataPrepResult(
        rows_raw=raw_rows,
        rows_final=prepared.shape[0],
        positive_rate=float(prepared["target_readmit_30d"].mean()),
    )


if __name__ == "__main__":
    result = run_data_prep()
    print(result)
