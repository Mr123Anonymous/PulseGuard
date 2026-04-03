from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "diabetic_data.csv"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed" / "model_input.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
