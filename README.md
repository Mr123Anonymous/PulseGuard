# GDSA Internship Showcase: Healthcare Readmission Analytics

This project is designed to mirror common Global Data Science & Analytics tasks in healthcare and medical-device adjacent analytics workflows.

## Project goals
- Clean and analyze real-world healthcare data.
- Build baseline readmission risk models.
- Provide SQL-based cohort and KPI analysis.
- Simulate simple model monitoring (drift and performance).
- Produce communication artifacts for stakeholders.

## Dataset
Use the UCI Diabetes 130-US hospitals dataset.

1. Download `diabetic_data.csv`.
2. Place it at `data/raw/diabetic_data.csv`.

## Folder structure
- `src/`: pipeline code for data prep, modeling, and monitoring.
- `sql/`: cohort analysis queries.
- `reports/`: literature summary and final report templates.
- `artifacts/`: generated metrics, models, and monitoring outputs.

## Quickstart
```bash
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python -m src.run_pipeline
streamlit run streamlit_app.py
```

## Cloud deployment
If you deploy without committing generated data and artifacts, set one or both of these sources so the dashboard can load files remotely:
- `PULSEGUARD_DATA_BASE_URL` for `data/raw/` and `data/processed/` files.
- `PULSEGUARD_ARTIFACTS_BASE_URL` for `artifacts/` files and figures.

The URLs should mirror the repo folder structure. For example, a GitHub raw base URL can point at the repository root and the app will request paths like `data/raw/diabetic_data.csv` and `artifacts/metrics.json` from that base.

## Streamlit dashboard
The dashboard implements five sections:
1. Problem and business outcome.
2. Data quality snapshot, including before/after cleaning counters.
3. EDA insights from generated charts.
4. Model comparison card with key metrics.
5. Monitoring view with trend and action guidance.

## Expected outputs
- `data/processed/model_input.csv`
- `artifacts/metrics.json`
- `artifacts/logistic_model.pkl`
- `artifacts/forest_model.pkl`
- `artifacts/monitoring_report.csv`
- `artifacts/monitoring_summary.json`
- `artifacts/eda_summary.json`
- `artifacts/figures/readmission_distribution.png`
- `artifacts/figures/readmission_by_age.png`
- `artifacts/figures/time_in_hospital_by_target.png`
- `artifacts/figures/num_lab_procedures_by_target.png`
- `artifacts/logistic_top_features.csv`
- `artifacts/forest_permutation_importance.csv`
- `artifacts/presentation_summary.json`
- `reports/executive_brief.md`

## Suggested next improvements
- Add SHAP for richer model explainability.
- Add threshold tuning based on care-management capacity.
