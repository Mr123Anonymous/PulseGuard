# GDSA Internship Showcase: Healthcare Readmission Analytics

This project is an internship-style implementation designed to mirror common Global Data Science & Analytics tasks in healthcare and medical-device adjacent analytics workflows.

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
- `sql/`: interview-ready cohort analysis queries.
- `reports/`: literature summary and final report templates.
- `artifacts/`: generated metrics, models, and monitoring outputs.

## Quickstart
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.run_pipeline
streamlit run streamlit_app.py
```

## Streamlit dashboard
The dashboard implements five interview-focused sections:
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

## Implemented next steps
- Added automated EDA charts and summary generation for quick storytelling.
- Added explainability outputs (logistic top coefficients and forest permutation importance).
- Added a presentation-ready executive brief and consolidated summary JSON.

## Suggested next improvements
- Add SHAP for richer model explainability.
- Add threshold tuning based on care-management capacity.
