from __future__ import annotations

from src.data_prep import run_data_prep
from src.eda import run_eda
from src.explainability import run_explainability
from src.monitoring import run_monitoring_simulation
from src.presentation_summary import build_presentation_summary
from src.train_model import train_and_evaluate


if __name__ == "__main__":
    prep = run_data_prep()
    print("Data prep complete:", prep)

    eda = run_eda()
    print("EDA complete:", eda)

    model = train_and_evaluate()
    print("Model training complete:", model)

    explainability = run_explainability()
    print("Explainability complete:", explainability)

    monitoring = run_monitoring_simulation()
    print("Monitoring simulation complete:", monitoring)

    summary = build_presentation_summary()
    print("Presentation summary complete:", summary)
