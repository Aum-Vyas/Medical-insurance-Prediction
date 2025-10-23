Insurance Cost Prediction, Pricing, and Analytics

Overview

- Predict medical insurance charges from tabular features (age, sex, bmi, children, smoker, region).
- Train multiple models, compare performance, and export the best full pipeline (preprocessing + model).
- Explain model behavior (SHAP, PDP), cluster customers, simulate scenarios, assess fairness, and serve an API.
- Interactive Streamlit dashboard for metrics, clusters, pricing, scenarios, and fairness.

Data Requirements

- CSV with columns (case-insensitive): `age, sex, bmi, children, smoker, region, charges`.
- Categorical: `sex, smoker, region`. Numeric: `age, bmi, children`. Target: `charges`.
- Column names are normalized to lowercase automatically.

Installation

- Create an environment and install dependencies:
  - Windows (PowerShell):
    - `python -m venv .venv`
    - `.\.venv\Scripts\Activate.ps1`
    - `pip install -r requirements.txt`
  - macOS/Linux (bash):
    - `python3 -m venv .venv && source .venv/bin/activate`
    - `pip install -r requirements.txt`
  - Conda alternative:
    - `conda create -n insurance python=3.11 -y && conda activate insurance`
    - `pip install -r requirements.txt`

Optional/Heavy Dependencies

- XGBoost: required for the `xgb` model (use Conda if pip wheel fails).
- PyTorch: required for the `nn` model (PyTorch MLP).
- SHAP: required for SHAP plots (used only for tree models).

Quick Start (End-to-End)

1) Place your CSV in this folder (or note its path).
2) Run the end-to-end pipeline (training, explainability, clustering, pricing, scenarios, fairness):

   `python -m insurance.run_all --data your_file.csv --outdir artifacts --margin 0.2 --clusters 4`

3) Review outputs in `artifacts/` (see Outputs below).

Models Trained and Compared

- Linear regression
- Ridge regression
- Random forest (`rf`)
- XGBoost (`xgb`, optional)
- Neural network (`nn`, PyTorch MLP, optional)

Single-Model CLI (Optional)

- Train a specific model:
  - `python -m insurance.train --data your_file.csv --target charges --model rf --outdir artifacts`

- Predict on new records:
  - `python -m insurance.predict --model artifacts/model_best.joblib --csv new_records.csv --out predictions.csv --price-margin 0.15`

API (FastAPI)

- Launch: `uvicorn insurance.api:app --reload --port 8000`
- Health: `GET http://localhost:8000/health`
- Predict: `POST http://localhost:8000/predict?include_scenarios=true`
  - Body:
    - `{ "age": 45, "sex": "male", "bmi": 28.3, "children": 2, "smoker": "no", "region": "southeast", "margin": 0.2 }`
  - Returns: predicted cost, premium (with margin), and cluster; optional scenario results if `include_scenarios=true`.

Dashboard (Streamlit)

- Launch the interactive dashboard after pipeline run:
  - `streamlit run insurance/dashboard.py`

- Sections:
  - Overview: model metrics table (RMSE/MAE/R²) and bar charts.
  - Explainability: SHAP summary (if available) and PDPs for `age, bmi, children`.
  - Clustering: cluster counts and average charges/premiums.
  - Pricing Explorer: per-record predicted cost and premium with filters (region, sex, smoker, cluster).
  - Scenario Simulator: what-if (BMI change, smoking cessation, children +1) with premium deltas; adjustable pricing margin.
  - Fairness: residuals/MAE by sex and region with charts and tables.

Scenario & Sensitivity Analysis

- Programmatic: see `insurance/scenario.py` (`evaluate_scenarios`).
- Dashboard: use Scenario Simulator to tweak BMI, smoker status, and children.
- CLI: results saved to `artifacts/scenario_results.csv` from `insurance.run_all`.

Fairness / Bias Analysis

- Residuals and MAE by demographic groups (sex, region).
- Files: `artifacts/fairness_sex.csv`, `artifacts/fairness_region.csv`, `artifacts/fairness_report.json`.
- Use these to identify systematic error gaps; consider reweighting/mitigation if gaps are material.

Outputs (Artifacts)

- Models: `model_best.joblib`, `model_*.joblib`, `model_metrics.csv`, `summary.json`
- Explainability: `explain/shap_summary.png` (tree models), `explain/pdp.png`
- Clustering: `kmeans.joblib`, `cluster_labels.csv`, `cluster_summary.csv`
- Pricing: `pricing.csv` (features + predicted_cost + premium + cluster)
- Scenarios: `scenario_results.csv`
- Fairness: `fairness_sex.csv`, `fairness_region.csv`, `fairness_report.json`
- Run status: `run_complete.json`

Project Structure

- `insurance/data.py` — CSV loading and column normalization
- `insurance/preprocess.py` — preprocessing pipeline (scaler + OHE)
- `insurance/eval.py` — RMSE, MAE, R²
- `insurance/models.py` — multi‑model training and comparison
- `insurance/train.py` — single‑model training CLI
- `insurance/predict.py` — batch predict CLI with margin pricing
- `insurance/explain.py` — SHAP summary and PDP plots
- `insurance/cluster.py` — KMeans on preprocessed features and summaries
- `insurance/scenario.py` — scenario generation and evaluation
- `insurance/fairness.py` — residuals/MAE by group
- `insurance/run_all.py` — end‑to‑end orchestration
- `insurance/api.py` — FastAPI app (`/predict`, `/health`)
- `insurance/dashboard.py` — Streamlit dashboard
- `.streamlit/config.toml` — dashboard theme

Troubleshooting

- Package install issues (Windows/WSL): prefer Windows native Python + PowerShell venv.
- XGBoost/PyTorch wheels: if pip fails, use Conda (`conda install -c conda-forge xgboost pytorch cpuonly` as appropriate), or skip models `xgb`/`nn`.
- SHAP errors: only supported for tree models here; PDPs work for all models.
- Column mismatches: ensure required columns exist; names are case-insensitive.
- Memory: OneHotEncoder uses dense output; dataset is small. If needed, set `sparse=True` and adapt scaler.

Licensing

- No license header is included. Add a project license if required.
