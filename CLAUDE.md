# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
**Today**: Telecom churn prediction model (HistGradientBoosting on 15 columns of a 100,000-row Kaggle dataset) + FastAPI service + Streamlit app with 6 pages + SQLite database. The CSV is loaded into one `customers` table. Model artifacts (.pkl) saved in models/. Resume and portfolio project; the dataset is CC BY-NC-SA 4.0, so it must stay non-commercial.

**Built**: SHAP explainability, FastAPI microservice, Docker containerization, GitHub Actions CI, MLflow experiment tracking, Evidently AI drift monitoring. **Planned**: public deployment, CSV upload and scoring, LLM retention recommendations (Claude API).

## Folder Structure & Purpose
- `app.py` — Streamlit multipage app entry point (navigation shell)
- `pages/` — Streamlit pages: 1_Dashboard.py, 2_Predict_Churn.py, 3_Insights.py, 4_Model_Info.py, 5_About.py, 6_Monitoring.py
- `src/` — Reusable logic:
  - `preprocess.py` (data loading, stratified split, ColumnTransformer feature engineering)
  - `schema.py` (the 15 model columns: limits, labels, meanings; shared by form, API and training)
  - `train.py` (HistGradientBoosting training, evaluation metrics, threshold calculation, MLflow tracking)
  - `experiments.py` (comparative benchmark experiments with 5-fold CV)
  - `predict.py` (cached artifact loader, predict_churn inference)
  - `data_ingest.py` (loads the CSV into SQLite)
  - `ui.py` (`apply_page_style()`: page config and CSS, call first on every page)
  - `prediction_log.py` (appends each `/predict` call to `data/predictions.db`; set `PREDICTION_LOG_PATH` to move it)
  - `monitoring.py` (Evidently drift check against the training data; alert rule and CLI)
- `models/` — ML artifacts: `churn_model.pkl`, `preprocessor.pkl` (ColumnTransformer), `metrics.json`
- `tests/` — Test suite: `test_pipeline.py` (unit/leakage tests), `test_predict.py` (inference), `test_app.py` (Streamlit AppTest), `test_mlflow.py` (MLflow tracking & registry), `test_monitoring.py` (drift check and log reader)
- `data/` — SQLite DB (`churn.db`), raw CSV in `data/raw/`
- `notebooks/` — EDA (`01_EDA.ipynb`), model training (`02_Model_Training.ipynb`)
- `docs/` — `project_definition.md` (business context)
- `.streamlit/` — Streamlit theme config

## Tech Stack (from requirements.txt and requirements-api.txt)
- streamlit==1.32.0
- pandas==2.2.1
- numpy==1.26.4
- scikit-learn==1.4.2
- matplotlib==3.8.3
- seaborn==0.13.2
- joblib==1.3.2
- pytest==9.1.1
- mlflow==2.14.3
- evidently==0.7.23
- shap==0.46.0
- fastapi==0.162.0
- uvicorn==0.54.0
- pydantic==2.14.1
- httpx==0.28.1
- setuptools<71
- sqlalchemy<2.1.0

## Common Commands
- **Run Streamlit**: `streamlit run app.py`
- **Train production model**: `python -m src.train`
- **Run benchmark experiments**: `python -m src.experiments`
- **Open MLflow UI**: `mlflow ui --backend-store-uri sqlite:///mlflow.db`
- **Run tests**: `pytest` or `python -m pytest`
- **Initialize DB**: `python src/data_ingest.py` (creates/loads churn.db)
- **Run API in Docker**: `docker build -t churn-api .` then `docker run -p 8000:8000 -v churn-log:/app/data churn-api` (the volume keeps the prediction log)
- **Smoke test a running API**: `python scripts/smoke_test.py [base_url]`
- **Check for drift**: `python -m src.monitoring` (saves `evidently_report.html`, exits 1 on an alert)

## Working Rules
- **Use .venv**: Run commands with the `.venv` python (Python 3.12, scikit-learn 1.4.2) to match serialized artifacts.
- **Dependencies**: anything `api/` or `src/` needs at runtime goes in `requirements-api.txt` (the Docker image installs only this); tools and Streamlit go in `requirements.txt`. Install with `-r requirements.txt`: running `pip install evidently` alone upgrades protobuf and breaks Streamlit and MLflow.
- **Never modify/delete** in `data/raw/` (source data immutability).
- **Put reusable logic in `src/`**, not in notebooks or Streamlit pages.
- **Preprocess format**: `preprocessor.pkl` stores a fitted `ColumnTransformer` with `set_output(transform="pandas")`.
- **Write pytest tests** for new features and verify `pytest` passes before committing.
- **Never hardcode API keys**; use `.env` (ensure `.env` in `.gitignore`).
- **Never commit changes to `models/*.pkl` without asking me first.** Each version adds to git history. Experiments must not overwrite the production artifacts (`promote=False`).
- **MLflow Tracking**: Wrapped under experiment `"churn-2.0"`. Only `promote=True` runs assign the `"champion"` alias to `"churn-model"` and update `models/`. Experiment runs (`promote=False`) are logged for comparison without touching production artifacts.
---

## SHAP Explainability — `explain_customer` Schema

The `src/explain.py` module provides `explain_customer(raw_input: dict, top_k: int = 5) -> dict`.

### Input
`raw_input`: Dictionary with the 15 keys in `src/schema.py` `MODEL_FEATURES`. A value may be `None`; gaps are filled the same way as in training.
- **tenure and handset**: `months` (int), `eqpdays` (float), `hnd_price` (float)
- **usage**: `mou_Mean`, `mou_cvce_Mean`, `avgqty`, `change_mou`, `ovrmou_Mean`, `drop_vce_Mean` (float)
- **billing**: `totmrc_Mean`, `avgrev`, `change_rev` (float)
- **account**: `uniqsubs` (int), `actvsubs` (int), `asl_flag` ("N" or "Y")

### Output Schema
```json
{
  "probability": 0.871,
  "base_value": -0.0201,
  "risk_level": "high",
  "reasons": [
    {
      "feature": "change_mou",
      "customer_value": -400.0,
      "shap_value": 0.892,
      "direction": "increases_risk"
    }
    // ... up to top_k entries
  ]
}
```

- **probability**: The model's score from 0 to 1. Show it as a risk score, never as a chance of churning: the training data is an even 50/50 sample
- **base_value**: Expected model output, in log-odds
- **risk_level**: `"high"` (≥ P90 threshold), `"moderate"` (≥ P70 threshold), `"low"` (below P70) — thresholds from `models/metrics.json`
- **reasons**: Top `top_k` raw features by absolute SHAP value, each with:
  - `feature`: Raw feature name (one-hot dummies summed)
  - `customer_value`: The customer's actual value for this feature
  - `shap_value`: Signed SHAP contribution in log-odds (positive = increases churn risk). `sigmoid(base_value + all shap values) == probability`
  - `direction`: `"increases_risk"` or `"decreases_risk"`

### Functions
- `explain_customer(raw_input, top_k=5)` → dict (schema above)
- `global_importance(sample_size=300)` → list[{"feature", "mean_abs_shap"}]
- `get_explainer()` → cached `shap.TreeExplainer` (lazy, never at import)

### Example
```python
from src.explain import explain_customer
from src.schema import EXAMPLE_HIGH_RISK

result = explain_customer(EXAMPLE_HIGH_RISK)
# result["risk_level"] == "high"
# result["reasons"][0]["feature"] == "change_mou"
```
