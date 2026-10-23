# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
**Today**: Telco churn prediction model (Random Forest) + Streamlit app with 5 pages + SQLite database. Data ingested from CSV into normalized schema (customers/accounts/services). Model artifacts (.pkl) saved in models/.

**Churn 2.0 Adds**: SHAP explainability, LLM retention recommendations (Claude API), FastAPI microservice, Docker containerization, GitHub Actions CI/CD, MLflow experiment tracking, Evidently AI drift monitoring.

## Folder Structure & Purpose
- `app.py` — Streamlit multipage app entry point (navigation shell)
- `pages/` — Streamlit pages: 1_Dashboard.py, 2_Predict_Churn.py, 3_Insights.py, 4_Model_Info.py, 5_About.py
- `src/` — Reusable logic:
  - `preprocess.py` (data loading, stratified split, ColumnTransformer feature engineering)
  - `train.py` (RandomForest training, evaluation metrics, threshold calculation, MLflow tracking)
  - `experiments.py` (comparative benchmark experiments with 5-fold CV)
  - `predict.py` (cached artifact loader, predict_churn inference)
  - `data_ingest.py` (DB creation/load), `sql_schema.sql` (DDL)
- `models/` — ML artifacts: `churn_model.pkl`, `preprocessor.pkl` (ColumnTransformer), `metrics.json`
- `tests/` — Test suite: `test_pipeline.py` (unit/leakage tests), `test_predict.py` (inference), `test_app.py` (Streamlit AppTest), `test_mlflow.py` (MLflow tracking & registry)
- `data/` — SQLite DB (`churn.db`), raw CSV in `data/raw/`
- `notebooks/` — EDA (`01_EDA.ipynb`), model training (`02_Model_Training.ipynb`)
- `docs/` — `project_definition.md` (business context)
- `.streamlit/` — Streamlit theme config

## Tech Stack (from requirements.txt)
- streamlit==1.32.0
- pandas==2.2.1
- numpy==1.26.4
- scikit-learn==1.4.2
- matplotlib==3.8.3
- seaborn==0.13.2
- joblib==1.3.2
- imbalanced-learn==0.14.2
- pytest==9.1.1
- mlflow==2.14.3
- setuptools<71
- sqlalchemy<2.1.0

## Common Commands
- **Run Streamlit**: `streamlit run app.py`
- **Train production model**: `python -m src.train`
- **Run benchmark experiments**: `python -m src.experiments`
- **Open MLflow UI**: `mlflow ui --backend-store-uri sqlite:///mlflow.db`
- **Run tests**: `pytest` or `python -m pytest`
- **Initialize DB**: `python src/data_ingest.py` (creates/loads churn.db)

## Working Rules
- **Use .venv**: Run commands with the `.venv` python (Python 3.12, scikit-learn 1.4.2) to match serialized artifacts.
- **Never modify/delete** in `data/raw/` (source data immutability).
- **Put reusable logic in `src/`**, not in notebooks or Streamlit pages.
- **Preprocess format**: `preprocessor.pkl` stores a fitted `ColumnTransformer` with `set_output(transform="pandas")`.
- **Write pytest tests** for new features and verify `pytest` passes before committing.
- **Never hardcode API keys**; use `.env` (ensure `.env` in `.gitignore`).
- **Never commit changes to `models/*.pkl` without asking me first.** Each version adds ~30 MB to git history. Experiments must not overwrite the production artifacts (`promote=False`).
- **MLflow Tracking**: Wrapped under experiment `"churn-2.0"`. Only `promote=True` runs assign the `"champion"` alias to `"churn-rf"` and update `models/`. Experiment runs (`promote=False`) are logged for comparison without touching production artifacts.
---

## SHAP Explainability — `explain_customer` Schema

The `src/explain.py` module provides `explain_customer(raw_input: dict, top_k: int = 5) -> dict`.

### Input
`raw_input`: Dictionary with all 19 raw feature keys:
- **demographics**: `gender` (str), `senior_citizen` (0/1), `partner` (str), `dependents` (str)
- **account**: `tenure` (int), `contract` (str), `paperless_billing` (str), `payment_method` (str), `monthly_charges` (float), `total_charges` (float)
- **services**: `phone_service` (str), `multiple_lines` (str), `internet_service` (str), `online_security` (str), `online_backup` (str), `device_protection` (str), `tech_support` (str), `streaming_tv` (str), `streaming_movies` (str)

### Output Schema
```json
{
  "probability": 0.8923,
  "base_value": 0.4995,
  "risk_level": "high",
  "reasons": [
    {
      "feature": "contract",
      "customer_value": "Month-to-month",
      "shap_value": 0.0901,
      "direction": "increases_risk"
    },
    {
      "feature": "tenure",
      "customer_value": 1,
      "shap_value": 0.0735,
      "direction": "increases_risk"
    }
    // ... up to top_k entries
  ]
}
```

- **probability**: Model's predicted churn probability for this customer
- **base_value**: Expected model output (mean prediction over training data)
- **risk_level**: `"high"` (≥ P90 threshold), `"moderate"` (≥ P70 threshold), `"low"` (below P70) — thresholds from `models/metrics.json`
- **reasons**: Top `top_k` raw features by absolute SHAP value, each with:
  - `feature`: Raw feature name (one-hot dummies summed)
  - `customer_value`: The customer's actual value for this feature
  - `shap_value`: Signed SHAP contribution (positive = increases churn risk)
  - `direction`: `"increases_risk"` or `"decreases_risk"`

### Functions
- `explain_customer(raw_input, top_k=5)` → dict (schema above)
- `global_importance(sample_size=300)` → list[{"feature", "mean_abs_shap"}]
- `get_explainer()` → cached `shap.TreeExplainer` (lazy, never at import)

### Example
```python
from src.explain import explain_customer

customer = {
    "gender": "Female", "senior_citizen": 0, "partner": "No", "dependents": "No",
    "tenure": 1, "contract": "Month-to-month", "paperless_billing": "Yes",
    "payment_method": "Electronic check", "monthly_charges": 85.0, "total_charges": 85.0,
    "phone_service": "Yes", "multiple_lines": "No", "internet_service": "Fiber optic",
    "online_security": "No", "online_backup": "No", "device_protection": "No",
    "tech_support": "No", "streaming_tv": "No", "streaming_movies": "No",
}

result = explain_customer(customer)
# result["risk_level"] == "high"
# result["reasons"][0]["feature"] == "contract"
```
