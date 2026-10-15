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
  - `train.py` (RandomForest training, evaluation metrics, threshold calculation)
  - `predict.py` (cached artifact loader, predict_churn inference)
  - `data_ingest.py` (DB creation/load), `sql_schema.sql` (DDL)
- `models/` — ML artifacts: `churn_model.pkl`, `preprocessor.pkl` (ColumnTransformer), `metrics.json`
- `tests/` — Test suite: `test_pipeline.py` (unit/leakage tests), `test_predict.py` (inference), `test_app.py` (Streamlit AppTest)
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

## Common Commands
- **Run Streamlit**: `streamlit run app.py`
- **Train model**: `python -m src.train`
- **Run tests**: `pytest` or `python -m pytest`
- **Initialize DB**: `python src/data_ingest.py` (creates/loads churn.db)

## Working Rules
- **Use .venv**: Run commands with the `.venv` python (Python 3.12, scikit-learn 1.4.2) to match serialized artifacts.
- **Never modify/delete** in `data/raw/` (source data immutability).
- **Put reusable logic in `src/`**, not in notebooks or Streamlit pages.
- **Preprocess format**: `preprocessor.pkl` stores a fitted `ColumnTransformer` with `set_output(transform="pandas")`.
- **Write pytest tests** for new features and verify `pytest` passes before committing.
- **Never hardcode API keys**; use `.env` (ensure `.env` in `.gitignore`).