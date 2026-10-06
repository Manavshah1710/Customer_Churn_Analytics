# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
**Today**: Telco churn prediction model (Random Forest) + Streamlit app with 5 pages + SQLite database. Data ingested from CSV into normalized schema (customers/accounts/services). Model artifacts (.pkl) saved in models/.

**Churn 2.0 Adds**: SHAP explainability, LLM retention recommendations (Claude API), FastAPI microservice, Docker containerization, GitHub Actions CI/CD, MLflow experiment tracking, Evidently AI drift monitoring.

## Folder Structure & Purpose
- `app.py` — Streamlit multipage app entry point (navigation shell)
- `pages/` — Streamlit pages: 1_Dashboard.py, 2_Predict_Churn.py, 3_Insights.py, 4_Model_Info.py, 5_About.py
- `src/` — Reusable logic: `data_ingest.py` (DB creation/load), `sql_schema.sql` (DDL)
- `models/` — ML artifacts: `churn_model.pkl`, `scaler.pkl`, `feature_columns.pkl`
- `data/` — SQLite DB (`churn.db`); **raw/ planned for source CSV**
- `notebooks/` — EDA (`01_EDA.ipynb`), model training (`02_Model_Training.ipynb`)
- `docs/` — `project_definition.md` (business context)
- `.streamlit/` — Streamlit theme config
- **Planned**: `api/` (FastAPI service), `tests/` (pytest suite), `.github/workflows/` (CI/CD)

## Tech Stack (from requirements.txt)
- streamlit==1.32.0
- pandas==2.2.1
- numpy==1.26.4
- scikit-learn==1.4.2
- matplotlib==3.8.3
- seaborn==0.13.2
- joblib==1.3.2

## Common Commands
- **Run Streamlit**: `streamlit run app.py`
- **Train model (planned)**: `python -m src.train` (to be implemented)
- **Evaluate (planned)**: `python -m src.evaluate`
- **Run API (planned)**: `uvicorn api.main:app --reload`
- **Run tests (planned)**: `pytest tests/`
- **Build Docker (planned)**: `docker build -t churn-2.0 .`
- **Initialize DB**: `python src/data_ingest.py` (creates/loads churn.db)

## Working Rules
- **Never modify/delete** in `data/raw/` (planned for source data immutability)
- **Ask before adding dependencies**; add to requirements.txt when approved
- **Put reusable logic in `src/`**, not in notebooks or Streamlit pages
- **Write pytest tests** for new code in `src/` and `api/`
- **Never hardcode API keys**; use `.env` (ensure `.env` in `.gitignore`)
- **Keep SHAP output format** and `/predict` response stable once defined
- **Explain changes** after each task; keep changes small and focused

## Definition of Done
Code runs → Tests pass (if exist) → No unrelated files touched → Changes documented in task explanation.

## Notes & Issues
- **Critical**: `.venv/` and model `.pkl` files are currently tracked in git (no .gitignore). This must be fixed before adding dependencies.
- **Ponytail review findings**: data_ingest.py uses inefficient row-wise inserts; Predict page has manual one-hot encoding; duplicate column alignment in pages 2 and 4; app.py has duplicated CSS; unused check_columns.py exists.