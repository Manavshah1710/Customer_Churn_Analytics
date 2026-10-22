# 📊 Customer Churn Analytics Platform

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4.2-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SQLite](https://img.shields.io/badge/SQLite-3-003B57.svg?logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-000000.svg)](https://github.com/psf/black)

An end-to-end Machine Learning and Business Intelligence solution designed to predict customer churn, uncover underlying risk drivers, and empower retention teams with real-time risk scoring and actionable insights.

---

## 📌 Table of Contents

- [Overview & Business Value](#-overview--business-value)
- [Key Features](#-key-features)
- [Project Architecture](#-project-architecture)
- [Updates & Changelog (Recent Milestones)](#-updates--changelog-recent-milestones)
- [Current Model Performance](#-current-model-performance)
- [Project Structure](#-project-structure)
- [Database Schema](#-database-schema)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Database Ingestion](#database-ingestion)
  - [Model Training](#model-training)
  - [Running the Streamlit Web Application](#running-the-streamlit-web-application)
  - [Running Tests](#running-tests)
- [Developer Tooling & Failover Proxy](#-developer-tooling--failover-proxy)
- [Roadmap (Churn 2.0)](#-roadmap-churn-20)

---

## 💼 Overview & Business Value

Customer retention is significantly more cost-effective than customer acquisition (5–7× cheaper on average). For subscription businesses, losing customers directly erodes recurring monthly revenue (MRR), lifetime value (LTV), and market share.

This platform bridges the gap between raw subscriber databases and retention operations:
- **Identifies At-Risk Customers Early**: Uses a calibrated Random Forest classifier optimized for high recall to minimize undetected churn.
- **Dynamic Decile Risk Stratification**: Empirically segments customers into High (top 10%), Moderate (top 30%), and Low risk tiers based on test-set distributions.
- **Explains Why Customers Churn**: Surfaces key risk drivers such as contract duration, payment method, tenure, and technical support add-ons.
- **Multi-Role Enablement**: Equips business leaders, customer support agents, and marketing teams with dedicated dashboards and interactive real-time prediction tools.

---

## ✨ Key Features

1. **Executive Analytics Dashboard (`1_Dashboard.py`)**
   - High-level KPIs: Total Customer Base (7,043), Churn Rate (26.54%), High-Risk Customers Count (1,869), and Estimated Annual Revenue at Risk ($1.45M+).
   - Cohort distributions and tenure vs. churn visualizations.

2. **Real-Time Churn Predictor (`2_Predict_Churn.py`)**
   - Interactive profile builder covering Demographics, Account & Billing, and Subscribed Services.
   - Real-time scoring using the production preprocessing pipeline (`src/predict.py`).
   - Visual risk gauge with dynamic percentile cutoffs (High Risk $\ge 80.1\%$, Moderate Risk $\ge 50.8\%$).
   - Integration slot for LLM-generated personalized retention action plans.

3. **Deep-Dive Customer Insights (`3_Insights.py`)**
   - Empirical analysis of churn risk across contract types (Month-to-Month vs. Annual/Two-Year).
   - Impact of payment methods (Electronic check vs. automated credit card/bank transfer).
   - Influence of service bundles (Fiber optic internet, Tech Support, Online Security).

4. **Production Model Card (`4_Model_Info.py`)**
   - Full transparency on model specifications, hyperparameters, and validation strategy.
   - Interactive display of hold-out test metrics (ROC-AUC, Precision, Recall, F1-Score).
   - Test confusion matrix and impurity-based global feature importances.

5. **Universal Application Shell (`app.py`)**
   - Polished, responsive design with light mode and dark mode theme switching via CSS variables.
   - Clean sidebar navigation across all 5 sub-pages.

---

## 🏗️ Project Architecture

```
                                  ┌────────────────────────┐
                                  │ Raw CSV (7,043 Records)│
                                  └───────────┬────────────┘
                                              │
                                              ▼
                                    [src/data_ingest.py]
                                (Bulk executemany Ingestion)
                                              │
                                              ▼
                                  ┌────────────────────────┐
                                  │   SQLite DB: churn.db  │
                                  │ (customers, accounts,  │
                                  │        services)       │
                                  └───────────┬────────────┘
                                              │
                       ┌──────────────────────┴──────────────────────┐
                       │                                             │
                       ▼                                             ▼
             [src/preprocess.py]                            [Streamlit App]
      (Stratified 80/20 Train/Test Split,            (Dashboard & Insights Pages)
    Median Imputation & ColumnTransformer)                          │
                       │                                             │
                       ▼                                             │
              [src/train.py]                                         │
        (Random Forest / SMOTE Training,                             │
         Holdout Test Evaluation & P90/P70)                          │
                       │                                             │
                       ▼                                             ▼
          ┌───────────────────────────┐                    [pages/2_Predict_Churn.py]
          │      models/ Artifacts    │ ◄─────────────────  (Real-Time Scoring via
          │  • churn_model.pkl        │                        src/predict.py)
          │  • preprocessor.pkl       │
          │  • metrics.json           │
          └───────────────────────────┘
```

---

## 🚀 Updates & Changelog (Recent Milestones)

Here is a summary of all major refactoring and enhancements implemented up to this version:

### 1. Data Ingestion & Database Optimization
- **Migrated from slow row-by-row iteration to bulk insertion**: Rewrote `src/data_ingest.py` to use `cursor.executemany()` with `DataFrame.to_records()`, dramatically speeding up database creation.
- **Relational Normalization**: Enforced 3 distinct relational tables (`customers`, `accounts`, `services`) with primary/foreign key relationships in `src/sql_schema.sql`.

### 2. Leakage-Free Preprocessing Pipeline (`src/preprocess.py`)
- **Strict Train/Test Isolation**: Implemented `split_data()` to perform stratified 80/20 splitting **before** any feature imputation or scaling, eliminating data leakage.
- **Scikit-Learn `ColumnTransformer`**: Standardized scaling (`StandardScaler`) on numeric columns (`tenure`, `monthly_charges`, `total_charges`) and one-hot encoding (`OneHotEncoder(handle_unknown="ignore")`) on categorical features.
- **Robust Inference Handler (`transform_new`)**: Accepts single raw dictionaries, series, or multi-row DataFrames, replaces manual one-hot encoding, and gracefully handles unseen/rare categories without crashing.

### 3. Model Training, Calibration & Artifact Bundling (`src/train.py`)
- **Balanced Class Weighting**: Re-trained Random Forest with `class_weight="balanced"`, `n_estimators=300`, and `max_depth=8` to dramatically boost recall on churners without synthetic artifacts.
- **Optional SMOTE Resampling**: Added `--smote` flag for comparative training experiments.
- **Automated Metric Serialization**: Held-out evaluation metrics, confusion matrix, and percentile thresholds are automatically saved to `models/metrics.json`.
- **Top 10% Decile Capture**: Implemented rank-order evaluation measuring how many actual churners are caught within the top 10% highest-risk predictions (capturing **28.61%** of all churners in the first decile).
- **Dynamic Risk Cutoffs**: Replaced arbitrary static 50% cutoffs with empirical test-set distribution percentiles (90th percentile for High Risk, 70th percentile for Moderate Risk).

### 4. Production Inference Module (`src/predict.py`)
- Created a standalone inference interface with `@functools.lru_cache` to cache model and preprocessor objects in memory.
- Unified interface accepting `dict`, `pd.Series`, or `pd.DataFrame`, producing calibrated probability outputs.

### 5. Streamlit Multipage Application Overhaul
- **Centralized CSS System**: Replaced duplicate, conflicting stylesheets across pages with CSS custom properties in `app.py`, enabling seamless Light / Dark mode toggling.
- **Form-Based Inference**: Redesigned `pages/2_Predict_Churn.py` with multi-column organized input fields (Demographics, Billing, Services), batch evaluation, and a visual risk badge.
- **Model Info Modernization**: Updated `pages/4_Model_Info.py` to dynamically load values from `models/metrics.json` and display the holdout confusion matrix and top-10 feature importance.

### 6. Model Artifact Archiving
- Safely archived legacy v1 artifacts into `models/archive_v1/` (`churn_model_v1.pkl`, `scaler_v1.pkl`, `feature_columns_v1.pkl`) and v2 artifacts into `models/archive_v2/`.
- Production bundle in `models/` now consists of clean, standardized files: `churn_model.pkl`, `preprocessor.pkl`, and `metrics.json`.

### 7. Developer Tooling & LLM Proxy
- Built `claude_proxy.py` and `launch_claude.bat`: A resilient local failover proxy that automatically catches OpenRouter HTTP 429 rate limit errors and rotates across backup models and API keys for uninterrupted development.

---

## 📈 Current Model Performance

Evaluated on an **untouched holdout test set** (20% stratified split = 1,409 customers: 1,035 retained, 374 churned):

**Model: Random Forest (class_weight='balanced', n_estimators=300, max_depth=8, 45 features including senior_citizen)**

| Metric | Score | Description / Significance |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.8512** | Strong discriminatory ability across decision thresholds |
| **Recall (Class 1)** | **0.7647 (76.5%)** | Successfully detects 3 out of every 4 churners |
| **Precision (Class 1)** | **0.5366 (53.7%)** | 53.7% of flagged customers actually churn |
| **F1-Score (Class 1)**| **0.6307** | Balanced harmonic mean optimized for churn operations |
| **Overall Accuracy** | **0.7622 (76.2%)** | Balanced classification accuracy |
| **Top 10% Decile Capture** | **28.61%** | 107 of 374 churners are captured in the top 10% risk group |

### Test Confusion Matrix

```
                          Predicted Retained (0)    Predicted Churned (1)
 Actual Retained (0)               788                       247
 Actual Churned  (1)                88                       286
```

### Risk Stratification Cutoffs

Instead of an arbitrary 0.50 threshold, risk tiers are derived from the model's empirical test-set distribution:
- 🚨 **High Risk Tier (Top 10% cutoff / P90)**: $\ge 81.80\%$ probability
- ⚠️ **Moderate Risk Tier (Top 30% cutoff / P70)**: $\ge 60.70\%$ probability
- ✅ **Low Risk Tier**: $< 60.70\%$ probability

---

## 📂 Project Structure

```
Customer_Churn_Analytics/
├── app.py                      # Streamlit application entry point & theme configuration
├── claude_proxy.py             # Local resilient OpenRouter failover proxy
├── launch_claude.bat           # Batch launcher for development environment
├── requirements.txt            # Python dependencies (Streamlit, Scikit-learn, etc.)
├── CLAUDE.md                   # Agent working guidelines and repository context
├── README.md                   # Complete project documentation and status
│
├── data/
│   ├── churn.db                # SQLite database with normalized tables
│   └── raw/
│       └── telco_customer_churn.csv # Original Telco churn dataset (7,043 rows)
│
├── docs/
│   └── project_definition.md   # Business scope, problem statement, and requirements
│
├── models/
│   ├── churn_model.pkl         # Production Random Forest classifier
│   ├── preprocessor.pkl        # Fitted ColumnTransformer & metadata dictionary
│   ├── metrics.json            # Serialized evaluation metrics & risk thresholds
│   ├── archive_v1/             # Deprecated v1 artifacts (legacy scaler & feature cols)
│   └── archive_v2/             # Experimental v2 artifacts
│
├── notebooks/
│   ├── 01_EDA.ipynb            # Exploratory Data Analysis
│   └── 02_Model_Training.ipynb # Experimental training & model comparison
│
├── pages/
│   ├── 1_Dashboard.py          # Executive KPI & cohort dashboard
│   ├── 2_Predict_Churn.py      # Real-time customer churn prediction calculator
│   ├── 3_Insights.py           # Feature analysis and business risk drivers
│   ├── 4_Model_Info.py         # Production model metrics & feature importance
│   └── 5_About.py              # Business problem, architecture & solution overview
│
├── src/
│   ├── __init__.py             # Exports PROJECT_ROOT path helper
│   ├── data_ingest.py          # CSV-to-SQLite bulk database ingestion
│   ├── predict.py              # Inference module with LRU artifact caching
│   ├── preprocess.py           # Preprocessing pipeline & ColumnTransformer
│   ├── sql_schema.sql          # Relational DDL for SQLite database
│   └── train.py                # CLI model training & evaluation pipeline
│
└── tests/
    ├── __init__.py
    ├── test_pipeline.py        # Pipeline tests (leakage, splits, parity)
    └── test_predict.py         # Inference tests (single row, batch, risk ordering)
```

---

## 🗄️ Database Schema

The SQLite database (`data/churn.db`) stores customer records in 3 relational tables:

```sql
-- CUSTOMERS (Demographics & Tenure)
CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY,
    gender TEXT,
    senior_citizen INTEGER,
    partner TEXT,
    dependents TEXT,
    tenure INTEGER,
    churn TEXT
);

-- ACCOUNTS (Contracts, Invoicing, Billing)
CREATE TABLE accounts (
    account_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT,
    contract TEXT,
    paperless_billing TEXT,
    payment_method TEXT,
    monthly_charges REAL,
    total_charges REAL,
    FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
);

-- SERVICES (Telecom & Internet Subscriptions)
CREATE TABLE services (
    service_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT,
    phone_service TEXT,
    multiple_lines TEXT,
    internet_service TEXT,
    online_security TEXT,
    online_backup TEXT,
    device_protection TEXT,
    tech_support TEXT,
    streaming_tv TEXT,
    streaming_movies TEXT,
    FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
);
```

---

## 🛠️ Getting Started

### Prerequisites

- **Python**: Version `3.10`, `3.11`, or `3.12`
- **Git**

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Manavshah1710/Customer_Churn_Analytics.git
   cd Customer_Churn_Analytics
   ```

2. Create and activate a Python virtual environment:
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Database Ingestion

Ensure the SQLite database is created and populated from the raw CSV data:
```bash
python src/data_ingest.py
```

### Model Training

To retrain the production model (`rf_balanced`) and refresh artifacts in `models/`:
```bash
python -m src.train
```

To run with SMOTE oversampling:
```bash
python -m src.train --smote
```

### Running the Streamlit Web Application

Launch the application locally:
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### Running Tests

Run the unit and integration test suite:
```bash
pytest tests/
```

---

## 🛡️ Developer Tooling & Failover Proxy

The project includes custom tooling for continuous AI pair-programming:
- **`claude_proxy.py`**: A multi-account reverse proxy for OpenRouter that routes Anthropic-compatible API calls through a cascading pool of models. If an active key hits an HTTP 429 rate limit, the proxy switches upstream endpoints without dropping your session.
- **`launch_claude.bat`**: One-click startup script that configures `.claude/settings.local.json`, starts the proxy in the background, and launches Claude Code.

---

## 🔮 Roadmap (Churn 2.0)

- [ ] **SHAP Explainability Integration**: Individual waterfall and beeswarm plots directly inside the Streamlit prediction view to explain per-customer risk drivers.
- [ ] **LLM Retention Copilot**: Connect the Claude API / OpenRouter proxy to generate customer-tailored retention offers and scripts.
- [ ] **FastAPI Microservice (`api/main.py`)**: Standalone REST API with OpenAPI documentation for automated batch and webhook inference.
- [ ] **Docker Containerization**: Multi-stage `Dockerfile` and `docker-compose.yml` for unified deployment.
- [ ] **CI/CD Automation**: GitHub Actions workflow for linting, automated testing, and model regression checks.
- [ ] **MLflow Tracking & Drift Monitoring**: Experiment tracking with MLflow and data/model drift monitoring with Evidently AI.

---

## 📄 License & Attribution

This project is open-source and intended for portfolio, educational, and enterprise demonstration purposes.
Data source: IBM Telco Customer Churn sample dataset.
