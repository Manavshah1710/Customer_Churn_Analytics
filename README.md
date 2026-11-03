# 📊 Customer Churn Analytics

[![CI](https://github.com/Manavshah1710/Customer_Churn_Analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/Manavshah1710/Customer_Churn_Analytics/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4.2-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-tested%20in%20CI-2496ED.svg?logo=docker&logoColor=white)](Dockerfile)

A portfolio project that predicts which telecom customers are likely to churn and explains why. It covers the work around the model as well: a FastAPI service, a Streamlit app, a Docker image, CI, experiment tracking and data drift monitoring.

It is trained on 100,000 customer records from a public Kaggle dataset. It is not a live business deployment.

---

## 📌 Table of Contents

- [Results](#-results)
- [Key Decisions](#-key-decisions)
- [What the Data Shows](#-what-the-data-shows)
- [Limitations](#-limitations)
- [Features](#-features)
- [Architecture](#-architecture)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Roadmap](#-roadmap)
- [License & Attribution](#-license--attribution)

---

## 📈 Results

Measured on an untouched 20% test set of 20,000 customers (10,088 retained, 9,912 churned). Settings were chosen by 5-fold cross-validation on the training data only.

| Metric | Score |
| :--- | :--- |
| **Cross-validated ROC-AUC** | **0.684 ± 0.003** |
| **Test ROC-AUC** | **0.683** |
| Precision (churn) | 0.623 |
| Recall (churn) | 0.633 |
| F1 (churn) | 0.628 |
| Accuracy | 0.628 |

**Top 10% of risk scores:** 77.5% of these 2,000 customers churned, against 49.6% overall. That is 1,549 churners, or 15.6% of all churners. With about half of the sample churning, a top 10% list can catch at most 20.2%.

An AUC of 0.68 is modest, and the dataset is hard: no single column scores above 0.574 on its own.

### Model comparison

Each variant used the same 15 columns and the same split.

| Model | CV AUC | Test AUC |
| :--- | :--- | :--- |
| **HistGradientBoosting (300 iterations, depth 6, learning rate 0.05)** | **0.6837 ± 0.0032** | **0.6833** |
| HistGradientBoosting (500 iterations, depth 4) | 0.6835 ± 0.0041 | 0.6829 |
| HistGradientBoosting (100 iterations, no depth limit) | 0.6833 ± 0.0022 | 0.6812 |
| HistGradientBoosting (600 iterations, depth 3) | 0.6810 ± 0.0042 | 0.6810 |
| Random Forest (balanced, depth 16) | 0.6782 ± 0.0042 | 0.6780 |
| Random Forest (balanced, depth 8) | 0.6671 ± 0.0041 | 0.6666 |
| Logistic Regression (balanced) | 0.6081 ± 0.0036 | 0.6054 |

### Risk tiers

Tiers come from the test-set score distribution: **High** is the top 10% (score ≥ 0.697), **Moderate** is the top 30% (score ≥ 0.584), and the rest are **Low**.

---

## 🧭 Key Decisions

- **15 columns instead of 77.** Cross-validated permutation importance kept 15 columns at a cost of about 0.001 AUC (0.6833 vs 0.6842). The same list drives the form, the API and training, and nobody can reasonably fill in 77 fields.
- **Demographics excluded.** Ethnicity, marital status, income, children and other household columns never reach the model. A test enforces this. Removing them cost about 0.005 AUC.
- **A suspected leak was tested, not assumed.** The dataset's dictionary flags `change_mou` as possible leakage. Alone it scores only 0.530 AUC, and removing it costs about 0.0085. It stays, and that figure is the most the results could be overstated by. Rows missing it churn at 76%, which could mean those customers had already stopped using the service, so gaps are filled with the median and the model cannot use that signal.
- **The published data dictionary was checked.** It lists columns that are not in the file and calls `change_mou` a percentage, though 22.6% of its values are below -100.
- **The drift alert was set by measurement.** On healthy data, an alert at 3 or more drifted columns fired in 11 of 20 runs at 200 rows and 0 of 20 at 500, so 500 rows is the minimum. In an earlier version of this project, Evidently's default rule of 50% of columns missed a simulated shift, so this project uses its own threshold.
- **Tree SHAP in log-odds.** It explains 1,000 customers in about 2 seconds, against about 18 seconds for probability units, which matters for scoring in bulk.

---

## 🔎 What the Data Shows

These are associations in this dataset, not proof of cause.

- **A sharp jump near one year.** Churn is 31% at months 7-9, 40% at month 10 and 65% at month 11. This fits contracts ending near one year, but the data does not record contract terms.
- **Older and cheaper handsets.** Churn is 38% at 181-270 days of handset age and 55% at 271-365 days. It is 57% on the cheapest handsets and 41% on the most expensive.
- **Dropped calls barely matter on their own.** Churn stays between 47% and 50% across every band.

---

## ⚠️ Limitations

- **The churn rate is not realistic.** The data is split almost evenly, so 49.6% should not be read as a real company's churn rate. Scores track outcomes closely on this sample, but they are shown as risk scores rather than chances of churning, because a real customer base would differ.
- **The original source is unknown.** The Kaggle page does not name the carrier, the time period or why churn is balanced.
- **Most column meanings are inferred.** The dictionary defines 6 of the 15 columns. The other 9 are labelled as inferred in `src/schema.py`.
- **Drift is simulated.** A static dataset has no real drift, so the Monitoring page includes a clearly labelled demo. The rule of 3 columns and 500 rows was measured against that one simulated shift, and needs revisiting with real traffic.

---

## ✨ Features

1. **Dashboard:** customer KPIs and churn by tenure, handset age, usage and change in usage.
2. **Predict Churn:** a 15-field form that returns a risk score, a risk tier and a SHAP chart of the top drivers.
3. **Insights:** the findings above, calculated from the data each time the page loads.
4. **Model Info:** held-out metrics, risk cutoffs, the confusion matrix and SHAP feature importance.
5. **Monitoring:** drift checks on logged predictions, a healthy baseline or simulated drift, with a threshold alert and a downloadable Evidently report.
6. **API:** `GET /health`, `POST /predict` and `GET /model-info`, with validated input. `/health` returns 503 when the service is degraded.

---

## 🏗️ Architecture

```
Raw CSV (100,000 rows) --> src/data_ingest.py --> SQLite (data/churn.db)
                                                        |
                       +--------------------------------+---------------------+
                       v                                                      v
 src/preprocess.py + src/train.py                                 Dashboard and Insights pages
 split first, then impute, scale, encode
 HistGradientBoosting, MLflow tracking
                       |
                       v
 models/: churn_model.pkl, preprocessor.pkl, metrics.json
                       |
                       v
 src/explain.py (SHAP) --> FastAPI /predict --> src/prediction_log.py (data/predictions.db)
                       |                                     |
                       v                                     v
              Predict Churn page                 src/monitoring.py (Evidently)
                                                             |
                                                             v
                                                      Monitoring page
```

---

## 🛠️ Getting Started

Requires **Python 3.12**. The pinned versions match the saved model.

```bash
git clone https://github.com/Manavshah1710/Customer_Churn_Analytics.git
cd Customer_Churn_Analytics

# Windows (PowerShell)
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3.12 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
python src/data_ingest.py
```

Install through `requirements.txt`. Installing Evidently on its own upgrades protobuf and breaks Streamlit and MLflow.

| Task | Command |
| :--- | :--- |
| Run the app | `streamlit run app.py` |
| Run the API | `uvicorn api.main:app --reload` |
| Run the tests | `python -m pytest` |
| Compare models | `python -m src.experiments` |
| Retrain and replace the production model | `python -m src.train` |
| View experiments | `mlflow ui --backend-store-uri sqlite:///mlflow.db` |
| Check for drift | `python -m src.monitoring` (exits 1 on an alert) |
| Run the API in Docker | `docker build -t churn-api .` then `docker run -p 8000:8000 -v churn-log:/app/data churn-api` |

---

## 📂 Project Structure

```
├── app.py                 # Streamlit entry point
├── api/                   # FastAPI service and request/response schemas
├── pages/                 # Dashboard, Predict Churn, Insights, Model Info, About, Monitoring
├── src/
│   ├── schema.py          # The 15 model columns: limits, labels, meanings
│   ├── data_ingest.py     # Loads the CSV into SQLite
│   ├── preprocess.py      # Split, impute, scale, encode
│   ├── train.py           # Training, evaluation, MLflow, promotion
│   ├── experiments.py     # Cross-validated model comparison
│   ├── predict.py         # Cached model loading and scoring
│   ├── explain.py         # SHAP explanations
│   ├── prediction_log.py  # Stores each API prediction
│   ├── monitoring.py      # Evidently drift check and alert rule
│   └── ui.py              # Shared page config and styles
├── models/                # Production model, preprocessor and metrics
├── data/raw/              # Source CSV files
├── scripts/               # SHAP benchmark and API smoke test
├── tests/                 # Pipeline, API, app, SHAP, MLflow and monitoring tests
├── notebooks/             # Early exploration of the previous dataset
├── Dockerfile             # API image, run as a non-root user
└── .github/workflows/     # Lint, tests and Docker smoke test
```

---

## 🔮 Roadmap

- [x] SHAP explanations per customer
- [x] FastAPI service
- [x] MLflow experiment tracking
- [x] Evidently drift monitoring with a threshold alert
- [x] Docker image and GitHub Actions CI
- [ ] Public deployment
- [ ] CSV upload: score a file and download a ranked list
- [ ] LLM retention recommendations

---

## 📄 License & Attribution

This is a resume and portfolio project, intended for educational and demonstration purposes. The code has no open-source licence.

Data sources:

- [Telecom Customer Churn - 100k Cleaned Records](https://www.kaggle.com/datasets/shenoudasafwat/telecom-customer-churn-100k-cleaned-records) by Kaggle user `shenoudasafwat`, licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), provided as is with no warranty. Stored at `data/raw/Telecom_customer churn_100k.csv`. Changes made: this project loads it into SQLite and trains on 15 of its columns.
- IBM Telco Customer Churn sample dataset (earlier versions of this project).

Non-commercial use only: models trained on the 100k dataset, and any modified versions of that data, are shared under the same CC BY-NC-SA 4.0 licence and may not be used for commercial purposes.
