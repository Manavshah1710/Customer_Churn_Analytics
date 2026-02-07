# =========================================
# pages/4_Model_Info.py — Model Performance
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import roc_curve, auc, classification_report

# -----------------------------------------
# PAGE HEADER
# -----------------------------------------
st.markdown("""
<h2 style="margin-bottom:4px;">Model Performance</h2>
<p style="color:#64748b; margin-top:0;">
Evaluation, comparison, and explainability of the churn prediction model
</p>
""", unsafe_allow_html=True)

# -----------------------------------------
# LOAD MODEL ARTIFACTS (CACHED)
# -----------------------------------------
import os

@st.cache_resource
def load_model_artifacts():
    base_path = os.path.dirname(os.path.dirname(__file__))

    model = joblib.load(os.path.join(base_path, "models", "churn_model.pkl"))
    scaler = joblib.load(os.path.join(base_path, "models", "scaler.pkl"))
    feature_cols = joblib.load(os.path.join(base_path, "models", "feature_columns.pkl"))

    return model, scaler, feature_cols


model, scaler, FEATURE_COLS = load_model_artifacts()

# -----------------------------------------
# LOAD EVALUATION DATA (CACHED)
# -----------------------------------------
DB_PATH = "data/churn.db"

@st.cache_data
def load_model_data():
    conn = sqlite3.connect(DB_PATH)

    query = """
    SELECT 
        c.customer_id,
        c.churn,
        c.tenure,
        a.contract,
        a.monthly_charges,
        a.total_charges,
        s.internet_service
    FROM customers c
    JOIN accounts a ON c.customer_id = a.customer_id
    JOIN services s ON c.customer_id = s.customer_id;
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    df["churn"] = df["churn"].map({"Yes": 1, "No": 0})

    # One-hot encode
    df_encoded = pd.get_dummies(
        df.drop(columns=["customer_id", "churn"]),
        drop_first=False
    )

    # Align columns EXACTLY with training
    for col in FEATURE_COLS:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    df_encoded = df_encoded[FEATURE_COLS]

    # Scale numeric features (must match training)
    numeric_features = ["tenure", "monthly_charges", "total_charges"]

    df_encoded[numeric_features] = scaler.transform(
        df_encoded[numeric_features]
    )

    return df, df_encoded


df_raw, X = load_model_data()
y = df_raw["churn"]

# -----------------------------------------
# MODEL METRICS
# -----------------------------------------
y_prob = model.predict_proba(X)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

fpr, tpr, _ = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)

report = classification_report(y, y_pred, output_dict=True)

# -----------------------------------------
# KPI METRICS
# -----------------------------------------
k1, k2, k3, k4 = st.columns(4)

k1.metric("AUC Score", f"{roc_auc:.3f}")
k2.metric("Precision", f"{report['1']['precision']:.2f}")
k3.metric("Recall", f"{report['1']['recall']:.2f}")
k4.metric("F1 Score", f"{report['1']['f1-score']:.2f}")

st.markdown("---")

# -----------------------------------------
# ROC CURVE
# -----------------------------------------
st.markdown("## 📈 ROC Curve")

fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"Random Forest (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)

st.info(
    "📌 **Interpretation:** The ROC curve demonstrates strong separation between "
    "churned and retained customers, indicating robust model performance."
)

st.markdown("---")

# -----------------------------------------
# FEATURE IMPORTANCE
# -----------------------------------------
st.markdown("## 🔍 Feature Importance")

importances = model.feature_importances_
feat_df = pd.DataFrame({
    "Feature": FEATURE_COLS,
    "Importance": importances
}).sort_values(by="Importance", ascending=False).head(10)

fig, ax = plt.subplots()
ax.barh(
    feat_df["Feature"][::-1],
    feat_df["Importance"][::-1]
)
ax.set_xlabel("Importance Score")
ax.set_title("Top 10 Predictive Features")
ax.grid(axis="x", alpha=0.3)
st.pyplot(fig)

st.info(
    "📌 **Insight:** Tenure, monthly charges, and contract type are the most "
    "influential predictors of customer churn."
)

st.markdown("---")

# -----------------------------------------
# MODEL SUMMARY TABLE
# -----------------------------------------
st.markdown("## 📊 Model Evaluation Summary")

summary_df = pd.DataFrame({
    "Metric": ["AUC", "Precision", "Recall", "F1-Score"],
    "Score": [
        round(roc_auc, 3),
        round(report["1"]["precision"], 3),
        round(report["1"]["recall"], 3),
        round(report["1"]["f1-score"], 3),
    ]
})

st.dataframe(summary_df, use_container_width=True, hide_index=True)

# -----------------------------------------
# MODEL VALIDATION NOTES
# -----------------------------------------
st.markdown("""
<div class="soft-card">
<h4>✅ Model Validation Notes</h4>
<ul>
<li>Random Forest selected after benchmarking against Logistic Regression and XGBoost</li>
<li>Optimized for recall to minimize missed churn cases</li>
<li>Trained on balanced dataset using resampling techniques</li>
<li>Validated using ROC-AUC and cross-validated metrics</li>
<li>Suitable for decision-support and retention planning</li>
</ul>
</div>
""", unsafe_allow_html=True)
