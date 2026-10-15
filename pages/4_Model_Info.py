# =========================================
# pages/4_Model_Info.py — Model Performance & Metrics
# =========================================

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import PROJECT_ROOT
from src.predict import load_artifacts

# -----------------------------------------
# PAGE HEADER
# -----------------------------------------
st.markdown("""
<h2 style="margin-bottom:4px;">📊 Model Architecture & Test Evaluation</h2>
<p style="color:#64748b; margin-top:0;">
Rigorous held-out test set performance, error matrices, and global feature importance for the primary churn model
</p>
""", unsafe_allow_html=True)


# -----------------------------------------
# LOAD METRICS & MODEL ARTIFACTS (CACHED)
# -----------------------------------------
@st.cache_data
def load_metrics_data():
    metrics_path = PROJECT_ROOT / "models" / "metrics.json"
    if not metrics_path.exists():
        st.error("models/metrics.json not found. Please train the model via `python -m src.train`.")
        st.stop()
    with open(metrics_path, "r") as f:
        return json.load(f)


@st.cache_resource
def get_model_and_preprocessor():
    return load_artifacts(str(PROJECT_ROOT / "models"))


metrics = load_metrics_data()
model, preprocessor = get_model_and_preprocessor()

# -----------------------------------------
# HELD-OUT PERFORMANCE KPIS
# -----------------------------------------
st.markdown("### 🎯 Held-Out Test Set Performance (20% Stratified Split)")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("AUC-ROC Score", f"{metrics['auc']:.4f}")
k2.metric("Precision (Churn)", f"{metrics['precision']:.4f}")
k3.metric("Recall (Churn)", f"{metrics['recall']:.4f}")
k4.metric("F1 Score", f"{metrics['f1']:.4f}")
k5.metric("Accuracy", f"{metrics['accuracy']:.4f}")

st.markdown("---")

# -----------------------------------------
# TOP DECILE CAPTURE & RISK CUTOFFS
# -----------------------------------------
col_decile, col_cutoffs = st.columns([1.1, 0.9], gap="large")

with col_decile:
    st.markdown("#### 🎯 Top-Decile Churn Capture Rate")
    capture_rate = metrics.get("top_10_decile_capture_rate", 0.0)
    caught = metrics.get("churners_caught_in_top_10", 0)
    total = metrics.get("total_churners_in_test", 0)

    st.markdown(f"""
    <div style="
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-radius: 10px;
        padding: 16px;
    ">
        <h2 style="color: #15803D; margin: 0 0 4px 0;">{capture_rate:.2f}%</h2>
        <p style="color: #166534; margin: 0; font-size: 0.95rem;">
            <b>{caught} out of {total}</b> test churners are successfully captured within the top 10% highest predicted risk cohort.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Top-decile capture measures how effectively retention campaigns can prioritize outreach to catch the maximum churners with minimum intervention cost.")

with col_cutoffs:
    st.markdown("#### 📐 Risk Classification Cutoffs")
    th_high = metrics["threshold_high_risk"]
    th_mod = metrics["threshold_moderate_risk"]

    st.markdown(f"""
    <div style="
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
    ">
        <p style="margin: 0 0 8px 0;">
            <span style="color:#DC2626; font-weight:600;">🚨 High Risk (Top 10% Cutoff):</span> &ge; <b>{th_high * 100:.1f}%</b>
        </p>
        <p style="margin: 0 0 8px 0;">
            <span style="color:#D97706; font-weight:600;">⚠️ Moderate Risk (Top 30% Cutoff):</span> &ge; <b>{th_mod * 100:.1f}%</b>
        </p>
        <p style="margin: 0;">
            <span style="color:#059669; font-weight:600;">✅ Low Risk:</span> &lt; <b>{th_mod * 100:.1f}%</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# -----------------------------------------
# CONFUSION MATRIX & FEATURE IMPORTANCE
# -----------------------------------------
cm_col, fi_col = st.columns([0.9, 1.1], gap="large")

with cm_col:
    st.markdown("#### 🧮 Test Confusion Matrix")
    cm = np.array(metrics["confusion_matrix"])
    tn, fp = cm[0]
    fn, tp = cm[1]

    cm_df = pd.DataFrame(
        [
            [f"TN: {tn:,}", f"FP: {fp:,}"],
            [f"FN: {fn:,}", f"TP: {tp:,}"],
        ],
        index=["Actual Retained (0)", "Actual Churned (1)"],
        columns=["Predicted Retained (0)", "Predicted Churned (1)"],
    )
    st.dataframe(cm_df, use_container_width=True)

    st.caption(
        f"Evaluated on {tn + fp + fn + tp:,} holdout test samples: "
        f"{tn + tp:,} correct predictions, {fp:,} false alarms, {fn:,} missed churners."
    )

with fi_col:
    st.markdown("#### 🔍 Global feature importance (impurity-based)")

    feature_names = preprocessor.get("feature_columns", [])
    if hasattr(model, "feature_importances_") and len(feature_names) == len(model.feature_importances_):
        importances = model.feature_importances_
        feat_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances,
        }).sort_values(by="Importance", ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.barh(
            feat_df["Feature"][::-1],
            feat_df["Importance"][::-1],
            color="#4F46E5",
        )
        ax.set_xlabel("Mean Impurity Decrease", fontsize=9)
        ax.set_title("Top 10 Most Predictive Features", fontsize=10, weight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Feature importance data is unavailable for this model.")

st.markdown("---")

# -----------------------------------------
# MODEL ARCHITECTURE & NOTES
# -----------------------------------------
st.markdown("### ⚙️ Production Model Architecture")

st.markdown(f"""
<div style="
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 20px;
">
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px;">
        <div>
            <b style="color:#64748B;">Algorithm</b><br>
            <span>Random Forest Classifier</span>
        </div>
        <div>
            <b style="color:#64748B;">Class Weighting</b><br>
            <span><code>class_weight='balanced'</code></span>
        </div>
        <div>
            <b style="color:#64748B;">Estimators (Trees)</b><br>
            <span>{metrics.get('n_estimators', 300)}</span>
        </div>
        <div>
            <b style="color:#64748B;">Max Tree Depth</b><br>
            <span>{metrics.get('max_depth', 12)}</span>
        </div>
        <div>
            <b style="color:#64748B;">Feature Encoding</b><br>
            <span>One-Hot (no drop_first) + StandardScaler</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
