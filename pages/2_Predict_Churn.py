# =========================================
# pages/2_Predict_Churn.py — Churn Prediction
# =========================================

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src import PROJECT_ROOT
from src.predict import load_artifacts, predict_churn
from src.explain import explain_customer
from src.ui import apply_page_style
from src.schema import ASL_FLAG_HELP, ASL_FLAG_LABEL, AslFlag, FEATURE_LABELS, FIELD_GROUPS, NUMERIC_FIELDS

apply_page_style()

# -----------------------------------------
# PAGE HEADER
# -----------------------------------------
st.markdown("""
<h2 style="margin-bottom:4px;">🔮 Predict Customer Churn</h2>
<p style="color:#64748b; margin-top:0;">
Estimate customer churn risk in real time using the gradient boosting model
</p>
""", unsafe_allow_html=True)


# -----------------------------------------
# LOAD MODEL ARTIFACTS & METRICS (CACHED)
# -----------------------------------------
@st.cache_resource
def get_model_and_preprocessor():
    return load_artifacts(str(PROJECT_ROOT / "models"))


@st.cache_data
def get_metrics_config():
    metrics_path = PROJECT_ROOT / "models" / "metrics.json"
    with open(metrics_path, "r") as f:
        return json.load(f)


model, preprocessor = get_model_and_preprocessor()
metrics = get_metrics_config()

threshold_high = metrics["threshold_high_risk"]
threshold_moderate = metrics["threshold_moderate_risk"]

# -----------------------------------------
# INPUT FORM (LEFT) + RESULT (RIGHT)
# -----------------------------------------
input_col, result_col = st.columns([1.2, 0.8], gap="large")

with input_col:
    with st.form("prediction_form"):
        st.markdown("### 📋 Customer Profile")

        raw_customer_dict = {}
        for group, fields in FIELD_GROUPS.items():
            st.markdown(f"##### {group}")
            columns = st.columns(3)
            for index, name in enumerate(fields):
                spec = NUMERIC_FIELDS[name]
                raw_customer_dict[name] = columns[index % 3].number_input(
                    spec["label"],
                    min_value=spec["min"],
                    max_value=spec["max"],
                    value=spec["default"],
                    step=spec["step"],
                    help=spec["help"],
                )

        raw_customer_dict["asl_flag"] = st.selectbox(
            ASL_FLAG_LABEL,
            list(AslFlag.__args__),  # type: ignore
            help=ASL_FLAG_HELP,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔮 Predict Churn Risk", use_container_width=True)

# -----------------------------------------
# PREDICTION RESULT (RIGHT COLUMN)
# -----------------------------------------
with result_col:
    st.markdown("### 📊 Prediction Assessment")

    if submitted:
        churn_prob = predict_churn(
            raw_customer_dict,
            model=model,
            preprocessor=preprocessor,
        )

        # Classify risk tier based on test-set percentile cutoffs
        if churn_prob >= threshold_high:
            risk_tier = "High Risk"
            badge_color = "#DC2626"
            bg_color = "#FEF2F2"
            border_color = "#FCA5A5"
            status_icon = "🚨"
        elif churn_prob >= threshold_moderate:
            risk_tier = "Moderate Risk"
            badge_color = "#D97706"
            bg_color = "#FFFBEB"
            border_color = "#FCD34D"
            status_icon = "⚠️"
        else:
            risk_tier = "Low Risk"
            badge_color = "#059669"
            bg_color = "#ECFDF5"
            border_color = "#6EE7B7"
            status_icon = "✅"

        st.markdown(f"""
        <div style="
            background:{bg_color};
            border:2px solid {border_color};
            border-radius:12px;
            padding:20px;
            text-align:center;
            margin-bottom:20px;
        ">
            <span style="font-size:2rem;">{status_icon}</span>
            <h2 style="color:{badge_color}; margin:6px 0 2px 0;">{risk_tier}</h2>
            <p style="color:#6B7280; font-size:0.9rem; margin:0;">Churn Risk Score</p>
            <h1 style="color:{badge_color}; margin:8px 0; font-size:2.8rem;">{churn_prob:.3f}</h1>
        </div>
        """, unsafe_allow_html=True)

        st.progress(min(max(float(churn_prob), 0.0), 1.0))

        st.markdown(f"""
        <div style="font-size:0.85rem; color:#6B7280; margin-top:8px;">
            <b>Dynamic Risk Cutoffs:</b><br>
            • High Risk (Top 10% test decile): &ge; <b>{threshold_high:.3f}</b><br>
            • Moderate Risk (Top 30% test cutoff): &ge; <b>{threshold_moderate:.3f}</b><br>
            The score ranks customers from 0 to 1. It is not a chance of churning.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # SHAP Explanation — Horizontal Bar Chart
        st.markdown("### 🔍 Why this score? (Top risk drivers)")
        try:
            explanation = explain_customer(raw_customer_dict, top_k=8)
            reasons = explanation["reasons"]

            # Build horizontal bar chart
            features = [FEATURE_LABELS[r["feature"]] for r in reasons]
            shap_vals = [r["shap_value"] for r in reasons]
            colors = ["#DC2626" if r["direction"] == "increases_risk" else "#059669" for r in reasons]

            fig, ax = plt.subplots(figsize=(7, 3.5))
            y_pos = np.arange(len(features))[::-1]  # reverse for top-to-bottom
            ax.barh(y_pos, shap_vals[::-1], color=colors[::-1], edgecolor="white", height=0.6)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features[::-1], fontsize=9)
            ax.axvline(x=0, color="#94A3B8", linewidth=0.8)
            ax.set_xlabel("SHAP value (log-odds impact on risk score)", fontsize=9)
            ax.set_title("Feature contributions to this prediction", fontsize=10, weight="bold")
            ax.grid(axis="x", linestyle="--", alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Legend
            st.markdown("""
            <div style="font-size:0.8rem; color:#6B7280; margin-top:4px;">
                <span style="color:#DC2626;">■</span> Increases churn risk &nbsp;&nbsp;
                <span style="color:#059669;">■</span> Decreases churn risk
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.caption(f"SHAP explanation unavailable: {e}")

        st.markdown("---")

        # LLM Recommendation Placeholder
        st.markdown("""
        <div style="
            background: #F8FAFC;
            border: 1px dashed #94A3B8;
            border-radius: 10px;
            padding: 16px;
            margin-top: 15px;
        ">
            <h4 style="margin:0 0 6px 0; color:#334155;">🤖 Retention recommendation (coming from the LLM layer)</h4>
            <p style="margin:0; font-size:0.85rem; color:#64748B;">
                Personalized retention actions tailored to this customer's usage, handset, and billing will be generated dynamically here by the LLM reasoning layer.
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info("👈 Fill in the customer profile and click **Predict Churn Risk** to generate a risk score.")

st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("ℹ️ Model: HistGradientBoosting • Evaluated on stratified hold-out test set.")
