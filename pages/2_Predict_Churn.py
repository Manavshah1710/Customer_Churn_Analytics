# =========================================
# pages/2_Predict_Churn.py — Churn Prediction
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------------------
# PAGE HEADER
# -----------------------------------------
st.markdown("""
<h2 style="margin-bottom:4px;">Predict Customer Churn</h2>
<p style="color:#64748b; margin-top:0;">
Estimate the likelihood of a customer leaving using a trained machine learning model
</p>
""", unsafe_allow_html=True)

# -----------------------------------------
# LOAD MODEL ARTIFACTS (CACHED)
# -----------------------------------------
import os

@st.cache_resource
def load_artifacts():
    base_path = os.path.dirname(os.path.dirname(__file__))

    model = joblib.load(os.path.join(base_path, "models", "churn_model.pkl"))
    scaler = joblib.load(os.path.join(base_path, "models", "scaler.pkl"))
    feature_cols = joblib.load(os.path.join(base_path, "models", "feature_columns.pkl"))

    return model, scaler, feature_cols


model, scaler, FEATURE_COLS = load_artifacts()

# -----------------------------------------
# INPUT FORM (LEFT) + RESULT (RIGHT)
# -----------------------------------------
input_col, result_col = st.columns([1.1, 0.9])

with input_col:
    st.markdown("""
    <div class="soft-card">
    <h4>📋 Customer Information</h4>
    <p style="color:#6b7280;">Fill in customer details to predict churn probability</p>
    """, unsafe_allow_html=True)

    with st.form("prediction_form"):
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        monthly_charges = st.number_input(
            "Monthly Charges ($)", min_value=0.0, value=70.0, step=1.0
        )

        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        submitted = st.form_submit_button("🔮 Predict Churn")

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------
# PREDICTION LOGIC
# -----------------------------------------
with result_col:
    st.markdown("""
    <div class="soft-card">
    <h4>📊 Prediction Result</h4>
    """, unsafe_allow_html=True)

    if submitted:
        # ---- Build input row using get_dummies for one-hot encoding ----
        input_data = {
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "contract": contract,
            "internet_service": internet_service
        }

        input_df = pd.DataFrame([input_data])
        input_df = pd.get_dummies(input_df)

        # ---- Align with training features ----
        input_df = input_df.reindex(columns=FEATURE_COLS, fill_value=0)

        # ---- Scale numeric features ----
        input_df[["tenure", "monthly_charges"]] = scaler.transform(
            input_df[["tenure", "monthly_charges"]]
        )

        # ---- Predict ----
        churn_prob = model.predict_proba(input_df)[0][1]
        churn_label = "High Risk" if churn_prob >= 0.5 else "Low Risk"

        # ---- Display result ----
        st.markdown(f"""
        <h2 style="margin-bottom:4px;">{churn_label}</h2>
        <p style="color:#6b7280;">Churn Probability</p>
        """, unsafe_allow_html=True)

        st.progress(int(churn_prob * 100))
        st.markdown(f"""
        <h3>{churn_prob * 100:.1f}%</h3>
        """, unsafe_allow_html=True)

        # ---- Business recommendation ----
        if churn_prob >= 0.7:
            message = "🚨 Immediate retention action recommended"
        elif churn_prob >= 0.5:
            message = "⚠️ Moderate churn risk – monitor closely"
        else:
            message = "✅ Low churn risk – customer appears stable"

        st.info(message)

    else:
        st.caption("Submit customer details to view prediction")

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------
# MODEL DISCLAIMER
# -----------------------------------------
st.markdown("""
<div style="
margin-top:30px;
padding:18px;
border-radius:14px;
background:#EEF2FF;
">
<b>ℹ️ Model Note:</b><br>
This prediction is generated using a Random Forest classifier trained on historical customer data.
Predictions should be used to support — not replace — business decisions.
</div>
""", unsafe_allow_html=True)
