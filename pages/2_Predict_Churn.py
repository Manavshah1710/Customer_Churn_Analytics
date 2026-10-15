# =========================================
# pages/2_Predict_Churn.py — Churn Prediction
# =========================================

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src import PROJECT_ROOT
from src.predict import load_artifacts, predict_churn

# -----------------------------------------
# PAGE HEADER
# -----------------------------------------
st.markdown("""
<h2 style="margin-bottom:4px;">🔮 Predict Customer Churn</h2>
<p style="color:#64748b; margin-top:0;">
Estimate customer churn risk in real time using the calibrated Random Forest model
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
        st.markdown("### 📋 Customer Profile & Service Information")

        # Tab / Section 1: Demographics
        st.markdown("##### 👤 Demographics")
        demo_col1, demo_col2 = st.columns(2)
        with demo_col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen_opt = st.selectbox("Senior Citizen", ["No", "Yes"])
            senior_citizen = 1 if senior_citizen_opt == "Yes" else 0
        with demo_col2:
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])

        st.markdown("---")

        # Tab / Section 2: Account & Billing
        st.markdown("##### 💳 Account & Billing Details")
        acc_col1, acc_col2 = st.columns(2)
        with acc_col1:
            tenure = st.slider("Tenure (Months with Company)", min_value=0, max_value=72, value=12)
            contract = st.selectbox(
                "Contract Type",
                ["Month-to-month", "One year", "Two year"]
            )
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

        with acc_col2:
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ]
            )
            monthly_charges = st.number_input(
                "Monthly Charges ($)", min_value=0.0, max_value=250.0, value=70.0, step=1.0
            )
            estimated_total = round(float(tenure * monthly_charges), 2)
            total_charges = st.number_input(
                "Total Charges ($)", min_value=0.0, max_value=15000.0, value=estimated_total, step=10.0
            )

        st.markdown("---")

        # Tab / Section 3: Subscribed Services
        st.markdown("##### 🌐 Subscribed Services")
        serv_col1, serv_col2, serv_col3 = st.columns(3)

        with serv_col1:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])

        with serv_col2:
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

        with serv_col3:
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔮 Predict Churn Risk", use_container_width=True)

# -----------------------------------------
# PREDICTION RESULT (RIGHT COLUMN)
# -----------------------------------------
with result_col:
    st.markdown("### 📊 Prediction Assessment")

    if submitted:
        raw_customer_dict = {
            "gender": gender,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "tenure": tenure,
            "contract": contract,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "phone_service": phone_service,
            "multiple_lines": multiple_lines,
            "internet_service": internet_service,
            "online_security": online_security,
            "online_backup": online_backup,
            "device_protection": device_protection,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
        }

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
            <p style="color:#6B7280; font-size:0.9rem; margin:0;">Estimated Churn Probability</p>
            <h1 style="color:{badge_color}; margin:8px 0; font-size:2.8rem;">{churn_prob * 100:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)

        st.progress(min(max(float(churn_prob), 0.0), 1.0))

        st.markdown(f"""
        <div style="font-size:0.85rem; color:#6B7280; margin-top:8px;">
            <b>Dynamic Risk Cutoffs:</b><br>
            • High Risk (Top 10% test decile): &ge; <b>{threshold_high * 100:.1f}%</b><br>
            • Moderate Risk (Top 30% test cutoff): &ge; <b>{threshold_moderate * 100:.1f}%</b>
        </div>
        """, unsafe_allow_html=True)

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
                Personalized retention actions tailored to this customer's contract, tenure, and services will be generated dynamically here by the LLM reasoning layer.
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info("👈 Fill in the customer profile and click **Predict Churn Risk** to generate a risk score.")

st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("ℹ️ Model: Balanced Random Forest Classifier (v2) • Evaluated on stratified hold-out test set.")
