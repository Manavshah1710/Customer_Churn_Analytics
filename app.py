# ===============================
# app.py — Application Shell
# ===============================

import streamlit as st

from src.ui import apply_page_style

# -------------------------------
# 1️⃣ PAGE CONFIG & STYLE (MUST BE FIRST)
# -------------------------------
apply_page_style()

# --------------------------------
# 2️⃣ MAIN LANDING CONTENT
# --------------------------------
st.title("📊 Customer Churn Analytics Platform")
st.caption(
    "An end-to-end data science application for churn prediction, insights, and business intelligence."
)

st.markdown("---")

st.info("⬅ Use the sidebar to navigate between Dashboard, Predictions, Insights, Model Info, About, and Monitoring pages.")

# --------------------------------
# 3️⃣ FOOTER (OPTIONAL, CLEAN)
# --------------------------------
st.markdown(
    """
    <div style="
        margin-top:40px;
        padding:20px;
        border-radius:14px;
        background:linear-gradient(135deg,#0F172A,#1E293B);
        color:#E5E7EB;
        text-align:center;
    ">
        <small>
        © 2026 Customer Churn Analytics Platform • Production-Ready ML Solution
        </small>
    </div>
    """,
    unsafe_allow_html=True
)
