# =========================================
# pages/3_Insights.py — Business Insights
# =========================================

import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

# -----------------------------------------
# PAGE HEADER
# -----------------------------------------
st.markdown("""
<h2 style="margin-bottom:4px;">Customer Insights</h2>
<p style="color:#64748b; margin-top:0;">
Deep-dive analysis into churn drivers and customer behavior patterns
</p>
""", unsafe_allow_html=True)

# -----------------------------------------
# DATA LOADING (CACHED)
# -----------------------------------------
DB_PATH = "data/churn.db"

@st.cache_data
def load_insights_data():
    conn = sqlite3.connect(DB_PATH)

    query = """
    SELECT 
        c.customer_id,
        c.churn,
        c.tenure,
        c.senior_citizen,
        a.contract,
        a.payment_method,
        a.monthly_charges,
        a.total_charges,
        s.internet_service,
        s.tech_support,
        s.online_security
    FROM customers c
    JOIN accounts a ON c.customer_id = a.customer_id
    JOIN services s ON c.customer_id = s.customer_id;
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    df["churn"] = df["churn"].map({"Yes": 1, "No": 0})
    return df

df = load_insights_data()

# -----------------------------------------
# INSIGHT 1 — TENURE VS CHURN
# -----------------------------------------
st.markdown("## ⏳ Tenure-Based Risk Analysis")

df["tenure_bucket"] = pd.cut(
    df["tenure"],
    bins=[0, 6, 12, 24, 36, 48, 60, 72],
    labels=["0–6", "6–12", "12–24", "24–36", "36–48", "48–60", "60+"]
)

tenure_churn = df.groupby("tenure_bucket")["churn"].mean() * 100

fig, ax = plt.subplots()
tenure_churn.plot(kind="line", marker="o", ax=ax)
ax.set_ylabel("Churn Rate (%)")
ax.set_xlabel("Tenure (Months)")
ax.grid(alpha=0.3)
st.pyplot(fig)

st.info(
    "📌 **Insight:** Churn risk is highest in the first 6–12 months. "
    "Early onboarding and engagement programs are critical for retention."
)

st.markdown("---")

# -----------------------------------------
# INSIGHT 2 — CONTRACT & PAYMENT EFFECT
# -----------------------------------------
st.markdown("## 📜 Contract & Payment Behavior")

col1, col2 = st.columns(2)

with col1:
    contract_churn = df.groupby("contract")["churn"].mean() * 100

    fig, ax = plt.subplots()
    contract_churn.plot(kind="bar", ax=ax)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_title("Churn Rate by Contract Type")
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)

with col2:
    payment_churn = df.groupby("payment_method")["churn"].mean() * 100

    fig, ax = plt.subplots()
    payment_churn.plot(kind="barh", ax=ax)
    ax.set_xlabel("Churn Rate (%)")
    ax.set_title("Churn Rate by Payment Method")
    ax.grid(axis="x", alpha=0.3)
    st.pyplot(fig)

st.info(
    "📌 **Insight:** Month-to-month contracts and electronic check payments "
    "are strongly associated with higher churn risk."
)

st.markdown("---")

# -----------------------------------------
# INSIGHT 3 — SERVICE ADD-ONS IMPACT
# -----------------------------------------
st.markdown("## 🧩 Service Add-ons & Churn")

addon_cols = st.columns(2)

with addon_cols[0]:
    tech_support_churn = df.groupby("tech_support")["churn"].mean() * 100

    fig, ax = plt.subplots()
    tech_support_churn.plot(kind="bar", ax=ax)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_title("Churn by Tech Support")
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)

with addon_cols[1]:
    security_churn = df.groupby("online_security")["churn"].mean() * 100

    fig, ax = plt.subplots()
    security_churn.plot(kind="bar", ax=ax)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_title("Churn by Online Security")
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)

st.info(
    "📌 **Insight:** Customers without Tech Support or Online Security "
    "are significantly more likely to churn. Bundled services improve retention."
)

st.markdown("---")

# -----------------------------------------
# INSIGHT 4 — REVENUE RISK SEGMENTATION
# -----------------------------------------
st.markdown("## 💰 Revenue Risk Segmentation")

high_risk = df[df["churn"] == 1]

fig, ax = plt.subplots()
ax.scatter(
    high_risk["tenure"],
    high_risk["monthly_charges"],
    alpha=0.6
)
ax.set_xlabel("Tenure (Months)")
ax.set_ylabel("Monthly Charges ($)")
ax.set_title("High-Risk Customers Revenue Profile")
ax.grid(alpha=0.3)

st.pyplot(fig)

st.info(
    "📌 **Insight:** High-risk customers often have higher monthly charges "
    "and shorter tenure, representing a critical revenue retention opportunity."
)

# -----------------------------------------
# EXECUTIVE SUMMARY
# -----------------------------------------
st.markdown("""
<div class="soft-card">
<h4>🧠 Executive Summary</h4>
<ul>
<li>Early-stage customers are the most vulnerable to churn</li>
<li>Contract length is the strongest retention lever</li>
<li>Service add-ons significantly reduce churn risk</li>
<li>High-risk customers represent disproportionate revenue loss</li>
<li>Targeted retention strategies can deliver measurable ROI</li>
</ul>
</div>
""", unsafe_allow_html=True)
