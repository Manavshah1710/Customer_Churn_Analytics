# =========================================
# pages/1_Dashboard.py — Analytics Dashboard
# =========================================

import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

# -----------------------------------------
# PAGE HEADER
# -----------------------------------------
st.markdown("""
<h2 style="margin-bottom:4px;">Analytics Dashboard</h2>
<p style="color:#64748b; margin-top:0;">
Real-time customer churn insights and key business metrics
</p>
""", unsafe_allow_html=True)

# -----------------------------------------
# DATA LOADING (CACHED)
# -----------------------------------------
DB_PATH = "data/churn.db"

@st.cache_data
def load_dashboard_data():
    conn = sqlite3.connect(DB_PATH)

    query = """
    SELECT 
        c.customer_id,
        c.churn,
        c.tenure,
        a.contract,
        a.monthly_charges,
        s.internet_service
    FROM customers c
    JOIN accounts a ON c.customer_id = a.customer_id
    JOIN services s ON c.customer_id = s.customer_id;
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    df["churn"] = df["churn"].map({"Yes": 1, "No": 0})
    return df

df = load_dashboard_data()

# -----------------------------------------
# KPI CALCULATIONS
# -----------------------------------------
total_customers = df.shape[0]
churn_rate = df["churn"].mean() * 100
high_risk_customers = df[df["churn"] == 1].shape[0]

avg_monthly_charge = df["monthly_charges"].mean()
retention_value = high_risk_customers * avg_monthly_charge * 12

# -----------------------------------------
# KPI SECTION
# -----------------------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric("Total Customers", f"{total_customers:,}")

with kpi2:
    st.metric("Churn Rate", f"{churn_rate:.1f}%")

with kpi3:
    st.metric("High-Risk Customers", f"{high_risk_customers:,}")

with kpi4:
    st.metric("Retention Opportunity", f"${retention_value/1e6:.2f}M")

st.markdown("---")

# -----------------------------------------
# CHART SECTION (2x2 GRID)
# -----------------------------------------
col1, col2 = st.columns(2)

# ---- Churn Trend by Tenure ----
df["tenure_bucket"] = pd.cut(
    df["tenure"],
    bins=[0, 6, 12, 24, 36, 48, 60, 72],
    labels=["0–6", "6–12", "12–24", "24–36", "36–48", "48–60", "60+"]
)

trend = df.groupby("tenure_bucket")["churn"].mean() * 100

with col1:
    st.markdown("### Churn Trend by Tenure")
    st.caption("Churn rate across customer lifecycle stages")

    fig, ax = plt.subplots()
    trend.plot(ax=ax, marker="o")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("Tenure (Months)")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

# ---- Churn by Contract Type ----
contract_churn = df.groupby(["contract", "churn"]).size().unstack()

with col2:
    st.markdown("### Churn by Contract Type")
    st.caption("Month-to-month contracts exhibit higher churn")

    fig, ax = plt.subplots()
    contract_churn.plot(kind="bar", ax=ax)
    ax.set_ylabel("Customers")
    ax.legend(["Retained", "Churned"])
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)

st.markdown("---")

# -----------------------------------------
# SECOND ROW CHARTS
# -----------------------------------------
col3, col4 = st.columns(2)

# ---- Churn Distribution ----
churn_counts = df["churn"].value_counts()

with col3:
    st.markdown("### Churn Distribution")
    st.caption("Overall churn vs retained customer split")

    fig, ax = plt.subplots()
    ax.pie(
        churn_counts,
        labels=["Retained", "Churned"],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(width=0.4)
    )
    ax.axis("equal")
    st.pyplot(fig)

# ---- Churn by Internet Service ----
service_churn = df.groupby(["internet_service", "churn"]).size().unstack()

with col4:
    st.markdown("### Churn by Internet Service")
    st.caption("Fiber optic customers show elevated churn risk")

    fig, ax = plt.subplots()
    service_churn.plot(kind="barh", ax=ax)
    ax.set_xlabel("Customers")
    ax.legend(["Retained", "Churned"])
    ax.grid(axis="x", alpha=0.3)
    st.pyplot(fig)

# -----------------------------------------
# DASHBOARD INSIGHT SUMMARY
# -----------------------------------------
st.markdown("""
<div class="soft-card">
<h4>📌 Key Dashboard Insights</h4>
<ul>
<li><b>Early churn risk:</b> Customers in their first year are most likely to churn</li>
<li><b>Contract stability:</b> Long-term contracts significantly reduce churn</li>
<li><b>Service quality:</b> Fiber optic users require proactive engagement</li>
<li><b>Revenue impact:</b> High-risk customers represent major retention opportunity</li>
</ul>
</div>
""", unsafe_allow_html=True)
