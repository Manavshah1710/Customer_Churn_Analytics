# =========================================
# pages/1_Dashboard.py — Analytics Dashboard
# =========================================

import streamlit as st
import matplotlib.pyplot as plt

from src.preprocess import churn_rate_by_band, load_customer_columns
from src.ui import apply_page_style

apply_page_style()

# -----------------------------------------
# PAGE HEADER
# -----------------------------------------
st.markdown("""
<h2 style="margin-bottom:4px;">Analytics Dashboard</h2>
<p style="color:#64748b; margin-top:0;">
Customer base and churn patterns in the 100,000-customer dataset
</p>
""", unsafe_allow_html=True)


# -----------------------------------------
# DATA LOADING (CACHED)
# -----------------------------------------
@st.cache_data
def load_dashboard_data():
    return load_customer_columns(["churn", "months", "eqpdays", "mou_Mean", "change_mou", "rev_Mean"])


df = load_dashboard_data()

# -----------------------------------------
# KPI SECTION
# -----------------------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric("Total Customers", f"{df.shape[0]:,}")

with kpi2:
    st.metric("Churn in Sample", f"{df['churn'].mean() * 100:.1f}%")

with kpi3:
    st.metric("Median Tenure", f"{df['months'].median():.0f} months")

with kpi4:
    st.metric("Average Monthly Revenue", f"${df['rev_Mean'].mean():.2f}")

st.caption(
    "This dataset is almost evenly split between churned and retained customers, "
    "so its churn rate should not be read as a real company's churn rate."
)

st.markdown("---")

# -----------------------------------------
# CHART SECTION (2x2 GRID)
# -----------------------------------------
col1, col2 = st.columns(2)

tenure = churn_rate_by_band(
    df, "months",
    bins=[0, 9, 10, 12, 18, 24, 36, 100],
    labels=["Up to 9", "10", "11-12", "13-18", "19-24", "25-36", "37+"],
)

with col1:
    st.markdown("### Churn by Tenure")
    st.caption("Churn rate across the customer lifecycle")

    fig, ax = plt.subplots()
    tenure.plot(ax=ax, marker="o")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("Tenure (Months)")
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

handset_age = churn_rate_by_band(
    df, "eqpdays",
    bins=[-10, 90, 180, 270, 365, 730, 4000],
    labels=["0-90", "91-180", "181-270", "271-365", "366-730", "731+"],
)

with col2:
    st.markdown("### Churn by Handset Age")
    st.caption("Days since the current handset was issued")

    fig, ax = plt.subplots()
    handset_age.plot(kind="bar", ax=ax)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("Handset Age (Days)")
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

st.markdown("---")

col3, col4 = st.columns(2)

minutes = churn_rate_by_band(
    df, "mou_Mean",
    bins=[0, 100, 250, 500, 1000, 20000],
    labels=["0-100", "101-250", "251-500", "501-1000", "1001+"],
)

with col3:
    st.markdown("### Churn by Monthly Minutes")
    st.caption("Average monthly minutes of use")

    fig, ax = plt.subplots()
    minutes.plot(kind="bar", ax=ax)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("Average Monthly Minutes")
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

usage_change = churn_rate_by_band(
    df, "change_mou",
    bins=[-5000, -300, -100, -20, 20, 100, 300, 50000],
    labels=["Below -300", "-300 to -100", "-100 to -20", "-20 to 20", "20 to 100", "100 to 300", "Above 300"],
)

with col4:
    st.markdown("### Churn by Change in Usage")
    st.caption("Recent change in monthly minutes")

    fig, ax = plt.subplots()
    usage_change.plot(kind="barh", ax=ax)
    ax.set_xlabel("Churn Rate (%)")
    ax.set_ylabel("Change in Monthly Minutes")
    ax.grid(axis="x", alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

# -----------------------------------------
# DASHBOARD INSIGHT SUMMARY
# -----------------------------------------
st.markdown(f"""
<div class="soft-card">
<h4>📌 What the Data Shows</h4>
<ul>
<li><b>Tenure:</b> churn is {tenure['Up to 9']:.0f}% up to month 9 and {tenure['11-12']:.0f}% at months 11-12</li>
<li><b>Handset age:</b> churn is {handset_age['181-270']:.0f}% at 181-270 days and {handset_age['271-365']:.0f}% at 271-365 days</li>
<li><b>Usage level:</b> churn is {minutes['0-100']:.0f}% for the lightest users and {minutes['1001+']:.0f}% for the heaviest</li>
<li><b>Falling usage:</b> churn is {usage_change['Below -300']:.0f}% after a large drop in minutes</li>
</ul>
</div>
""", unsafe_allow_html=True)
