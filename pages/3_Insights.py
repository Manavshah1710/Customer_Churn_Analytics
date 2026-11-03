# =========================================
# pages/3_Insights.py — Business Insights
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
<h2 style="margin-bottom:4px;">Customer Insights</h2>
<p style="color:#64748b; margin-top:0;">
Patterns measured in this dataset. They show association, not proof of cause.
</p>
""", unsafe_allow_html=True)


# -----------------------------------------
# DATA LOADING (CACHED)
# -----------------------------------------
@st.cache_data
def load_insights_data():
    return load_customer_columns(["churn", "months", "eqpdays", "hnd_price", "drop_vce_Mean", "asl_flag"])


df = load_insights_data()

# -----------------------------------------
# INSIGHT 1 — THE ONE-YEAR JUMP
# -----------------------------------------
st.markdown("## ⏳ The One-Year Jump")

first_two_years = df[df["months"] <= 24].groupby("months")["churn"].mean() * 100

fig, ax = plt.subplots(figsize=(9, 3.5))
first_two_years.plot(kind="line", marker="o", ax=ax)
ax.set_ylabel("Churn Rate (%)")
ax.set_xlabel("Tenure (Months)")
ax.grid(alpha=0.3)
st.pyplot(fig)
plt.close(fig)

early = df[df["months"].between(7, 9)]["churn"].mean() * 100
st.info(
    f"📌 **Insight:** Churn is {early:.0f}% at months 7-9, then {first_two_years[10]:.0f}% at month 10 and "
    f"{first_two_years[11]:.0f}% at month 11. This fits contracts ending near one year, but the dataset does not "
    "record contract terms. If that is the cause, retention outreach should start before month 10."
)

st.markdown("---")

# -----------------------------------------
# INSIGHT 2 — HANDSETS
# -----------------------------------------
st.markdown("## 📱 Handset Age and Price")

col1, col2 = st.columns(2)

handset_age = churn_rate_by_band(
    df, "eqpdays",
    bins=[-10, 90, 180, 270, 365, 730, 4000],
    labels=["0-90", "91-180", "181-270", "271-365", "366-730", "731+"],
)
handset_price = churn_rate_by_band(
    df, "hnd_price",
    bins=[0, 30, 60, 100, 150, 2000],
    labels=["Up to $30", "$31-60", "$61-100", "$101-150", "Over $150"],
)

with col1:
    fig, ax = plt.subplots()
    handset_age.plot(kind="bar", ax=ax)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("Handset Age (Days)")
    ax.set_title("Churn by Handset Age")
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

with col2:
    fig, ax = plt.subplots()
    handset_price.plot(kind="bar", ax=ax)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("Handset Price")
    ax.set_title("Churn by Handset Price")
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

st.info(
    f"📌 **Insight:** Churn moves from {handset_age['181-270']:.0f}% at 181-270 days to "
    f"{handset_age['271-365']:.0f}% at 271-365 days. Customers on the cheapest handsets churn at "
    f"{handset_price['Up to $30']:.0f}%, against {handset_price['Over $150']:.0f}% on the most expensive."
)

st.markdown("---")

# -----------------------------------------
# INSIGHT 3 — WHAT MATTERS LESS THAN EXPECTED
# -----------------------------------------
st.markdown("## 📶 Dropped Calls and Spending Limits")

col3, col4 = st.columns(2)

dropped_calls = churn_rate_by_band(
    df, "drop_vce_Mean",
    bins=[0, 2, 5, 10, 20, 500],
    labels=["0-2", "3-5", "6-10", "11-20", "21+"],
)
spending_limit = df.groupby("asl_flag")["churn"].mean() * 100

with col3:
    fig, ax = plt.subplots()
    dropped_calls.plot(kind="bar", ax=ax)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("Dropped Calls per Month")
    ax.set_title("Churn by Dropped Calls")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

with col4:
    fig, ax = plt.subplots()
    spending_limit.rename({"N": "No limit", "Y": "Has limit"}).plot(kind="bar", ax=ax)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("Account Spending Limit")
    ax.set_title("Churn by Spending Limit")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

st.info(
    f"📌 **Insight:** Dropped calls barely separate customers on their own: churn stays between "
    f"{dropped_calls.min():.0f}% and {dropped_calls.max():.0f}% across every band. Accounts with a spending limit "
    f"churn at {spending_limit['Y']:.0f}%, against {spending_limit['N']:.0f}% without one."
)

# -----------------------------------------
# SUMMARY
# -----------------------------------------
st.markdown(f"""
<div class="soft-card">
<h4>🧠 Summary</h4>
<ul>
<li>The sharpest pattern is by tenure: month {first_two_years.idxmax()} has the highest churn in the first two years, at {first_two_years.max():.0f}%</li>
<li>Older and cheaper handsets are associated with more churn</li>
<li>Network quality, measured by dropped calls, shows little link to churn on its own</li>
<li>These figures come from an evenly split sample, so compare groups rather than reading them as real churn rates</li>
</ul>
</div>
""", unsafe_allow_html=True)
