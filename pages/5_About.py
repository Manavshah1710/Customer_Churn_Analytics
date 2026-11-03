# =========================================
# pages/5_About.py — About the Project
# =========================================

import streamlit as st

from src.ui import apply_page_style

apply_page_style()

# -----------------------------------------
# PAGE HEADER
# -----------------------------------------
st.markdown("""
<h2 style="margin-bottom:6px;">About This Project</h2>
<p style="color:#64748b; margin-top:0;">
End-to-end customer churn analytics portfolio project
</p>
""", unsafe_allow_html=True)

# -----------------------------------------
# PROJECT OVERVIEW
# -----------------------------------------
st.markdown("""
<div class="section-card">
<h3>🚀 Project Overview</h3>
<p style="font-size:16px; line-height:1.7;">
Customer churn is one of the most critical challenges for subscription-based businesses.
Acquiring new customers is significantly more expensive than retaining existing ones.
This project focuses on building an <b>end-to-end churn prediction system</b>
that enables businesses to identify at-risk customers early and take proactive retention actions.
</p>

<p style="font-size:16px; line-height:1.7;">
The project integrates <b>data engineering, machine learning, analytics, and modern UI design</b>
into a single application that supports both operational and strategic decision-making.
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------------------
# PROBLEM STATEMENT
# -----------------------------------------
st.markdown("""
<div class="soft-card">
<h3>🎯 Business Problem</h3>
<ul>
<li>High customer churn directly impacts revenue and growth</li>
<li>Reactive retention strategies are often too late</li>
<li>Businesses lack visibility into churn drivers</li>
<li>Decision-makers need interpretable and actionable insights</li>
</ul>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------
# SOLUTION APPROACH
# -----------------------------------------
st.markdown("""
<div class="soft-card">
<h3>🧠 Solution Approach</h3>
<ul>
<li>Loaded 100,000 customer records into SQLite and split them before any preprocessing</li>
<li>Reduced 77 candidate columns to 15 using cross-validated permutation importance</li>
<li>Excluded demographic and household columns for fairness</li>
<li>Compared models with cross-validation and chose gradient boosting</li>
<li>Explained each score with SHAP and served predictions through FastAPI</li>
<li>Added data drift monitoring with Evidently AI</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------------------
# KEY FEATURES
# -----------------------------------------
st.markdown("""
<div class="soft-card">
<h3>✨ Key Features</h3>
<ul>
<li><b>Interactive Dashboard:</b> KPIs, churn trends, and customer segmentation</li>
<li><b>Churn Prediction:</b> Real-time risk scores for individual customers</li>
<li><b>Business Insights:</b> Clear explanations of churn drivers</li>
<li><b>Model Transparency:</b> Held-out metrics and SHAP feature importance</li>
<li><b>Monitoring:</b> Data drift checks with a threshold alert</li>
</ul>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------
# TECHNOLOGY STACK
# -----------------------------------------
st.markdown("""
<div class="soft-card">
<h3>🧰 Technology Stack</h3>
<ul>
<li><b>Languages:</b> Python, SQL</li>
<li><b>Data & ML:</b> Pandas, NumPy, Scikit-learn, SHAP</li>
<li><b>Visualization:</b> Matplotlib, Seaborn</li>
<li><b>Application:</b> Streamlit, FastAPI</li>
<li><b>MLOps:</b> MLflow, Evidently AI, Docker, GitHub Actions</li>
<li><b>Storage:</b> SQLite</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------------------
# BUSINESS IMPACT
# -----------------------------------------
st.markdown("""
<div class="soft-card">
<h3>📊 Results</h3>
<ul>
<li>Test AUC of 0.683, with a cross-validated AUC of 0.684</li>
<li>About 77% of the customers flagged in the top 10% churned, against 50% overall</li>
<li>Cutting 77 candidate columns to 15 cost about 0.001 AUC, which keeps the model practical to use</li>
<li>Results come from a public dataset, not a live business deployment</li>
</ul>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------
# KEY LEARNINGS
# -----------------------------------------
st.markdown("""
<div class="soft-card">
<h3>📘 Key Learnings</h3>
<ul>
<li>Churn jumps sharply around month 11 in this dataset</li>
<li>Usage, handset age, and tenure carry the most signal</li>
<li>A published data dictionary still has to be checked against the data</li>
<li>Honest metrics on hard data are worth more than high scores on easy data</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------------------
# CLOSING NOTE
# -----------------------------------------
st.markdown("""
<div style="
margin-top:20px;
padding:24px;
border-radius:18px;
background:linear-gradient(135deg,#0F172A,#1E293B);
color:#E5E7EB;
text-align:center;
">
<p style="font-size:16px;">
This project covers the lifecycle of a data science solution,
from raw data to a containerized API and a user-friendly analytics app.
</p>
<p>
<b>Dataset: Telecom Customer Churn - 100k Cleaned Records (Kaggle), CC BY-NC-SA 4.0. Non-commercial use only.</b>
</p>
</div>
""", unsafe_allow_html=True)
