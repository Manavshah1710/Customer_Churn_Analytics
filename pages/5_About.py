# =========================================
# pages/5_About.py — About the Project
# =========================================

import streamlit as st

# -----------------------------------------
# PAGE HEADER
# -----------------------------------------
st.markdown("""
<h2 style="margin-bottom:6px;">About This Project</h2>
<p style="color:#64748b; margin-top:0;">
End-to-end customer churn analytics platform designed for real-world business impact
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
This project focuses on building a <b>production-ready churn prediction system</b>
that enables businesses to identify at-risk customers early and take proactive retention actions.
</p>

<p style="font-size:16px; line-height:1.7;">
The platform integrates <b>data engineering, machine learning, analytics, and modern UI design</b>
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
<li>Structured customer data using a relational SQL model</li>
<li>Performed exploratory data analysis to uncover churn drivers</li>
<li>Engineered predictive features from behavioral and service data</li>
<li>Trained and evaluated multiple ML models</li>
<li>Selected Random Forest for optimal recall and AUC</li>
<li>Delivered insights through an interactive Streamlit application</li>
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
<li><b>Churn Prediction:</b> Real-time probability estimates for individual customers</li>
<li><b>Business Insights:</b> Clear explanations of churn drivers</li>
<li><b>Model Transparency:</b> ROC curve, AUC, and feature importance</li>
<li><b>Modern UI:</b> Premium design with dark mode support</li>
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
<li><b>Data & ML:</b> Pandas, NumPy, Scikit-learn</li>
<li><b>Visualization:</b> Matplotlib, Seaborn</li>
<li><b>Application:</b> Streamlit</li>
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
<h3>💼 Business Impact</h3>
<ul>
<li>Identified high-risk customers before churn occurs</li>
<li>Enabled targeted retention strategies</li>
<li>Improved decision-making using data-driven insights</li>
<li>Estimated significant revenue retention opportunity</li>
<li>Reduced reliance on reactive customer support interventions</li>
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
<li>Early customer lifecycle stages are the most vulnerable</li>
<li>Contract length is the strongest churn predictor</li>
<li>Service add-ons significantly reduce churn probability</li>
<li>Model interpretability is as important as accuracy</li>
<li>UX design greatly improves analytics adoption</li>
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
This project demonstrates the complete lifecycle of a data science solution —
from raw data to deployed, user-friendly analytics.
</p>
<p>
<b>Designed for scalability, interpretability, and real-world business value.</b>
</p>
</div>
""", unsafe_allow_html=True)
