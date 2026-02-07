# ===============================
# app.py — Application Shell
# ===============================

import streamlit as st

# -------------------------------
# 1️⃣ PAGE CONFIG (MUST BE FIRST)
# -------------------------------
st.set_page_config(
    page_title="Customer Churn Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------
# 2️⃣ SESSION STATE INITIALIZATION
# --------------------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# -------------------------------
# 3️⃣ SIDEBAR (PREFERENCES ONLY)
# -------------------------------
with st.sidebar:
    st.markdown("### ⚙ Preferences")
    st.session_state.dark_mode = st.toggle(
        "🌙 Dark Mode",
        value=st.session_state.dark_mode
    )

# -------------------------------
# 4️⃣ GLOBAL CSS (LIGHT / DARK)
# -------------------------------
if st.session_state.dark_mode:
    st.markdown(
        """
        <style>
        body {
            background-color: #0F172A;
            color: #E5E7EB;
        }

        .stApp {
            background-color: #0F172A;
        }

        .soft-card {
            background-color: #111827 !important;
            color: #E5E7EB;
            border: 1px solid #1F2937;
        }

        .section-card {
            background: linear-gradient(135deg, #312E81, #4338CA);
            color: white;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #020617, #020617);
            border-right: 1px solid #1E293B;
        }

        section[data-testid="stSidebar"] a {
            color: #CBD5F5;
        }

        section[data-testid="stSidebar"] a:hover {
            background: #1E293B;
            color: #A5B4FC;
        }

        section[data-testid="stSidebar"] a[aria-current="page"] {
            background: linear-gradient(135deg, #4338CA, #6366F1);
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        body {
            background-color: #F8FAFF;
            color: #0F172A;
        }

        .stApp {
            background-color: #F8FAFF;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# --------------------------------
# 5️⃣ SIDEBAR ICON ALIGN + HOVER
# --------------------------------
st.markdown(
    """
    <style>
    /* Sidebar navigation items */
    section[data-testid="stSidebar"] a {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 16px;
        margin: 6px 10px;
        border-radius: 14px;
        font-weight: 500;
        transition: all 0.25s ease;
        text-decoration: none;
    }

    section[data-testid="stSidebar"] a:hover {
        background: linear-gradient(135deg, #EEF2FF, #E0E7FF);
        color: #4F46E5;
        transform: translateX(4px);
        box-shadow: 0 6px 16px rgba(79,70,229,0.15);
    }

    section[data-testid="stSidebar"] a[aria-current="page"] {
        background: linear-gradient(135deg, #4F46E5, #6366F1);
        color: white;
        box-shadow: 0 12px 30px rgba(79,70,229,0.35);
    }

    section[data-testid="stSidebar"] a[aria-current="page"] svg {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------
# 6️⃣ MAIN LANDING CONTENT
# --------------------------------
st.title("📊 Customer Churn Analytics Platform")
st.caption(
    "An end-to-end data science application for churn prediction, insights, and business intelligence."
)

st.markdown("---")

st.info("⬅ Use the sidebar to navigate between Dashboard, Predictions, Insights, Model Info, and About pages.")

# --------------------------------
# 7️⃣ FOOTER (OPTIONAL, CLEAN)
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
