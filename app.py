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
# 4️⃣ GLOBAL CSS (LIGHT / DARK + SIDEBAR)
# -------------------------------
# Single stylesheet with CSS custom properties — toggled via body class
theme_class = "dark" if st.session_state.dark_mode else "light"
st.markdown(
    f"""
    <style>
    :root {{
        /* Light theme (default) */
        --bg-primary: #F8FAFF;
        --bg-secondary: #FFFFFF;
        --bg-sidebar: linear-gradient(180deg, #F8FAFF, #F1F5F9);
        --bg-sidebar-border: #E2E8F0;
        --text-primary: #0F172A;
        --text-secondary: #64748B;
        --text-sidebar: #334155;
        --card-bg: #FFFFFF;
        --card-border: #E2E8F0;
        --sidebar-hover-bg: linear-gradient(135deg, #EEF2FF, #E0E7FF);
        --sidebar-hover-text: #4F46E5;
        --sidebar-active-bg: linear-gradient(135deg, #4F46E5, #6366F1);
        --sidebar-active-text: white;
        --sidebar-active-shadow: rgba(79,70,229,0.35);
        --section-card-bg: linear-gradient(135deg, #312E81, #4338CA);
        --section-card-text: white;
    }}
    body.dark {{
        --bg-primary: #0F172A;
        --bg-secondary: #111827;
        --bg-sidebar: linear-gradient(180deg, #020617, #020617);
        --bg-sidebar-border: #1E293B;
        --text-primary: #E5E7EB;
        --text-secondary: #94A3B8;
        --text-sidebar: #CBD5F5;
        --card-bg: #111827;
        --card-border: #1F2937;
        --sidebar-hover-bg: linear-gradient(135deg, #312E81, #4338CA);
        --sidebar-hover-text: #C7D2FE;
        --sidebar-active-bg: linear-gradient(135deg, #4F46E5, #6366F1);
        --sidebar-active-text: white;
        --sidebar-active-shadow: rgba(79,70,229,0.35);
        --section-card-bg: linear-gradient(135deg, #312E81, #4338CA);
        --section-card-text: white;
    }}

    body {{
        background-color: var(--bg-primary);
        color: var(--text-primary);
    }}
    .stApp {{
        background-color: var(--bg-primary);
    }}
    .soft-card {{
        background-color: var(--card-bg) !important;
        color: var(--text-primary);
        border: 1px solid var(--card-border);
    }}
    .section-card {{
        background: var(--section-card-bg);
        color: var(--section-card-text);
    }}
    section[data-testid="stSidebar"] {{
        background: var(--bg-sidebar);
        border-right: 1px solid var(--bg-sidebar-border);
    }}
    section[data-testid="stSidebar"] a {{
        color: var(--text-sidebar);
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 16px;
        margin: 6px 10px;
        border-radius: 14px;
        font-weight: 500;
        transition: all 0.25s ease;
        text-decoration: none;
    }}
    section[data-testid="stSidebar"] a:hover {{
        background: var(--sidebar-hover-bg);
        color: var(--sidebar-hover-text);
        transform: translateX(4px);
        box-shadow: 0 6px 16px rgba(79,70,229,0.15);
    }}
    section[data-testid="stSidebar"] a[aria-current="page"] {{
        background: var(--sidebar-active-bg);
        color: var(--sidebar-active-text);
        box-shadow: 0 12px 30px var(--sidebar-active-shadow);
    }}
    section[data-testid="stSidebar"] a[aria-current="page"] svg {{
        color: var(--sidebar-active-text);
    }}
    </style>
    <script>
    document.body.className = '{theme_class}';
    </script>
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
