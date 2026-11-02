"""
Shared page setup for the Streamlit app.
"""

import streamlit as st


def apply_page_style() -> None:
    """
    Sets page config and shared CSS. Call first on every page:
    Streamlit runs only the selected page's script, never app.py as well.
    Colors come from the theme in .streamlit/config.toml.
    """
    st.set_page_config(
        page_title="Customer Churn Analytics Platform",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .soft-card, .section-card {
            padding: 16px 20px;
            border-radius: 14px;
            margin-bottom: 16px;
        }
        .soft-card {
            background-color: #112240;
            border: 1px solid #233554;
        }
        .section-card {
            background: linear-gradient(135deg, #312E81, #4338CA);
            color: white;
        }
        section[data-testid="stSidebar"] a {
            color: #CBD5F5;
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
            background: linear-gradient(135deg, #312E81, #4338CA);
            color: #C7D2FE;
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
        unsafe_allow_html=True,
    )
