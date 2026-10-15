"""
Integration and UI tests for Streamlit application pages using streamlit.testing.v1.AppTest.
Verifies all pages load without exceptions and the Predict page submits default values to return probabilities.
"""

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture
def default_timeout():
    return 30


def test_app_shell_loads(default_timeout):
    """Verifies that the main application shell (app.py) loads without exceptions."""
    at = AppTest.from_file("app.py", default_timeout=default_timeout).run()
    assert not at.exception, f"app.py raised exception: {at.exception}"


def test_dashboard_page_loads(default_timeout):
    """Verifies that pages/1_Dashboard.py loads and computes executive KPIs without exceptions."""
    at = AppTest.from_file("pages/1_Dashboard.py", default_timeout=default_timeout).run()
    assert not at.exception, f"Dashboard page raised exception: {at.exception}"


def test_predict_churn_page_loads_and_submits(default_timeout):
    """
    Verifies that pages/2_Predict_Churn.py:
    1. Loads cleanly without exceptions.
    2. Submits default customer input values via the predict button.
    3. Successfully calculates and renders a churn probability without exceptions.
    """
    at = AppTest.from_file("pages/2_Predict_Churn.py", default_timeout=default_timeout).run()
    assert not at.exception, f"Predict page failed to load: {at.exception}"

    # Click the form submission button with defaults
    assert len(at.button) > 0, "No submit button found on Predict Churn page"
    at.button[0].click().run()

    assert not at.exception, f"Predict page raised exception on submit: {at.exception}"

    # Verify that prediction output rendered
    text_content = " ".join([m.value for m in at.markdown])
    assert "%" in text_content or "Risk" in text_content or "Score" in text_content, (
        "Expected churn probability or risk assessment in page output after submission."
    )


def test_insights_page_loads(default_timeout):
    """Verifies that pages/3_Insights.py loads and displays risk factor analyses without exceptions."""
    at = AppTest.from_file("pages/3_Insights.py", default_timeout=default_timeout).run()
    assert not at.exception, f"Insights page raised exception: {at.exception}"


def test_model_info_page_loads(default_timeout):
    """Verifies that pages/4_Model_Info.py loads and displays model architecture and metrics without exceptions."""
    at = AppTest.from_file("pages/4_Model_Info.py", default_timeout=default_timeout).run()
    assert not at.exception, f"Model Info page raised exception: {at.exception}"


def test_about_page_loads(default_timeout):
    """Verifies that pages/5_About.py loads documentation without exceptions."""
    at = AppTest.from_file("pages/5_About.py", default_timeout=default_timeout).run()
    assert not at.exception, f"About page raised exception: {at.exception}"
