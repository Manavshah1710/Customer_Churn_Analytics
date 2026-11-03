"""
Integration and UI tests for Streamlit application pages using streamlit.testing.v1.AppTest.
Verifies all pages load without exceptions and the Predict page submits default values to return probabilities.
"""

import re

import pytest
from streamlit.testing.v1 import AppTest

from src.schema import EXAMPLE_HIGH_RISK, EXAMPLE_LOW_RISK, NUMERIC_FIELDS


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


@pytest.mark.parametrize("customer, expected", [(EXAMPLE_HIGH_RISK, "High Risk"), (EXAMPLE_LOW_RISK, "Low Risk")])
def test_predict_form_uses_the_entered_values(default_timeout, customer, expected):
    """Values typed into every field must reach the model and change the result."""
    at = AppTest.from_file("pages/2_Predict_Churn.py", default_timeout=default_timeout).run()
    inputs = {n.label: n for n in at.number_input}
    assert len(inputs) == len(NUMERIC_FIELDS)
    for name, spec in NUMERIC_FIELDS.items():
        inputs[spec["label"]].set_value(customer[name])
    at.selectbox[0].set_value(customer["asl_flag"])
    at.button[0].click().run()

    assert not at.exception, f"Predict page raised exception on submit: {at.exception}"
    assert expected in " ".join(m.value for m in at.markdown)


def test_predict_does_not_show_score_as_percent(default_timeout):
    """Scores are uncalibrated, so they must not read like a percent chance of churning."""
    at = AppTest.from_file("pages/2_Predict_Churn.py", default_timeout=default_timeout).run()
    at.button[0].click().run()

    assert not at.exception, f"Predict page raised exception on submit: {at.exception}"
    text_content = " ".join(m.value for m in at.markdown)
    assert "Churn Risk Score" in text_content
    assert not re.search(r"\d\.\d%", text_content), "Score or cutoff shown as a percentage"


def test_insights_page_loads(default_timeout):
    """Verifies that pages/3_Insights.py loads and displays risk factor analyses without exceptions."""
    at = AppTest.from_file("pages/3_Insights.py", default_timeout=default_timeout).run()
    assert not at.exception, f"Insights page raised exception: {at.exception}"


def test_model_info_page_loads(default_timeout):
    """Verifies that pages/4_Model_Info.py loads and displays model architecture and metrics without exceptions."""
    at = AppTest.from_file("pages/4_Model_Info.py", default_timeout=default_timeout).run()
    assert not at.exception, f"Model Info page raised exception: {at.exception}"


def test_monitoring_page_separates_healthy_data_from_drift():
    """With no log it asks for more data; the healthy baseline passes and simulated drift alerts."""
    at = AppTest.from_file("pages/6_Monitoring.py", default_timeout=120).run()
    assert not at.exception, f"Monitoring page failed to load: {at.exception}"
    assert len(at.info) == 1 and not at.error

    at.radio[0].set_value("Held-out test set (healthy baseline)").run()
    assert not at.exception, f"Monitoring page raised exception: {at.exception}"
    assert len(at.success) == 1 and not at.error

    at.radio[0].set_value("Simulated drift (demo)").run()
    assert not at.exception, f"Monitoring page raised exception: {at.exception}"
    assert len(at.error) == 1 and "mou_Mean" in at.error[0].value


def test_about_page_loads(default_timeout):
    """Verifies that pages/5_About.py loads documentation without exceptions."""
    at = AppTest.from_file("pages/5_About.py", default_timeout=default_timeout).run()
    assert not at.exception, f"About page raised exception: {at.exception}"
