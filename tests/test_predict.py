"""
Unit and integration tests for src/predict.py inference module.
"""

import numpy as np
import pandas as pd
import pytest

from src.predict import load_artifacts, predict_churn
from src.preprocess import load_data


def test_load_artifacts():
    """Verifies that load_artifacts properly loads the trained model and preprocessor dictionary."""
    model, preprocessor = load_artifacts()

    assert hasattr(model, "predict_proba"), "Model object does not have predict_proba method."
    assert isinstance(preprocessor, dict), "Preprocessor artifact is not a dictionary."
    for key in ["preprocessor", "feature_columns", "train_medians", "cat_cols", "num_cols"]:
        assert key in preprocessor, f"Missing key '{key}' in preprocessor artifact."
    assert hasattr(preprocessor["preprocessor"], "transform"), "preprocessor['preprocessor'] must have transform method."


def test_predict_churn_single_row_from_db():
    """End-to-end test calling predict_churn on a real raw record loaded from SQLite DB."""
    df = load_data("data/churn.db")
    assert not df.empty, "Database table is empty."

    raw_row_dict = df.iloc[0].to_dict()
    prob = predict_churn(raw_row_dict)

    assert isinstance(prob, float), f"Expected float, got {type(prob)}"
    assert 0.0 <= prob <= 1.0, f"Predicted probability {prob} is outside [0.0, 1.0]"


def test_predict_churn_batch_dataframe():
    """Verifies predict_churn works on multi-row DataFrames and returns numpy array."""
    df = load_data("data/churn.db").head(5)
    probs = predict_churn(df)

    assert isinstance(probs, np.ndarray), f"Expected np.ndarray, got {type(probs)}"
    assert len(probs) == 5
    assert np.all((probs >= 0.0) & (probs <= 1.0)), "Batch probabilities outside [0.0, 1.0]"


def test_customer_risk_profile_ordering():
    """
    Business sanity test:
    (a) High risk: New month-to-month fiber-optic customer with electronic check & no add-ons.
    (b) Low risk: 60-month two-year-contract DSL customer with several add-ons.
    Asserts that (a) scores significantly higher probability than (b).
    """
    customer_a = {
        "gender": "Female",
        "senior_citizen": 0,
        "partner": "No",
        "dependents": "No",
        "tenure": 1,
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 85.0,
        "total_charges": 85.0,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "No",
        "streaming_movies": "No",
    }

    customer_b = {
        "gender": "Male",
        "senior_citizen": 0,
        "partner": "Yes",
        "dependents": "Yes",
        "tenure": 60,
        "contract": "Two year",
        "paperless_billing": "No",
        "payment_method": "Credit card (automatic)",
        "monthly_charges": 60.0,
        "total_charges": 3600.0,
        "phone_service": "Yes",
        "multiple_lines": "Yes",
        "internet_service": "DSL",
        "online_security": "Yes",
        "online_backup": "Yes",
        "device_protection": "Yes",
        "tech_support": "Yes",
        "streaming_tv": "No",
        "streaming_movies": "No",
    }

    prob_a = predict_churn(customer_a)
    prob_b = predict_churn(customer_b)

    assert prob_a > prob_b, (
        f"High risk customer score ({prob_a:.4f}) should be higher than low risk ({prob_b:.4f})"
    )
    assert prob_a - prob_b > 0.30, (
        f"Expected clear separation (>30%) between risk profiles, got diff {prob_a - prob_b:.4f}"
    )
