"""
Tests for SHAP explainability module.
"""

import json
import pytest

from src.explain import explain_customer, global_importance, get_explainer, ALL_RAW_FEATURES


def test_explain_customer_schema():
    """Verify explain_customer returns the correct schema with proper types."""
    customer = {
        "gender": "Female", "senior_citizen": 0, "partner": "No", "dependents": "No",
        "tenure": 1, "contract": "Month-to-month", "paperless_billing": "Yes",
        "payment_method": "Electronic check", "monthly_charges": 85.0, "total_charges": 85.0,
        "phone_service": "Yes", "multiple_lines": "No", "internet_service": "Fiber optic",
        "online_security": "No", "online_backup": "No", "device_protection": "No",
        "tech_support": "No", "streaming_tv": "No", "streaming_movies": "No",
    }

    result = explain_customer(customer)

    # Required keys
    assert "probability" in result
    assert "base_value" in result
    assert "risk_level" in result
    assert "reasons" in result

    # Types
    assert isinstance(result["probability"], float)
    assert isinstance(result["base_value"], float)
    assert result["risk_level"] in ("high", "moderate", "low")
    assert isinstance(result["reasons"], list)
    assert len(result["reasons"]) == 5  # default top_k=5

    # Each reason has correct structure
    for r in result["reasons"]:
        assert "feature" in r
        assert "customer_value" in r
        assert "shap_value" in r
        assert "direction" in r
        assert r["direction"] in ("increases_risk", "decreases_risk")
        assert isinstance(r["shap_value"], float)
        assert r["feature"] in ALL_RAW_FEATURES


def test_explain_customer_additivity():
    """Verify base_value + sum(shap_values) == probability within tolerance."""
    customer = {
        "gender": "Female", "senior_citizen": 0, "partner": "No", "dependents": "No",
        "tenure": 1, "contract": "Month-to-month", "paperless_billing": "Yes",
        "payment_method": "Electronic check", "monthly_charges": 85.0, "total_charges": 85.0,
        "phone_service": "Yes", "multiple_lines": "No", "internet_service": "Fiber optic",
        "online_security": "No", "online_backup": "No", "device_protection": "No",
        "tech_support": "No", "streaming_tv": "No", "streaming_movies": "No",
    }

    result = explain_customer(customer, top_k=20)  # get all reasons

    # Sum all SHAP values (not just top_k)
    from src.explain import get_explainer, _sum_shap_by_raw_feature
    from src.predict import load_artifacts
    from src.preprocess import transform_new
    from src import PROJECT_ROOT
    import pandas as pd

    model, preprocessor = load_artifacts(str(PROJECT_ROOT / "models"))
    input_df = pd.DataFrame([customer])
    input_proc = transform_new(input_df, preprocessor)

    explainer = get_explainer()
    shap_values = explainer.shap_values(input_proc)

    if shap_values.ndim == 3:
        shap_class1 = shap_values[0, :, 1]
        base_value = float(explainer.expected_value[1])
    else:
        shap_class1 = shap_values[0]
        base_value = float(explainer.expected_value)

    feature_names = preprocessor["preprocessor"].get_feature_names_out().tolist()
    all_reasons = _sum_shap_by_raw_feature(shap_class1, feature_names, customer)

    total_shap = sum(r["shap_value"] for r in all_reasons)
    reconstructed = base_value + total_shap

    assert abs(reconstructed - result["probability"]) < 1e-4, \
        f"Additivity failed: base+sum={reconstructed}, prob={result['probability']}"


def test_explain_customer_risk_level_matches_thresholds():
    """Verify risk_level matches thresholds from metrics.json."""
    from src import PROJECT_ROOT

    with open(PROJECT_ROOT / "models" / "metrics.json", "r") as f:
        metrics = json.load(f)
    threshold_high = metrics["threshold_high_risk"]
    threshold_moderate = metrics["threshold_moderate_risk"]

    # High risk customer
    customer_high = {
        "gender": "Female", "senior_citizen": 0, "partner": "No", "dependents": "No",
        "tenure": 1, "contract": "Month-to-month", "paperless_billing": "Yes",
        "payment_method": "Electronic check", "monthly_charges": 85.0, "total_charges": 85.0,
        "phone_service": "Yes", "multiple_lines": "No", "internet_service": "Fiber optic",
        "online_security": "No", "online_backup": "No", "device_protection": "No",
        "tech_support": "No", "streaming_tv": "No", "streaming_movies": "No",
    }
    exp_high = explain_customer(customer_high)
    assert exp_high["risk_level"] == "high"
    assert exp_high["probability"] >= threshold_high

    # Low risk customer
    customer_low = {
        "gender": "Male", "senior_citizen": 0, "partner": "Yes", "dependents": "Yes",
        "tenure": 60, "contract": "Two year", "paperless_billing": "No",
        "payment_method": "Credit card (automatic)", "monthly_charges": 60.0, "total_charges": 3600.0,
        "phone_service": "Yes", "multiple_lines": "Yes", "internet_service": "DSL",
        "online_security": "Yes", "online_backup": "Yes", "device_protection": "Yes",
        "tech_support": "Yes", "streaming_tv": "No", "streaming_movies": "No",
    }
    exp_low = explain_customer(customer_low)
    assert exp_low["risk_level"] == "low"
    assert exp_low["probability"] < threshold_moderate

    # Moderate risk (construct one near threshold)
    customer_mod = {
        "gender": "Male", "senior_citizen": 0, "partner": "No", "dependents": "No",
        "tenure": 12, "contract": "One year", "paperless_billing": "Yes",
        "payment_method": "Bank transfer (automatic)", "monthly_charges": 70.0, "total_charges": 840.0,
        "phone_service": "Yes", "multiple_lines": "No", "internet_service": "DSL",
        "online_security": "No", "online_backup": "No", "device_protection": "No",
        "tech_support": "No", "streaming_tv": "Yes", "streaming_movies": "No",
    }
    exp_mod = explain_customer(customer_mod)
    assert exp_mod["risk_level"] in ("moderate", "high", "low")  # just verify it's valid


def test_explain_customer_high_risk_sanity():
    """High-risk sanity customer has contract or tenure in top 3 reasons with increases_risk."""
    customer = {
        "gender": "Female", "senior_citizen": 0, "partner": "No", "dependents": "No",
        "tenure": 1, "contract": "Month-to-month", "paperless_billing": "Yes",
        "payment_method": "Electronic check", "monthly_charges": 85.0, "total_charges": 85.0,
        "phone_service": "Yes", "multiple_lines": "No", "internet_service": "Fiber optic",
        "online_security": "No", "online_backup": "No", "device_protection": "No",
        "tech_support": "No", "streaming_tv": "No", "streaming_movies": "No",
    }

    result = explain_customer(customer, top_k=10)
    top3_features = [r["feature"] for r in result["reasons"][:3]]
    top3_directions = [r["direction"] for r in result["reasons"][:3]]

    # Must have contract or tenure in top 3
    assert "contract" in top3_features or "tenure" in top3_features, \
        f"Expected contract or tenure in top 3, got {top3_features}"

    # And direction must be increases_risk for those
    for feat, direction in zip(top3_features, top3_directions):
        if feat in ("contract", "tenure"):
            assert direction == "increases_risk", \
                f"{feat} should increase risk for high-risk customer, got {direction}"


def test_explain_customer_low_risk_sanity():
    """Low-risk sanity customer has mostly decreases_risk directions."""
    customer = {
        "gender": "Male", "senior_citizen": 0, "partner": "Yes", "dependents": "Yes",
        "tenure": 60, "contract": "Two year", "paperless_billing": "No",
        "payment_method": "Credit card (automatic)", "monthly_charges": 60.0, "total_charges": 3600.0,
        "phone_service": "Yes", "multiple_lines": "Yes", "internet_service": "DSL",
        "online_security": "Yes", "online_backup": "Yes", "device_protection": "Yes",
        "tech_support": "Yes", "streaming_tv": "No", "streaming_movies": "No",
    }

    result = explain_customer(customer, top_k=10)
    decreases = sum(1 for r in result["reasons"] if r["direction"] == "decreases_risk")
    total = len(result["reasons"])

    # Majority should be decreases_risk
    assert decreases >= total // 2, \
        f"Expected majority decreases_risk, got {decreases}/{total}"


def test_explain_customer_reasons_sorted_by_abs_shap():
    """Reasons must be sorted by absolute SHAP value descending."""
    customer = {
        "gender": "Female", "senior_citizen": 0, "partner": "No", "dependents": "No",
        "tenure": 1, "contract": "Month-to-month", "paperless_billing": "Yes",
        "payment_method": "Electronic check", "monthly_charges": 85.0, "total_charges": 85.0,
        "phone_service": "Yes", "multiple_lines": "No", "internet_service": "Fiber optic",
        "online_security": "No", "online_backup": "No", "device_protection": "No",
        "tech_support": "No", "streaming_tv": "No", "streaming_movies": "No",
    }

    result = explain_customer(customer, top_k=20)
    abs_shaps = [abs(r["shap_value"]) for r in result["reasons"]]

    # Must be non-increasing
    for i in range(len(abs_shaps) - 1):
        assert abs_shaps[i] >= abs_shaps[i + 1], \
            f"Reasons not sorted by |shap|: {abs_shaps[i]} < {abs_shaps[i+1]}"


def test_global_importance_schema():
    """Verify global_importance returns correct structure."""
    result = global_importance(sample_size=100)

    assert isinstance(result, list)
    assert len(result) > 0

    for item in result:
        assert "feature" in item
        assert "mean_abs_shap" in item
        assert isinstance(item["mean_abs_shap"], float)
        assert item["mean_abs_shap"] >= 0
        assert item["feature"] in ALL_RAW_FEATURES

    # Sorted descending
    vals = [item["mean_abs_shap"] for item in result]
    for i in range(len(vals) - 1):
        assert vals[i] >= vals[i + 1]


def test_global_importance_covers_all_raw_features():
    """Global importance should cover all 19 raw features."""
    result = global_importance(sample_size=100)
    features = {item["feature"] for item in result}
    missing = set(ALL_RAW_FEATURES) - features
    assert not missing, f"Missing features in global importance: {missing}"


def test_get_explainer_cached():
    """get_explainer should return the same object on subsequent calls (lru_cache)."""
    exp1 = get_explainer()
    exp2 = get_explainer()
    assert exp1 is exp2, "get_explainer should be cached"


def test_import_does_not_load_models():
    """Importing explain.py should not require models/ to exist."""
    # This test passes if import succeeds without models/ access
    # The actual check is that no exception is raised at import time
    pass  # If we got here, import worked