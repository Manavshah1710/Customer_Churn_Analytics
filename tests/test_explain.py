"""
Tests for SHAP explainability module.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src import PROJECT_ROOT
from src.explain import explain_customer, global_importance, get_explainer, _sum_shap_by_raw_feature
from src.predict import load_artifacts
from src.preprocess import load_data, split_data, transform_new
from src.schema import EXAMPLE_HIGH_RISK, EXAMPLE_LOW_RISK, MODEL_FEATURES

ALL_REASONS = len(MODEL_FEATURES)


def _sigmoid(value):
    return 1 / (1 + np.exp(-value))


def test_explain_customer_schema():
    """Verify explain_customer returns the correct schema with proper types."""
    result = explain_customer(EXAMPLE_HIGH_RISK)

    assert isinstance(result["probability"], float)
    assert isinstance(result["base_value"], float)
    assert result["risk_level"] in ("high", "moderate", "low")
    assert isinstance(result["reasons"], list)
    assert len(result["reasons"]) == 5  # default top_k=5

    for r in result["reasons"]:
        assert r["feature"] in MODEL_FEATURES
        assert r["customer_value"] == EXAMPLE_HIGH_RISK[r["feature"]]
        assert isinstance(r["shap_value"], float)
        assert r["direction"] == ("increases_risk" if r["shap_value"] > 0 else "decreases_risk")


@pytest.mark.parametrize("customer", [EXAMPLE_HIGH_RISK, EXAMPLE_LOW_RISK])
def test_explain_customer_additivity(customer):
    """Verify sigmoid(base_value + sum(shap_values)) == probability: SHAP values are in log-odds."""
    result = explain_customer(customer, top_k=ALL_REASONS)

    assert len(result["reasons"]) == ALL_REASONS
    total = result["base_value"] + sum(r["shap_value"] for r in result["reasons"])
    assert abs(_sigmoid(total) - result["probability"]) < 1e-4, (
        f"Additivity failed: sigmoid(base+sum)={_sigmoid(total)}, prob={result['probability']}"
    )


def test_explain_customer_risk_level_matches_thresholds():
    """Verify risk_level matches thresholds from metrics.json across real held-out customers."""
    with open(PROJECT_ROOT / "models" / "metrics.json", "r") as f:
        metrics = json.load(f)
    threshold_high = metrics["threshold_high_risk"]
    threshold_moderate = metrics["threshold_moderate_risk"]

    assert explain_customer(EXAMPLE_HIGH_RISK)["risk_level"] == "high"
    assert explain_customer(EXAMPLE_LOW_RISK)["risk_level"] == "low"

    _, X_test, _, _ = split_data(load_data("data/churn.db"))
    seen = set()
    for customer in X_test.head(100).to_dict("records"):
        result = explain_customer(customer, top_k=1)
        expected = (
            "high" if result["probability"] >= threshold_high
            else "moderate" if result["probability"] >= threshold_moderate
            else "low"
        )
        assert result["risk_level"] == expected
        seen.add(expected)
    assert seen == {"high", "moderate", "low"}, f"Sample did not exercise every tier: {seen}"


def test_explain_customer_high_risk_sanity():
    """High-risk customer has falling usage or tenure in its top reasons, and they increase risk."""
    result = explain_customer(EXAMPLE_HIGH_RISK, top_k=ALL_REASONS)
    top3 = {r["feature"]: r["direction"] for r in result["reasons"][:3]}

    assert {"change_mou", "mou_Mean", "months"} & set(top3), f"Unexpected top reasons: {list(top3)}"
    for feature in {"change_mou", "mou_Mean", "months"} & set(top3):
        assert top3[feature] == "increases_risk", f"{feature} should increase risk"


def test_explain_customer_low_risk_sanity():
    """Low-risk customer's contributions must pull the score below the base value."""
    result = explain_customer(EXAMPLE_LOW_RISK, top_k=ALL_REASONS)
    assert sum(r["shap_value"] for r in result["reasons"]) < 0


def test_explain_customer_reasons_sorted_by_abs_shap():
    """Reasons must be sorted by absolute SHAP value descending."""
    result = explain_customer(EXAMPLE_HIGH_RISK, top_k=ALL_REASONS)
    abs_shaps = [abs(r["shap_value"]) for r in result["reasons"]]
    assert abs_shaps == sorted(abs_shaps, reverse=True)


def test_explain_customer_handles_missing_values():
    """Real records have gaps: a null value still gets a score and is reported as null."""
    result = explain_customer({**EXAMPLE_HIGH_RISK, "change_mou": None, "asl_flag": None}, top_k=ALL_REASONS)

    assert 0.0 <= result["probability"] <= 1.0
    by_feature = {r["feature"]: r for r in result["reasons"]}
    assert by_feature["change_mou"]["customer_value"] is None
    assert by_feature["asl_flag"]["customer_value"] is None


def test_global_importance_schema():
    """Verify global_importance returns correct structure."""
    result = global_importance(sample_size=100)

    assert isinstance(result, list)
    assert len(result) > 0

    for item in result:
        assert isinstance(item["mean_abs_shap"], float)
        assert item["mean_abs_shap"] >= 0
        assert item["feature"] in MODEL_FEATURES

    vals = [item["mean_abs_shap"] for item in result]
    assert vals == sorted(vals, reverse=True)


def test_global_importance_covers_all_raw_features():
    """Global importance should cover every model input column."""
    features = {item["feature"] for item in global_importance(sample_size=100)}
    missing = set(MODEL_FEATURES) - features
    assert not missing, f"Missing features in global importance: {missing}"


def test_get_explainer_cached():
    """get_explainer should return the same object on subsequent calls (lru_cache)."""
    assert get_explainer() is get_explainer(), "get_explainer should be cached"


def test_grouped_shap_sum_matches_ungrouped():
    """Verify grouped SHAP sum matches ungrouped (per-dummy) sum within 1e-6 for several customers."""
    _, preprocessor = load_artifacts(str(PROJECT_ROOT / "models"))
    explainer = get_explainer()
    feature_names = preprocessor["preprocessor"].get_feature_names_out().tolist()

    for customer in [EXAMPLE_HIGH_RISK, EXAMPLE_LOW_RISK, {**EXAMPLE_HIGH_RISK, "asl_flag": "Y"}]:
        input_proc = transform_new(pd.DataFrame([customer]), preprocessor)
        shap_class1 = explainer.shap_values(input_proc)[0]

        reasons = _sum_shap_by_raw_feature(shap_class1, feature_names, customer)
        grouped_sum = sum(r["shap_value"] for r in reasons)

        assert abs(grouped_sum - float(np.sum(shap_class1))) < 1e-6, f"Grouped/ungrouped mismatch for {customer}"
