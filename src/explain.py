"""
SHAP explainability module for Customer Churn Analytics.

Provides grouped per-customer explanations and global feature importance.
Explainer is built lazily on first use (never at import time).
"""

import sys
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import load_artifacts
from src.preprocess import transform_new


# Raw feature to group mapping
RAW_FEATURE_GROUPS = {
    "demographics": [
        "gender", "senior_citizen", "partner", "dependents"
    ],
    "account": [
        "tenure", "contract", "paperless_billing", "payment_method",
        "monthly_charges", "total_charges"
    ],
    "services": [
        "phone_service", "multiple_lines", "internet_service",
        "online_security", "online_backup", "device_protection",
        "tech_support", "streaming_tv", "streaming_movies"
    ]
}

# Flatten for validation
ALL_RAW_FEATURES = []
for group_features in RAW_FEATURE_GROUPS.values():
    ALL_RAW_FEATURES.extend(group_features)


def _get_raw_feature_value(raw_input: Dict[str, Any], feature: str) -> Any:
    """Extract raw value for a feature from input dict."""
    return raw_input.get(feature, None)


def _sum_shap_by_raw_feature(
    shap_values: np.ndarray,
    feature_names: List[str],
    raw_input: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Sum one-hot dummy SHAP values into their raw feature.
    Returns list of dicts with raw feature name, customer value, summed SHAP, direction.
    """
    # Map transformed feature -> raw feature
    raw_to_shap = {feat: 0.0 for feat in ALL_RAW_FEATURES}
    raw_to_dummy_count = {feat: 0 for feat in ALL_RAW_FEATURES}

    for i, fname in enumerate(feature_names):
        if fname.startswith("num__"):
            raw_name = fname[5:]
            if raw_name in raw_to_shap:
                raw_to_shap[raw_name] += shap_values[i]
                raw_to_dummy_count[raw_name] += 1
        elif fname.startswith("bin__"):
            raw_name = fname[5:]
            if raw_name in raw_to_shap:
                raw_to_shap[raw_name] += shap_values[i]
                raw_to_dummy_count[raw_name] += 1
        elif fname.startswith("cat__"):
            suffix = fname[5:]
            best_match = None
            best_len = 0
            for raw in ALL_RAW_FEATURES:
                if suffix.startswith(raw + "_") and len(raw) > best_len:
                    best_len = len(raw)
                    best_match = raw
            if best_match:
                raw_to_shap[best_match] += shap_values[i]
                raw_to_dummy_count[best_match] += 1

    # Build reasons list
    reasons = []
    for raw_feat in ALL_RAW_FEATURES:
        shap_sum = raw_to_shap[raw_feat]
        if raw_to_dummy_count[raw_feat] > 0 or raw_feat in ["senior_citizen"]:  # include binary/numeric even if 0
            reasons.append({
                "feature": raw_feat,
                "customer_value": _get_raw_feature_value(raw_input, raw_feat),
                "shap_value": float(shap_sum),
                "direction": "increases_risk" if shap_sum > 0 else "decreases_risk"
            })

    # Sort by absolute SHAP value descending
    reasons.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    return reasons


@lru_cache(maxsize=1)
def get_explainer():
    """
    Lazily build and cache the SHAP TreeExplainer.
    Never runs at import time.
    """
    import shap

    model, preprocessor = load_artifacts(str(PROJECT_ROOT / "models"))
    return shap.TreeExplainer(model)


def explain_customer(raw_input: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
    """
    Generate a grouped SHAP explanation for a single customer.

    Args:
        raw_input: Dict with 19 raw feature values
        top_k: Number of top reasons to return (default 5)

    Returns:
        Dict with keys: probability, base_value, risk_level, reasons[]
    """
    model, preprocessor = load_artifacts(str(PROJECT_ROOT / "models"))
    artifacts = preprocessor

    # Get thresholds from metrics
    import json
    metrics_path = PROJECT_ROOT / "models" / "metrics.json"
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    threshold_high = metrics["threshold_high_risk"]
    threshold_moderate = metrics["threshold_moderate_risk"]

    # Transform input
    input_df = pd.DataFrame([raw_input])
    input_proc = transform_new(input_df, artifacts)

    # Get model probability
    prob = float(model.predict_proba(input_proc)[0, 1])

    # Get SHAP values (lazy explainer)
    explainer = get_explainer()
    shap_values = explainer.shap_values(input_proc)  # shape (1, n_features, 2)

    # For binary classification: [0] = class 0, [1] = class 1
    if shap_values.ndim == 3:
        shap_class1 = shap_values[0, :, 1]
        base_value = float(explainer.expected_value[1])
    else:
        shap_class1 = shap_values[0]
        base_value = float(explainer.expected_value)

    feature_names = artifacts["preprocessor"].get_feature_names_out().tolist()

    # Sum dummies into raw features
    reasons = _sum_shap_by_raw_feature(shap_class1, feature_names, raw_input)

    # Determine risk level
    if prob >= threshold_high:
        risk_level = "high"
    elif prob >= threshold_moderate:
        risk_level = "moderate"
    else:
        risk_level = "low"

    return {
        "probability": prob,
        "base_value": base_value,
        "risk_level": risk_level,
        "reasons": reasons[:top_k]
    }


def global_importance(sample_size: int = 300) -> List[Dict[str, Any]]:
    """
    Compute global feature importance as mean |SHAP| over a sample.

    Returns list of dicts with raw feature name and mean_abs_shap, sorted descending.
    """
    model, preprocessor = load_artifacts(str(PROJECT_ROOT / "models"))
    artifacts = preprocessor

    from src.preprocess import load_data, split_data, build_features
    df = load_data("data/churn.db")
    X_train, X_test, y_train, y_test = split_data(df, target_col="churn", test_size=0.2, random_state=42)
    X_train_proc, X_test_proc, _ = build_features(X_train, X_test)

    # Sample
    n = min(sample_size, len(X_test_proc))
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_test_proc), n, replace=False)
    X_sample = X_test_proc.iloc[sample_idx]

    explainer = get_explainer()
    shap_values = explainer.shap_values(X_sample)

    if shap_values.ndim == 3:
        shap_class1 = shap_values[:, :, 1]
    else:
        shap_class1 = shap_values

    feature_names = artifacts["preprocessor"].get_feature_names_out().tolist()

    # Sum dummies into raw features for each sample, then average
    raw_shap_sums = {feat: [] for feat in ALL_RAW_FEATURES}

    for i in range(shap_class1.shape[0]):
        reasons = _sum_shap_by_raw_feature(shap_class1[i], feature_names, {})
        for r in reasons:
            raw_shap_sums[r["feature"]].append(abs(r["shap_value"]))

    # Mean |SHAP| per raw feature
    global_imp = []
    for feat, vals in raw_shap_sums.items():
        if vals:
            global_imp.append({
                "feature": feat,
                "mean_abs_shap": float(np.mean(vals))
            })

    global_imp.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
    return global_imp


# Self-check when run directly
if __name__ == "__main__":
    import time

    # Test import time
    import_start = time.perf_counter()
    import shap
    import_time = time.perf_counter() - import_start
    print(f"import shap: {import_time*1000:.1f}ms")

    # Test explainer build
    build_start = time.perf_counter()
    explainer = get_explainer()
    build_time = time.perf_counter() - build_start
    print(f"explainer build: {build_time*1000:.1f}ms")

    # Test customers
    customer_a = {
        "gender": "Female", "senior_citizen": 0, "partner": "No", "dependents": "No",
        "tenure": 1, "contract": "Month-to-month", "paperless_billing": "Yes",
        "payment_method": "Electronic check", "monthly_charges": 85.0, "total_charges": 85.0,
        "phone_service": "Yes", "multiple_lines": "No", "internet_service": "Fiber optic",
        "online_security": "No", "online_backup": "No", "device_protection": "No",
        "tech_support": "No", "streaming_tv": "No", "streaming_movies": "No",
    }

    customer_b = {
        "gender": "Male", "senior_citizen": 0, "partner": "Yes", "dependents": "Yes",
        "tenure": 60, "contract": "Two year", "paperless_billing": "No",
        "payment_method": "Credit card (automatic)", "monthly_charges": 60.0, "total_charges": 3600.0,
        "phone_service": "Yes", "multiple_lines": "Yes", "internet_service": "DSL",
        "online_security": "Yes", "online_backup": "Yes", "device_protection": "Yes",
        "tech_support": "Yes", "streaming_tv": "No", "streaming_movies": "No",
    }

    # Cold call
    cold_start = time.perf_counter()
    exp_a = explain_customer(customer_a)
    cold_time = time.perf_counter() - cold_start

    # Warm call
    warm_start = time.perf_counter()
    exp_b = explain_customer(customer_b)
    warm_time = time.perf_counter() - warm_start

    print(f"cold explain_customer: {cold_time*1000:.1f}ms")
    print(f"warm explain_customer: {warm_time*1000:.1f}ms")

    import json
    print("\nCustomer A (high risk):")
    print(json.dumps(exp_a, indent=2))
    print("\nCustomer B (low risk):")
    print(json.dumps(exp_b, indent=2))

    # Global
    glob = global_importance(300)
    print("\nTop 8 global features (mean |SHAP|):")
    for g in glob[:8]:
        print(f"  {g['feature']}: {g['mean_abs_shap']:.4f}")