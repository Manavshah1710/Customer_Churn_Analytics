"""
Tests for FastAPI service.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.explain import explain_customer


# Test client - must be used as context manager for lifespan to run
@pytest.fixture(scope="module")
def api_client():
    with TestClient(app) as client:
        yield client


# Sanity check customers
HIGH_RISK_CUSTOMER = {
    "gender": "Female", "senior_citizen": 0, "partner": "No", "dependents": "No",
    "tenure": 1, "contract": "Month-to-month", "paperless_billing": "Yes",
    "payment_method": "Electronic check", "monthly_charges": 85.0, "total_charges": 85.0,
    "phone_service": "Yes", "multiple_lines": "No", "internet_service": "Fiber optic",
    "online_security": "No", "online_backup": "No", "device_protection": "No",
    "tech_support": "No", "streaming_tv": "No", "streaming_movies": "No",
}

LOW_RISK_CUSTOMER = {
    "gender": "Male", "senior_citizen": 0, "partner": "Yes", "dependents": "Yes",
    "tenure": 60, "contract": "Two year", "paperless_billing": "No",
    "payment_method": "Credit card (automatic)", "monthly_charges": 60.0, "total_charges": 3600.0,
    "phone_service": "Yes", "multiple_lines": "Yes", "internet_service": "DSL",
    "online_security": "Yes", "online_backup": "Yes", "device_protection": "Yes",
    "tech_support": "Yes", "streaming_tv": "No", "streaming_movies": "No",
}


class TestHealth:
    def test_health_returns_200(self, api_client):
        """Health endpoint returns 200 OK."""
        response = api_client.get("/health")
        assert response.status_code == 200

    def test_health_structure(self, api_client):
        """Health response has expected structure."""
        response = api_client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] in ("ok", "degraded")
        assert "artifacts_loaded" in data
        assert "explainer_loaded" in data
        assert "model_type" in data
        assert "max_depth" in data
        assert "n_features" in data
        assert "n_estimators" in data
        assert "class_weight" in data

    def test_health_artifacts_loaded(self, api_client):
        """Artifacts and explainer should be loaded in test environment."""
        response = api_client.get("/health")
        data = response.json()
        assert data["artifacts_loaded"] is True
        assert data["explainer_loaded"] is True
        assert data["model_type"] == "rf_balanced"
        assert data["max_depth"] == 8


class TestPredict:
    def test_predict_valid_customer_returns_200(self, api_client):
        """Valid customer returns 200 with full schema."""
        response = api_client.post("/predict", json=HIGH_RISK_CUSTOMER)
        assert response.status_code == 200

    def test_predict_response_schema(self, api_client):
        """Response matches explain_customer schema exactly."""
        response = api_client.post("/predict", json=HIGH_RISK_CUSTOMER)
        data = response.json()

        # Required keys
        assert "probability" in data
        assert "base_value" in data
        assert "risk_level" in data
        assert "reasons" in data

        # Types
        assert isinstance(data["probability"], float)
        assert isinstance(data["base_value"], float)
        assert data["risk_level"] in ("high", "moderate", "low")
        assert isinstance(data["reasons"], list)

        # Each reason has correct structure
        for reason in data["reasons"]:
            assert "feature" in reason
            assert "customer_value" in reason
            assert "shap_value" in reason
            assert "direction" in reason
            assert reason["direction"] in ("increases_risk", "decreases_risk")
            assert isinstance(reason["shap_value"], float)

    def test_predict_probability_in_range(self, api_client):
        """Probability is within [0, 1]."""
        response = api_client.post("/predict", json=HIGH_RISK_CUSTOMER)
        prob = response.json()["probability"]
        assert 0.0 <= prob <= 1.0

    def test_predict_risk_level_matches_thresholds(self, api_client):
        """Risk level matches thresholds from metrics.json."""
        # Load thresholds
        metrics_path = Path("models") / "metrics.json"
        with open(metrics_path) as f:
            metrics = json.load(f)
        threshold_high = metrics["threshold_high_risk"]
        threshold_moderate = metrics["threshold_moderate_risk"]

        # High risk customer
        response = api_client.post("/predict", json=HIGH_RISK_CUSTOMER)
        data = response.json()
        assert data["risk_level"] == "high"
        assert data["probability"] >= threshold_high

        # Low risk customer
        response = api_client.post("/predict", json=LOW_RISK_CUSTOMER)
        data = response.json()
        assert data["risk_level"] == "low"
        assert data["probability"] < threshold_moderate

    def test_predict_reasons_count_equals_top_k(self, api_client):
        """Number of reasons equals top_k parameter."""
        for k in [1, 3, 5, 8, 10]:
            response = api_client.post("/predict", json=HIGH_RISK_CUSTOMER, params={"top_k": k})
            assert response.status_code == 200
            data = response.json()
            assert len(data["reasons"]) == k

    def test_predict_reasons_sorted_by_abs_shap(self, api_client):
        """Reasons are sorted by absolute SHAP value descending."""
        response = api_client.post("/predict", json=HIGH_RISK_CUSTOMER, params={"top_k": 10})
        data = response.json()
        abs_shaps = [abs(r["shap_value"]) for r in data["reasons"]]
        for i in range(len(abs_shaps) - 1):
            assert abs_shaps[i] >= abs_shaps[i + 1], \
                f"Reasons not sorted by |shap|: {abs_shaps[i]} < {abs_shaps[i+1]}"

    def test_predict_matches_explain_customer_directly(self, api_client):
        """API output equals calling explain_customer directly."""
        # Get API result
        response = api_client.post("/predict", json=HIGH_RISK_CUSTOMER, params={"top_k": 8})
        api_result = response.json()

        # Get direct result
        direct_result = explain_customer(HIGH_RISK_CUSTOMER, top_k=8)

        # Compare key fields
        assert abs(api_result["probability"] - direct_result["probability"]) < 1e-10
        assert abs(api_result["base_value"] - direct_result["base_value"]) < 1e-10
        assert api_result["risk_level"] == direct_result["risk_level"]
        assert len(api_result["reasons"]) == len(direct_result["reasons"])

        for api_r, direct_r in zip(api_result["reasons"], direct_result["reasons"]):
            assert api_r["feature"] == direct_r["feature"]
            assert api_r["customer_value"] == direct_r["customer_value"]
            assert abs(api_r["shap_value"] - direct_r["shap_value"]) < 1e-10
            assert api_r["direction"] == direct_r["direction"]

    def test_high_risk_scores_above_low_risk(self, api_client):
        """High-risk sanity customer scores far above low-risk one."""
        high_resp = api_client.post("/predict", json=HIGH_RISK_CUSTOMER)
        low_resp = api_client.post("/predict", json=LOW_RISK_CUSTOMER)

        high_prob = high_resp.json()["probability"]
        low_prob = low_resp.json()["probability"]

        assert high_prob > 0.8  # High risk
        assert low_prob < 0.1   # Low risk
        assert high_prob - low_prob > 0.5  # Significant gap


class TestPredictValidation:
    def test_missing_field_returns_422(self, api_client):
        """Missing required field returns 422."""
        bad_customer = HIGH_RISK_CUSTOMER.copy()
        del bad_customer["streaming_movies"]
        response = api_client.post("/predict", json=bad_customer)
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        assert any("streaming_movies" in str(e) for e in data["detail"])

    def test_bad_enum_value_returns_422(self, api_client):
        """Invalid enum value returns 422."""
        bad_customer = HIGH_RISK_CUSTOMER.copy()
        bad_customer["gender"] = "Invalid"
        response = api_client.post("/predict", json=bad_customer)
        assert response.status_code == 422

    def test_negative_tenure_returns_422(self, api_client):
        """Negative tenure returns 422."""
        bad_customer = HIGH_RISK_CUSTOMER.copy()
        bad_customer["tenure"] = -5
        response = api_client.post("/predict", json=bad_customer)
        assert response.status_code == 422

    def test_unknown_extra_field_returns_422(self, api_client):
        """Unknown extra field returns 422 (extra='forbid')."""
        bad_customer = HIGH_RISK_CUSTOMER.copy()
        bad_customer["extra_field"] = "not_allowed"
        response = api_client.post("/predict", json=bad_customer)
        assert response.status_code == 422


class TestModelInfo:
    def test_model_info_returns_200(self, api_client):
        """Model info endpoint returns 200."""
        response = api_client.get("/model-info")
        assert response.status_code == 200

    def test_model_info_structure(self, api_client):
        """Model info has all expected fields."""
        response = api_client.get("/model-info")
        data = response.json()

        expected_fields = [
            "auc", "precision", "recall", "f1", "accuracy",
            "top_10_decile_capture_rate", "churners_caught_in_top_10",
            "total_churners_in_test", "threshold_high_risk",
            "threshold_moderate_risk", "confusion_matrix",
            "model_type", "max_depth", "n_estimators", "class_weight"
        ]
        for field in expected_fields:
            assert field in data, f"Missing field: {field}"

    def test_model_info_matches_metrics_json(self, api_client):
        """Model info matches metrics.json exactly."""
        with open("models/metrics.json") as f:
            metrics = json.load(f)

        response = api_client.get("/model-info")
        data = response.json()

        for key in metrics:
            if key in data:
                assert data[key] == metrics[key], f"Mismatch in {key}: API={data[key]}, JSON={metrics[key]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])