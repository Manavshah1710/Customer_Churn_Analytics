"""
FastAPI service for Customer Churn Prediction.
Provides /health, /predict, and /model-info endpoints.
"""

import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Response

from src import PROJECT_ROOT
from src.predict import load_artifacts
from src.explain import explain_customer, get_explainer
from src.prediction_log import log_prediction
from src.schema import EXAMPLE_HIGH_RISK
from api.schemas import (
    CustomerInput,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    Reason,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state for startup-loaded artifacts
_artifacts = {
    "model": None,
    "preprocessor": None,
    "metrics": None,
    "explainer": None,
    "startup_time_ms": 0,
    "warmup_done": False,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model, preprocessor, metrics, and build SHAP explainer at startup."""
    start = time.perf_counter()
    models_dir = PROJECT_ROOT / "models"

    try:
        # Load model and preprocessor
        _artifacts["model"], _artifacts["preprocessor"] = load_artifacts(str(models_dir))
        logger.info("Model and preprocessor loaded")

        # Load metrics
        metrics_path = models_dir / "metrics.json"
        if not metrics_path.exists():
            raise RuntimeError(f"metrics.json not found at {metrics_path}")
        with open(metrics_path, "r") as f:
            _artifacts["metrics"] = json.load(f)
        logger.info("Metrics loaded")

        # Build SHAP explainer (lazy in explain module, but force it here)
        _artifacts["explainer"] = get_explainer()
        logger.info("SHAP explainer built")

        # Warm-up: run one explanation to avoid cold-start latency on first request
        _ = explain_customer(EXAMPLE_HIGH_RISK, top_k=5)
        _artifacts["warmup_done"] = True
        logger.info("Warm-up explanation completed")

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        # Don't raise here - let health endpoint report degraded state

    _artifacts["startup_time_ms"] = (time.perf_counter() - start) * 1000
    logger.info(f"Startup completed in {_artifacts['startup_time_ms']:.1f}ms")

    yield

    # Cleanup (if needed)
    logger.info("Shutting down")


app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


def _check_artifacts():
    """Verify all artifacts are loaded, raise HTTPException if not."""
    if _artifacts["model"] is None or _artifacts["preprocessor"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not loaded. Check server logs for startup errors."
        )
    if _artifacts["metrics"] is None:
        raise HTTPException(
            status_code=503,
            detail="Metrics not loaded."
        )


@app.get("/health", response_model=HealthResponse)
def health(response: Response):
    """Health check endpoint with model metadata. Returns 503 when degraded."""
    m = _artifacts["metrics"] or {}
    model = _artifacts["model"]
    artifacts_ok = model is not None and _artifacts["preprocessor"] is not None
    explainer_ok = _artifacts["explainer"] is not None and _artifacts["warmup_done"]
    healthy = artifacts_ok and explainer_ok

    if not healthy:
        response.status_code = 503

    return HealthResponse(
        status="ok" if healthy else "degraded",
        artifacts_loaded=artifacts_ok,
        explainer_loaded=explainer_ok,
        model_type=m.get("model_type"),
        max_depth=m.get("max_depth"),
        n_features=model.n_features_in_ if model is not None else None,
        n_estimators=m.get("n_estimators"),
        learning_rate=m.get("learning_rate"),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput, top_k: int = Query(5, ge=1, le=10)):
    """
    Predict a churn risk score and return SHAP explanation.
    Returns the exact same schema as explain_customer().
    """
    _check_artifacts()

    # Convert Pydantic model to dict
    raw_input = customer.model_dump()

    try:
        # Call the shared explain_customer function (which loads model/preprocessor internally)
        result = explain_customer(raw_input, top_k=top_k)

        # A monitoring failure must never break a prediction
        try:
            log_prediction(raw_input, result)
        except Exception as e:
            logger.error(f"Prediction logging failed: {type(e).__name__}")

        # Convert to response model
        return PredictionResponse(
            probability=result["probability"],
            base_value=result["base_value"],
            risk_level=result["risk_level"],
            reasons=[
                Reason(
                    feature=r["feature"],
                    customer_value=r["customer_value"],
                    shap_value=r["shap_value"],
                    direction=r["direction"],
                )
                for r in result["reasons"]
            ],
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    """Return held-out test metrics and thresholds from metrics.json."""
    _check_artifacts()

    m = _artifacts["metrics"]
    return ModelInfoResponse(
        auc=m["auc"],
        precision=m["precision"],
        recall=m["recall"],
        f1=m["f1"],
        accuracy=m["accuracy"],
        top_10_decile_capture_rate=m["top_10_decile_capture_rate"],
        churners_caught_in_top_10=m["churners_caught_in_top_10"],
        total_churners_in_test=m["total_churners_in_test"],
        threshold_high_risk=m["threshold_high_risk"],
        threshold_moderate_risk=m["threshold_moderate_risk"],
        confusion_matrix=m["confusion_matrix"],
        model_type=m["model_type"],
        max_depth=m["max_depth"],
        n_estimators=m["n_estimators"],
        learning_rate=m["learning_rate"],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)