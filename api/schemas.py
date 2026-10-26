"""
Pydantic schemas for the FastAPI service.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator


# ============================================================
# Request / Response Models
# ============================================================

class CustomerInput(BaseModel):
    """Raw customer data for prediction. All 19 fields from the feature set."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    # Demographics
    gender: Literal["Female", "Male"]
    senior_citizen: Literal[0, 1]
    partner: Literal["No", "Yes"]
    dependents: Literal["No", "Yes"]

    # Account & Billing
    tenure: int = Field(ge=0, le=72)
    contract: Literal["Month-to-month", "One year", "Two year"]
    paperless_billing: Literal["Yes", "No"]
    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    monthly_charges: float = Field(ge=0.0, le=250.0)
    total_charges: Optional[float] = Field(default=None, ge=0.0, le=15000.0)

    # Services
    phone_service: Literal["Yes", "No"]
    multiple_lines: Literal["No", "Yes", "No phone service"]
    internet_service: Literal["Fiber optic", "DSL", "No"]
    online_security: Literal["No", "Yes", "No internet service"]
    online_backup: Literal["No", "Yes", "No internet service"]
    device_protection: Literal["No", "Yes", "No internet service"]
    tech_support: Literal["No", "Yes", "No internet service"]
    streaming_tv: Literal["No", "Yes", "No internet service"]
    streaming_movies: Literal["No", "Yes", "No internet service"]

    @field_validator("total_charges", mode="before")
    @classmethod
    def default_total_charges(cls, v, info):
        """Default total_charges to tenure * monthly_charges if not provided."""
        if v is None:
            # We can't access other fields here in mode="before", so return None
            # The actual default will be computed in the endpoint
            return None
        return v


class Reason(BaseModel):
    """Single feature contribution to the prediction."""
    feature: str
    customer_value: str | int | float
    shap_value: float
    direction: Literal["increases_risk", "decreases_risk"]


class PredictionResponse(BaseModel):
    """Response from /predict endpoint - matches explain_customer schema exactly."""
    probability: float
    base_value: float
    risk_level: Literal["high", "moderate", "low"]
    reasons: list[Reason]


class HealthResponse(BaseModel):
    """Response from /health endpoint."""
    status: Literal["ok", "degraded"]
    artifacts_loaded: bool
    explainer_loaded: bool
    model_type: str
    max_depth: int
    n_features: int
    n_estimators: int
    class_weight: str


class ModelInfoResponse(BaseModel):
    """Response from /model-info endpoint."""
    auc: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    top_10_decile_capture_rate: float
    churners_caught_in_top_10: int
    total_churners_in_test: int
    threshold_high_risk: float
    threshold_moderate_risk: float
    confusion_matrix: list[list[int]]
    model_type: str
    max_depth: int
    n_estimators: int
    class_weight: str