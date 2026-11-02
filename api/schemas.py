"""
Pydantic schemas for the FastAPI service.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, ConfigDict

from src.schema import (
    Gender,
    SeniorCitizen,
    Partner,
    Dependents,
    ContractType,
    PaperlessBilling,
    PaymentMethod,
    PhoneService,
    MultipleLines,
    InternetService,
    OnlineSecurity,
    OnlineBackup,
    DeviceProtection,
    TechSupport,
    StreamingTV,
    StreamingMovies,
    TENURE_MIN,
    TENURE_MAX,
    MONTHLY_CHARGES_MIN,
    MONTHLY_CHARGES_MAX,
    TOTAL_CHARGES_MIN,
    TOTAL_CHARGES_MAX,
)


# ============================================================
# Request / Response Models
# ============================================================

class CustomerInput(BaseModel):
    """Raw customer data for prediction. All 19 fields from the feature set."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    # Demographics
    gender: Gender
    senior_citizen: SeniorCitizen
    partner: Partner
    dependents: Dependents

    # Account & Billing
    tenure: int = Field(ge=TENURE_MIN, le=TENURE_MAX)
    contract: ContractType
    paperless_billing: PaperlessBilling
    payment_method: PaymentMethod
    monthly_charges: float = Field(ge=MONTHLY_CHARGES_MIN, le=MONTHLY_CHARGES_MAX)
    # Blank means tenure * monthly_charges, computed in the /predict endpoint
    total_charges: Optional[float] = Field(default=None, ge=TOTAL_CHARGES_MIN, le=TOTAL_CHARGES_MAX)

    # Services
    phone_service: PhoneService
    multiple_lines: MultipleLines
    internet_service: InternetService
    online_security: OnlineSecurity
    online_backup: OnlineBackup
    device_protection: DeviceProtection
    tech_support: TechSupport
    streaming_tv: StreamingTV
    streaming_movies: StreamingMovies


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
    # None when the matching artifact failed to load
    model_type: Optional[str]
    max_depth: Optional[int]
    n_features: Optional[int]
    n_estimators: Optional[int]
    class_weight: Optional[str]


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