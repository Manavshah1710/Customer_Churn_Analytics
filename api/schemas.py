"""
Pydantic schemas for the FastAPI service.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, ConfigDict

from src.schema import AslFlag, NUMERIC_FIELDS


def _limited(name: str):
    """A required field within the shared limits for this column."""
    return Field(ge=NUMERIC_FIELDS[name]["min"], le=NUMERIC_FIELDS[name]["max"])


# ============================================================
# Request / Response Models
# ============================================================

class CustomerInput(BaseModel):
    """
    Customer data for prediction. Every field must be sent, but real records have gaps,
    so null is allowed and is filled the same way as gaps in the training data.
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    months: Optional[int] = _limited("months")
    eqpdays: Optional[float] = _limited("eqpdays")
    change_mou: Optional[float] = _limited("change_mou")
    mou_Mean: Optional[float] = _limited("mou_Mean")
    totmrc_Mean: Optional[float] = _limited("totmrc_Mean")
    avgqty: Optional[float] = _limited("avgqty")
    hnd_price: Optional[float] = _limited("hnd_price")
    uniqsubs: Optional[int] = _limited("uniqsubs")
    drop_vce_Mean: Optional[float] = _limited("drop_vce_Mean")
    change_rev: Optional[float] = _limited("change_rev")
    mou_cvce_Mean: Optional[float] = _limited("mou_cvce_Mean")
    avgrev: Optional[float] = _limited("avgrev")
    ovrmou_Mean: Optional[float] = _limited("ovrmou_Mean")
    actvsubs: Optional[int] = _limited("actvsubs")
    asl_flag: Optional[AslFlag]


class Reason(BaseModel):
    """Single feature contribution to the prediction."""
    feature: str
    customer_value: str | int | float | None
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
    learning_rate: Optional[float]


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
    learning_rate: float
