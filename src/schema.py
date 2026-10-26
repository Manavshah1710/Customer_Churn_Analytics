"""
Shared schema definitions for Customer Churn Analytics.
Contains allowed values for all categorical fields to keep Streamlit and API in sync.
"""

from typing import Literal

# ============================================================
# Categorical field allowed values (shared between UI and API)
# ============================================================

Gender = Literal["Female", "Male"]
SeniorCitizen = Literal[0, 1]
Partner = Literal["No", "Yes"]
Dependents = Literal["No", "Yes"]

ContractType = Literal["Month-to-month", "One year", "Two year"]
PaperlessBilling = Literal["Yes", "No"]
PaymentMethod = Literal[
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]

PhoneService = Literal["Yes", "No"]
MultipleLines = Literal["No", "Yes", "No phone service"]
InternetService = Literal["Fiber optic", "DSL", "No"]
OnlineSecurity = Literal["No", "Yes", "No internet service"]
OnlineBackup = Literal["No", "Yes", "No internet service"]
DeviceProtection = Literal["No", "Yes", "No internet service"]
TechSupport = Literal["No", "Yes", "No internet service"]
StreamingTV = Literal["No", "Yes", "No internet service"]
StreamingMovies = Literal["No", "Yes", "No internet service"]

# ============================================================
# Numeric field constraints
# ============================================================

TENURE_MIN = 0
TENURE_MAX = 72
MONTHLY_CHARGES_MIN = 0.0
MONTHLY_CHARGES_MAX = 250.0
TOTAL_CHARGES_MIN = 0.0
TOTAL_CHARGES_MAX = 15000.0

# ============================================================
# Top-k for explanations
# ============================================================

TOP_K_MIN = 1
TOP_K_MAX = 10
TOP_K_DEFAULT = 5

# ============================================================
# All raw feature names (for validation)
# ============================================================

ALL_RAW_FEATURES = [
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "tenure",
    "contract",
    "paperless_billing",
    "payment_method",
    "monthly_charges",
    "total_charges",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
]