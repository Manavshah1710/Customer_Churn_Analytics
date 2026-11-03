"""
Shared schema definitions for Customer Churn Analytics.
Contains allowed values for all categorical fields to keep Streamlit and API in sync.
"""

from typing import Literal

# ============================================================
# Model input columns (shared by training, form, API and upload template)
# ============================================================

# Chosen by permutation importance with cross-validation on the training split: these 15 score
# AUC 0.6833 against 0.6842 for all 77 non-demographic columns. Demographic and household columns
# are excluded for fairness; crclscod and area because their codes only make sense for this carrier.
NUMERICAL_FEATURES = [
    "months",
    "eqpdays",
    "change_mou",
    "mou_Mean",
    "totmrc_Mean",
    "avgqty",
    "hnd_price",
    "uniqsubs",
    "drop_vce_Mean",
    "change_rev",
    "mou_cvce_Mean",
    "avgrev",
    "ovrmou_Mean",
    "actvsubs",
]
CATEGORICAL_FEATURES = ["asl_flag"]
MODEL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

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