"""
Model input columns, shared by training, the Predict form, the API and the upload template.

These 15 columns were chosen by permutation importance with cross-validation on the training split:
they score AUC 0.6833 against 0.6842 for all 77 non-demographic columns. Demographic and household
columns are excluded for fairness; crclscod and area because their codes only make sense for this carrier.

The dataset's published dictionary defines only months, change_mou, mou_Mean, hnd_price, drop_vce_Mean
and change_rev, and its "percentage" wording for the change columns does not fit the data (22.6% of
change_mou is below -100). Every other meaning below is inferred from the column name and values.
"""

from typing import Literal

AslFlag = Literal["N", "Y"]

# min and max are sanity limits that contain every row of the dataset, not typical ranges.
NUMERIC_FIELDS = {
    "months": {
        "label": "Months with Company", "min": 0, "max": 120, "default": 16, "step": 1,
        "help": "Total months the customer has been with the company.",
    },
    "eqpdays": {
        "label": "Days with Current Handset", "min": -10.0, "max": 3650.0, "default": 342.0, "step": 10.0,
        "help": "Days since the current handset was issued (inferred).",
    },
    "change_mou": {
        "label": "Change in Monthly Minutes", "min": -5000.0, "max": 50000.0, "default": -6.0, "step": 10.0,
        "help": "Recent change in monthly minutes against the customer's baseline. Negative means less use (inferred).",
    },
    "mou_Mean": {
        "label": "Average Monthly Minutes", "min": 0.0, "max": 20000.0, "default": 355.0, "step": 10.0,
        "help": "Average monthly minutes of use.",
    },
    "totmrc_Mean": {
        "label": "Monthly Recurring Charge ($)", "min": -50.0, "max": 1000.0, "default": 45.0, "step": 5.0,
        "help": "Average monthly recurring charge (inferred).",
    },
    "avgqty": {
        "label": "Average Monthly Calls", "min": 0.0, "max": 5000.0, "default": 127.0, "step": 10.0,
        "help": "Average monthly calls over the customer's lifetime (inferred).",
    },
    "hnd_price": {
        "label": "Handset Price ($)", "min": 0.0, "max": 2000.0, "default": 100.0, "step": 10.0,
        "help": "Estimated retail price of the current handset.",
    },
    "uniqsubs": {
        "label": "Subscribers on Account", "min": 0, "max": 500, "default": 1, "step": 1,
        "help": "Unique subscribers in the household (inferred).",
    },
    "drop_vce_Mean": {
        "label": "Dropped Calls per Month", "min": 0.0, "max": 500.0, "default": 3.0, "step": 1.0,
        "help": "Average monthly voice calls dropped.",
    },
    "change_rev": {
        "label": "Change in Monthly Revenue ($)", "min": -2000.0, "max": 20000.0, "default": 0.0, "step": 5.0,
        "help": "Recent change in monthly revenue against the customer's baseline (inferred).",
    },
    "mou_cvce_Mean": {
        "label": "Completed Voice Minutes", "min": 0.0, "max": 10000.0, "default": 146.0, "step": 10.0,
        "help": "Average monthly minutes on completed voice calls (inferred).",
    },
    "avgrev": {
        "label": "Lifetime Average Revenue ($)", "min": 0.0, "max": 2000.0, "default": 50.0, "step": 5.0,
        "help": "Average monthly revenue over the customer's lifetime (inferred).",
    },
    "ovrmou_Mean": {
        "label": "Overage Minutes per Month", "min": 0.0, "max": 10000.0, "default": 3.0, "step": 5.0,
        "help": "Average monthly minutes used beyond the plan (inferred).",
    },
    "actvsubs": {
        "label": "Active Subscribers", "min": 0, "max": 100, "default": 1, "step": 1,
        "help": "Active subscribers in the household (inferred).",
    },
}

NUMERICAL_FEATURES = list(NUMERIC_FIELDS)
CATEGORICAL_FEATURES = ["asl_flag"]
MODEL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

ASL_FLAG_LABEL = "Account Spending Limit"
ASL_FLAG_HELP = "Whether the account has a spending limit (inferred)."
FEATURE_LABELS = {**{name: spec["label"] for name, spec in NUMERIC_FIELDS.items()}, "asl_flag": ASL_FLAG_LABEL}

FIELD_GROUPS = {
    "🕒 Tenure & Handset": ["months", "eqpdays", "hnd_price"],
    "📞 Usage": ["mou_Mean", "mou_cvce_Mean", "avgqty", "change_mou", "ovrmou_Mean", "drop_vce_Mean"],
    "💳 Billing": ["totmrc_Mean", "avgrev", "change_rev"],
    "👥 Account": ["uniqsubs", "actvsubs"],
}

EXAMPLE_HIGH_RISK = {
    "months": 11, "eqpdays": 700.0, "change_mou": -400.0, "mou_Mean": 80.0, "totmrc_Mean": 30.0,
    "avgqty": 60.0, "hnd_price": 9.99, "uniqsubs": 1, "drop_vce_Mean": 1.0, "change_rev": -20.0,
    "mou_cvce_Mean": 30.0, "avgrev": 30.0, "ovrmou_Mean": 0.0, "actvsubs": 1, "asl_flag": "N",
}

EXAMPLE_LOW_RISK = {
    "months": 8, "eqpdays": 200.0, "change_mou": 100.0, "mou_Mean": 800.0, "totmrc_Mean": 60.0,
    "avgqty": 400.0, "hnd_price": 199.99, "uniqsubs": 2, "drop_vce_Mean": 5.0, "change_rev": 5.0,
    "mou_cvce_Mean": 400.0, "avgrev": 70.0, "ovrmou_Mean": 5.0, "actvsubs": 2, "asl_flag": "Y",
}
