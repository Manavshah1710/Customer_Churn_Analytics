"""
Data loading, splitting, and preprocessing module for Customer Churn Analytics.
"""

import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_COLUMNS: List[str] = ["tenure", "monthly_charges", "total_charges"]


def load_data(db_path: str = "data/churn.db") -> pd.DataFrame:
    """
    Loads raw joined customer, account, and service records from the SQLite database.
    Cleans whitespace, encodes the target variable, drops customer_id,
    and coerces numeric columns to numeric types.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found at {db_path}")

    conn = sqlite3.connect(db_path)
    query = """
    SELECT
        c.customer_id, c.gender, c.senior_citizen, c.partner, c.dependents, c.tenure, c.churn,
        a.contract, a.paperless_billing, a.payment_method, a.monthly_charges, a.total_charges,
        s.phone_service, s.multiple_lines, s.internet_service, s.online_security,
        s.online_backup, s.device_protection, s.tech_support, s.streaming_tv,
        s.streaming_movies
    FROM customers c
    JOIN accounts a ON c.customer_id = a.customer_id
    JOIN services s ON c.customer_id = s.customer_id;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Clean whitespace on string columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # Encode target column
    if "churn" in df.columns:
        df["churn"] = df["churn"].map({"Yes": 1, "No": 0})

    # Drop non-feature identifier
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])

    # Numeric conversion
    for col in ["monthly_charges", "total_charges", "tenure"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def split_data(
    df: pd.DataFrame,
    target_col: str = "churn",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Performs a stratified train/test split on raw data before imputation or feature engineering.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor(X_train: pd.DataFrame, num_cols: Optional[List[str]] = None) -> Tuple[ColumnTransformer, List[str]]:
    """
    Builds a ColumnTransformer with median imputation + standard scaling for numeric,
    and one-hot encoding (no drop_first) for categorical features.
    Configured with set_output(transform="pandas") to return named DataFrames.
    """
    if num_cols is None:
        num_cols = list(NUMERIC_COLUMNS)

    # Identify categorical columns
    cat_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop=None, sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    ).set_output(transform="pandas")

    return preprocessor, cat_cols


def build_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    num_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Fits ColumnTransformer strictly on X_train, transforms both train and test splits,
    and returns transformed DataFrames alongside artifacts dictionary.
    """
    if num_cols is None:
        num_cols = list(NUMERIC_COLUMNS)

    preprocessor, cat_cols = build_preprocessor(X_train, num_cols)
    preprocessor.fit(X_train)

    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out().tolist()

    # Extract fitted imputer statistics for validation / metadata
    imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]
    train_medians = {col: float(med) for col, med in zip(num_cols, imputer.statistics_)}

    artifacts: Dict[str, Any] = {
        "preprocessor": preprocessor,
        "feature_columns": feature_names,
        "train_medians": train_medians,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
    }

    return X_train_transformed, X_test_transformed, artifacts


def transform_new(df: pd.DataFrame, artifacts: Dict[str, Any]) -> pd.DataFrame:
    """
    Transforms single-row or batch new data for inference using saved ColumnTransformer artifacts.
    Returns a pandas DataFrame with named columns.
    Raises ValueError if preprocessor is not the expected ColumnTransformer format.
    """
    if not isinstance(artifacts, dict) or "preprocessor" not in artifacts:
        raise ValueError(
            "Preprocessor artifact does not contain a fitted ColumnTransformer under 'preprocessor'. "
            "Please retrain with src.train to generate the expected artifact format."
        )

    preprocessor = artifacts["preprocessor"]
    if not hasattr(preprocessor, "transform"):
        raise ValueError("artifacts['preprocessor'] does not have a transform method.")

    df_copy = df.copy()

    # Remove non-feature columns if present
    for col in ["customer_id", "churn"]:
        if col in df_copy.columns:
            df_copy = df_copy.drop(columns=[col])

    # Strip string whitespace
    for col in df_copy.select_dtypes(include=["object", "string"]).columns:
        df_copy[col] = df_copy[col].astype(str).str.strip()

    # Handle numeric columns: coerce types and fill missing columns with NaN for SimpleImputer
    num_cols = artifacts["num_cols"]
    for col in num_cols:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")
        else:
            df_copy[col] = np.nan

    # Ensure all expected categorical columns are present
    cat_cols = artifacts.get("cat_cols", [])
    for col in cat_cols:
        if col not in df_copy.columns:
            df_copy[col] = "missing"

    transformed = preprocessor.transform(df_copy)
    return transformed


def save_artifacts(artifacts: Dict[str, Any], path: str) -> None:
    """Save preprocessor artifacts to disk."""
    joblib.dump(artifacts, path)


def load_artifacts(path: str) -> Dict[str, Any]:
    """Load preprocessor artifacts from disk."""
    return joblib.load(path)