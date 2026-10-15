"""
Inference module for Customer Churn Analytics.
Provides functions to load saved model artifacts and predict churn probability
for raw customer records using the standardized preprocessing pipeline.
"""

import functools
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from src import PROJECT_ROOT


@functools.lru_cache(maxsize=1)
def load_artifacts(models_dir: str = "models") -> Tuple[Any, Dict[str, Any]]:
    """
    Loads and caches the trained churn model and preprocessor artifacts dictionary.
    """
    dir_path = Path(models_dir)
    if not dir_path.is_absolute():
        dir_path = PROJECT_ROOT / dir_path

    model_path = dir_path / "churn_model.pkl"
    preprocessor_path = dir_path / "preprocessor.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found at {preprocessor_path}")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    return model, preprocessor


def predict_churn(
    raw_input: Union[Dict[str, Any], pd.Series, pd.DataFrame],
    model: Optional[Any] = None,
    preprocessor: Optional[Dict[str, Any]] = None,
    models_dir: str = "models",
) -> Union[float, np.ndarray]:
    """
    Predicts churn probability for raw customer data.

    Parameters:
        raw_input: A dictionary of feature key-values, a pd.Series, or a pd.DataFrame.
        model: Pre-loaded classifier. If None, loaded via load_artifacts().
        preprocessor: Pre-loaded preprocessor dict. If None, loaded via load_artifacts().
        models_dir: Directory containing model artifacts if loading is required.

    Returns:
        float churn probability for a single customer, or np.ndarray for batch inputs.
    """
    if model is None or preprocessor is None:
        loaded_model, loaded_preprocessor = load_artifacts(models_dir)
        model = model or loaded_model
        preprocessor = preprocessor or loaded_preprocessor

    # Format input into DataFrame
    is_single_row = False
    if isinstance(raw_input, dict):
        df = pd.DataFrame([raw_input])
        is_single_row = True
    elif isinstance(raw_input, pd.Series):
        df = pd.DataFrame([raw_input.to_dict()])
        is_single_row = True
    elif isinstance(raw_input, pd.DataFrame):
        df = raw_input.copy()
        if len(df) == 1:
            is_single_row = True
    else:
        raise TypeError(f"Unsupported input type: {type(raw_input)}. Expected dict, pd.Series, or pd.DataFrame.")

    # Validate preprocessor format (must be ColumnTransformer format)
    if not isinstance(preprocessor, dict) or "preprocessor" not in preprocessor:
        raise ValueError(
            "Preprocessor format not recognized. Expected ColumnTransformer format "
            "with keys: 'preprocessor', 'feature_columns', 'train_medians', 'cat_cols', 'num_cols'. "
            "Retrain model with current src/train.py to regenerate artifacts."
        )
    if not hasattr(preprocessor["preprocessor"], "transform"):
        raise ValueError("Preprocessor['preprocessor'] is not a fitted ColumnTransformer.")

    # Transform using saved pipeline (returns DataFrame with named columns via set_output)
    from src.preprocess import transform_new
    transformed = transform_new(df, preprocessor)

    # Predict probabilities (positive class churn = 1)
    probabilities = model.predict_proba(transformed)[:, 1]

    if is_single_row and len(probabilities) == 1:
        return float(probabilities[0])

    return probabilities