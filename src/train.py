"""
Model training and evaluation pipeline for Customer Churn Analytics.
Trains Random Forest with class_weight='balanced' (default primary model)
or optional SMOTE resampling, evaluates on untouched test data,
computes held-out metrics and risk thresholds, and saves model artifacts to models/.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import build_features, load_data, split_data


def calculate_top_decile_capture(
    y_true: pd.Series, y_prob: np.ndarray
) -> Tuple[float, int, int]:
    """
    Calculates the percentage of actual positive instances captured
    within the top 10% highest predicted risk instances.
    """
    n_top_10 = int(np.ceil(len(y_true) * 0.10))
    top_indices = np.argsort(y_prob)[::-1][:n_top_10]
    churners_caught = int(y_true.iloc[top_indices].sum())
    total_churners = int(y_true.sum())
    capture_rate = (churners_caught / total_churners * 100.0) if total_churners > 0 else 0.0
    return capture_rate, churners_caught, total_churners


def evaluate_model(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, Any]:
    """
    Evaluates a trained classifier on untouched test data and returns a metrics dictionary,
    including top-decile capture rate and percentile-based risk cutoffs.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = float(roc_auc_score(y_test, y_prob))
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()
    capture_rate, caught, total = calculate_top_decile_capture(y_test, y_prob)

    # Compute risk thresholds on test set distribution:
    # High risk = top 10% highest predicted probability (90th percentile)
    # Moderate risk = top 30% highest predicted probability (70th percentile)
    cutoff_top_10 = float(np.percentile(y_prob, 90))
    cutoff_top_30 = float(np.percentile(y_prob, 70))

    metrics = {
        "auc": round(auc, 4),
        "precision": round(report["1"]["precision"], 4),
        "recall": round(report["1"]["recall"], 4),
        "f1": round(report["1"]["f1-score"], 4),
        "accuracy": round(report["accuracy"], 4),
        "confusion_matrix": cm,
        "top_10_decile_capture_rate": round(capture_rate, 2),
        "churners_caught_in_top_10": caught,
        "total_churners_in_test": total,
        "test_support_0": int(report["0"]["support"]),
        "test_support_1": int(report["1"]["support"]),
        "threshold_high_risk": round(cutoff_top_10, 4),
        "threshold_moderate_risk": round(cutoff_top_30, 4),
    }
    return metrics


def train_model(
    db_path: str = "data/churn.db",
    output_dir: str = "models",
    use_smote: bool = False,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """
    Executes end-to-end training and evaluation:
    - Primary default: Random Forest with class_weight='balanced'
    - Optional: Random Forest with SMOTE oversampling
    Saves clean artifacts (churn_model.pkl, preprocessor.pkl, metrics.json) in output_dir.
    """
    print(f"1. Loading raw dataset from '{db_path}'...")
    df = load_data(db_path)

    print("2. Splitting data (80/20 stratified)...")
    X_train, X_test, y_train, y_test = split_data(df, target_col="churn", test_size=0.2, random_state=42)

    print("3. Building features and preprocessing...")
    X_train_proc, X_test_proc, preprocessor = build_features(X_train, X_test)

    model_type = "rf_smote" if use_smote else "rf_balanced"
    print(f"\n4. Training model: {model_type}...")

    if use_smote:
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train_proc, y_train)
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train_res, y_train_res)
    else:
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train_proc, y_train)

    print("5. Evaluating on untouched test data...")
    metrics = evaluate_model(model, X_test_proc, y_test)
    metrics["model_type"] = model_type
    metrics["n_estimators"] = 300
    metrics["max_depth"] = 12
    metrics["class_weight"] = "balanced" if not use_smote else None
    metrics["use_smote"] = use_smote

    print("\n" + "=" * 60)
    print("                 MODEL EVALUATION METRICS")
    print("=" * 60)
    print(f"Model Type:                 {model_type}")
    print(f"AUC:                        {metrics['auc']:.4f}")
    print(f"Precision (Class 1):        {metrics['precision']:.4f}")
    print(f"Recall (Class 1):           {metrics['recall']:.4f}")
    print(f"F1 Score (Class 1):         {metrics['f1']:.4f}")
    print(f"Accuracy:                   {metrics['accuracy']:.4f}")
    print(f"Top 10% Decile Capture:     {metrics['top_10_decile_capture_rate']:.2f}% ({metrics['churners_caught_in_top_10']}/{metrics['total_churners_in_test']})")
    print(f"High Risk Threshold (P90):  {metrics['threshold_high_risk']:.4f}")
    print(f"Moderate Risk Cutoff (P70): {metrics['threshold_moderate_risk']:.4f}")
    print(f"Confusion Matrix:\n{np.array(metrics['confusion_matrix'])}")
    print("=" * 60)

    # -------------------------------------------------------------
    # Save clean artifacts
    # -------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "churn_model.pkl")
    preprocessor_path = os.path.join(output_dir, "preprocessor.pkl")
    metrics_path = os.path.join(output_dir, "metrics.json")

    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nSaved artifacts to '{output_dir}':")
    print(f" - {model_path} ({os.path.getsize(model_path):,} bytes)")
    print(f" - {preprocessor_path} ({os.path.getsize(preprocessor_path):,} bytes)")
    print(f" - {metrics_path}")

    return model, preprocessor, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train customer churn prediction model.")
    parser.add_argument(
        "--smote",
        action="store_true",
        help="Train with SMOTE oversampling instead of class_weight='balanced'",
    )
    parser.add_argument(
        "--db-path",
        default="data/churn.db",
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory to save model artifacts",
    )
    args = parser.parse_args()

    train_model(db_path=args.db_path, output_dir=args.output_dir, use_smote=args.smote)
