"""
Model training, evaluation, and MLflow experiment tracking pipeline for Customer Churn Analytics.
Trains Random Forest with class_weight='balanced' (default primary model)
or optional SMOTE resampling, evaluates on untouched test data,
computes held-out metrics and risk thresholds, logs runs and artifacts to MLflow,
and promotes production artifacts to models/ when promote=True.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import build_features, load_data, split_data

DEFAULT_DB_FILE = (PROJECT_ROOT / "mlflow.db").resolve().as_posix()
DEFAULT_TRACKING_URI = f"sqlite:///{DEFAULT_DB_FILE}"
DEFAULT_ARTIFACT_DIR = (PROJECT_ROOT / "mlartifacts").resolve().as_uri()



def get_git_commit_hash() -> str:
    """Retrieves current git commit hash, safely falling back to 'unknown' on error."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def setup_mlflow(tracking_uri: Optional[str] = None) -> str:
    """Initializes MLflow tracking URI, defaulting to project root sqlite:///mlflow.db."""
    uri = tracking_uri if tracking_uri is not None else DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(uri)
    return uri


def get_or_create_experiment(experiment_name: str = "churn-2.0") -> str:
    """Retrieves existing experiment ID or creates a new one with project root artifact path."""
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return mlflow.create_experiment(
            name=experiment_name,
            artifact_location=DEFAULT_ARTIFACT_DIR,
        )
    return exp.experiment_id


def plot_confusion_matrix(cm: List[List[int]], save_path: str) -> None:
    """Plots and saves confusion matrix image using Agg backend without display GUI."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=["No Churn (0)", "Churn (1)"],
        yticklabels=["No Churn (0)", "Churn (1)"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


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
    promote: bool = True,
    tracking_uri: Optional[str] = None,
    experiment_name: str = "churn-2.0",
    run_name: Optional[str] = None,
    n_estimators: int = 300,
    max_depth: Optional[int] = 8,
    model: Optional[Any] = None,
    extra_params: Optional[Dict[str, Any]] = None,
    extra_metrics: Optional[Dict[str, float]] = None,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """
    Executes end-to-end training, evaluation, and MLflow logging:
    - Primary default: Random Forest with class_weight='balanced'
    - Optional: Random Forest with SMOTE oversampling or custom passed estimator
    - Logs params, metrics, confusion matrix, feature names, and sklearn model to MLflow.
    - When promote=True: registers 'churn-rf' with alias 'champion' and writes models/ artifacts.
    - When promote=False: strictly leaves models/ and model registry untouched.
    """
    print(f"1. Loading raw dataset from '{db_path}'...")
    df = load_data(db_path)

    print("2. Splitting data (80/20 stratified)...")
    X_train, X_test, y_train, y_test = split_data(
        df, target_col="churn", test_size=0.2, random_state=42
    )

    print("3. Building features and preprocessing...")
    X_train_proc, X_test_proc, preprocessor = build_features(X_train, X_test)

    # Determine model configuration
    if model is None:
        model_type = "rf_smote" if use_smote else "rf_balanced"
        if use_smote:
            smote = SMOTE(random_state=42)
            X_train_fit, y_train_fit = smote.fit_resample(X_train_proc, y_train)
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_train_fit, y_train_fit)
        else:
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_train_proc, y_train)
    else:
        model_type = type(model).__name__
        if use_smote:
            smote = SMOTE(random_state=42)
            X_train_fit, y_train_fit = smote.fit_resample(X_train_proc, y_train)
            model.fit(X_train_fit, y_train_fit)
        else:
            model.fit(X_train_proc, y_train)

    print(f"\n4. Trained model: {model_type}...")
    print("5. Evaluating on untouched test data...")
    metrics = evaluate_model(model, X_test_proc, y_test)
    metrics["model_type"] = model_type
    metrics["n_estimators"] = getattr(model, "n_estimators", n_estimators)
    metrics["max_depth"] = getattr(model, "max_depth", max_depth)
    metrics["class_weight"] = "balanced" if getattr(model, "class_weight", None) == "balanced" else str(getattr(model, "class_weight", None))
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
    # MLflow Tracking & Logging
    # -------------------------------------------------------------
    setup_mlflow(tracking_uri)
    exp_id = get_or_create_experiment(experiment_name)
    actual_run_name = run_name or (f"{model_type}_promote" if promote else model_type)

    with mlflow.start_run(experiment_id=exp_id, run_name=actual_run_name) as run:
        # Params
        params_to_log = {
            "model_type": model_type,
            "n_estimators": str(metrics["n_estimators"]),
            "max_depth": str(metrics["max_depth"]),
            "class_weight": str(metrics["class_weight"]),
            "use_smote": str(use_smote),
            "test_size": 0.2,
            "random_state": 42,
            "number_of_features": int(X_train_proc.shape[1]),
            "train_row_count": int(len(X_train)),
            "test_row_count": int(len(X_test)),
        }
        if extra_params:
            params_to_log.update({k: str(v) for k, v in extra_params.items()})
        mlflow.log_params(params_to_log)

        # Metrics
        metrics_to_log = {
            "auc": metrics["auc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "accuracy": metrics["accuracy"],
            "top_10_decile_capture_rate": metrics["top_10_decile_capture_rate"],
            "threshold_high_risk": metrics["threshold_high_risk"],
            "threshold_moderate_risk": metrics["threshold_moderate_risk"],
        }
        if extra_metrics:
            metrics_to_log.update(extra_metrics)
        mlflow.log_metrics(metrics_to_log)

        # Tags
        mlflow.set_tags({
            "git_commit": get_git_commit_hash(),
            "data_row_count": str(len(df)),
            "promote": str(promote),
        })

        # Artifacts: metrics.json, confusion_matrix.png, feature_names.json
        with tempfile.TemporaryDirectory() as tmp_dir:
            cm_path = os.path.join(tmp_dir, f"confusion_matrix_{run.info.run_id}.png")
            plot_confusion_matrix(metrics["confusion_matrix"], cm_path)
            mlflow.log_artifact(cm_path)

        mlflow.log_dict(metrics, "metrics.json")
        mlflow.log_dict({"feature_columns": preprocessor["feature_columns"]}, "feature_names.json")

        # Model logging with signature & input example
        signature = infer_signature(X_test_proc, model.predict(X_test_proc))
        input_example = X_test_proc.iloc[:5]
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )

        # Registry & Promotion logic
        if promote:
            model_uri = f"runs:/{run.info.run_id}/model"
            reg_model = mlflow.register_model(model_uri=model_uri, name="churn-rf")
            client = MlflowClient()
            client.set_registered_model_alias(
                name="churn-rf", alias="champion", version=reg_model.version
            )
            print(f"\n[Promote] Registered model 'churn-rf' v{reg_model.version} with alias 'champion'.")

            # Save production artifacts to models/
            os.makedirs(output_dir, exist_ok=True)
            model_path = os.path.join(output_dir, "churn_model.pkl")
            preprocessor_path = os.path.join(output_dir, "preprocessor.pkl")
            metrics_path = os.path.join(output_dir, "metrics.json")

            joblib.dump(model, model_path)
            joblib.dump(preprocessor, preprocessor_path)
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)

            print(f"[Promote] Saved clean production artifacts to '{output_dir}':")
            print(f" - {model_path} ({os.path.getsize(model_path):,} bytes)")
            print(f" - {preprocessor_path} ({os.path.getsize(preprocessor_path):,} bytes)")
            print(f" - {metrics_path}")
        else:
            print(f"\n[Experiment] promote=False: production artifacts in '{output_dir}' and registry unchanged.")

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
    parser.add_argument(
        "--no-promote",
        dest="promote",
        action="store_false",
        help="Skip promoting to models/ and registering champion in model registry",
    )
    parser.set_defaults(promote=True)
    args = parser.parse_args()

    train_model(
        db_path=args.db_path,
        output_dir=args.output_dir,
        use_smote=args.smote,
        promote=args.promote,
    )
