"""
Benchmarking and experiment runner for Customer Churn Analytics.
Runs comparable model variants on the exact same train/test split:
HistGradientBoosting at several depths and learning rates, Random Forest, and Logistic Regression.

Computes 5-fold Stratified CV AUC on the training split only (leakage-free),
evaluates once on untouched test data, and logs all runs to MLflow under 'churn-2.0'.
All experiment runs set promote=False so production artifacts in models/ are strictly preserved.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import build_features, build_preprocessor, load_data, split_data
from src.train import (
    evaluate_model,
    get_git_commit_hash,
    get_or_create_experiment,
    plot_confusion_matrix,
    setup_mlflow,
)


def compute_cv_auc(
    model_builder: Callable[[], Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[float, float]:
    """
    Computes 5-fold stratified CV AUC strictly on the training split.
    Refits a fresh ColumnTransformer preprocessor inside each fold to ensure zero data leakage.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_aucs: List[float] = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Fresh preprocessor constructed for this specific training fold
        preprocessor, _ = build_preprocessor(X_tr)
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model_builder()),
        ])

        pipeline.fit(X_tr, y_tr)
        val_probs = pipeline.predict_proba(X_val)[:, 1]
        fold_auc = float(roc_auc_score(y_val, val_probs))
        fold_aucs.append(fold_auc)

    cv_mean = float(np.mean(fold_aucs))
    cv_std = float(np.std(fold_aucs))
    return round(cv_mean, 4), round(cv_std, 4)


def run_single_experiment(
    variant_name: str,
    model_builder: Callable[[], Any],
    params: Dict[str, Any],
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    X_train_proc: pd.DataFrame,
    X_test_proc: pd.DataFrame,
    y_test: pd.Series,
    preprocessor_dict: Dict[str, Any],
    total_rows: int,
    experiment_name: str = "churn-2.0",
    tracking_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs cross-validation, final model fitting, evaluation, and MLflow logging for one variant."""
    print(f"\n---> Running variant: {variant_name}")

    # 1. 5-Fold Cross-Validation on training split only
    print("     Computing 5-fold Stratified CV AUC on training split...")
    cv_mean, cv_std = compute_cv_auc(
        model_builder=model_builder,
        X_train=X_train_raw,
        y_train=y_train,
    )
    print(f"     CV AUC: {cv_mean:.4f} (+/- {cv_std:.4f})")

    # 2. Fit final model on full training split
    print("     Fitting final model on full training set...")
    model = model_builder()
    model.fit(X_train_proc, y_train)

    # 3. Evaluate on untouched held-out test data
    print("     Evaluating on untouched test set...")
    metrics = evaluate_model(model, X_test_proc, y_test)
    metrics["cv_auc_mean"] = cv_mean
    metrics["cv_auc_std"] = cv_std

    # 4. Log run to MLflow (promote=False: never register or overwrite models/)
    setup_mlflow(tracking_uri)
    exp_id = get_or_create_experiment(experiment_name)

    with mlflow.start_run(experiment_id=exp_id, run_name=variant_name) as run:
        # Params
        full_params = {
            "variant_name": variant_name,
            "test_size": 0.2,
            "random_state": 42,
            "number_of_features": int(X_train_proc.shape[1]),
            "train_row_count": int(len(X_train_raw)),
            "test_row_count": int(len(y_test)),
        }
        full_params.update(params)
        mlflow.log_params({k: str(v) for k, v in full_params.items()})

        # Metrics
        mlflow.log_metrics({
            "auc": metrics["auc"],
            "cv_auc_mean": cv_mean,
            "cv_auc_std": cv_std,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "accuracy": metrics["accuracy"],
            "top_10_decile_capture_rate": metrics["top_10_decile_capture_rate"],
            "threshold_high_risk": metrics["threshold_high_risk"],
            "threshold_moderate_risk": metrics["threshold_moderate_risk"],
        })

        # Tags
        mlflow.set_tags({
            "git_commit": get_git_commit_hash(),
            "data_row_count": str(total_rows),
            "promote": "false",
        })

        # Artifacts
        with tempfile.TemporaryDirectory() as tmp_dir:
            cm_path = os.path.join(tmp_dir, f"confusion_matrix_{run.info.run_id}.png")
            plot_confusion_matrix(metrics["confusion_matrix"], cm_path)
            mlflow.log_artifact(cm_path)

        mlflow.log_dict(metrics, "metrics.json")
        mlflow.log_dict({"feature_columns": preprocessor_dict["feature_columns"]}, "feature_names.json")

        # Log model with signature
        signature = infer_signature(X_test_proc, model.predict(X_test_proc))
        input_example = X_test_proc.iloc[:5]
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )

    return {
        "Variant": variant_name,
        "Test AUC": metrics["auc"],
        "CV AUC": f"{cv_mean:.4f} ± {cv_std:.4f}",
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1 Score": metrics["f1"],
        "Top-10% Capture": f"{metrics['top_10_decile_capture_rate']:.2f}%",
    }


def run_all_experiments(
    db_path: str = "data/churn.db",
    tracking_uri: Optional[str] = None,
    experiment_name: str = "churn-2.0",
) -> List[Dict[str, Any]]:
    """Runs all comparative model variants on the exact same dataset split."""
    print("=" * 70)
    print("       STARTING MLFLOW EXPERIMENT BENCHMARK SUITE")
    print("=" * 70)
    print(f"Loading data from '{db_path}'...")
    df = load_data(db_path)

    # Identical stratified split for all variants
    X_train_raw, X_test_raw, y_train, y_test = split_data(
        df, target_col="churn", test_size=0.2, random_state=42
    )
    X_train_proc, X_test_proc, preprocessor_dict = build_features(X_train_raw, X_test_raw)

    def hgb(max_iter: int, max_depth: Optional[int], learning_rate: float) -> Dict[str, Any]:
        return {
            "name": f"HistGB (iter={max_iter}, depth={max_depth}, lr={learning_rate})",
            "builder": lambda: HistGradientBoostingClassifier(
                max_iter=max_iter, max_depth=max_depth, learning_rate=learning_rate, random_state=42
            ),
            "params": {
                "model_type": "hist_gradient_boosting",
                "n_estimators": max_iter,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
            },
        }

    def forest(max_depth: int) -> Dict[str, Any]:
        return {
            "name": f"RF Balanced (depth={max_depth})",
            "builder": lambda: RandomForestClassifier(
                n_estimators=300, max_depth=max_depth, class_weight="balanced", random_state=42, n_jobs=-1
            ),
            "params": {
                "model_type": "rf_balanced",
                "n_estimators": 300,
                "max_depth": max_depth,
                "class_weight": "balanced",
            },
        }

    variants = [
        hgb(100, None, 0.1),
        hgb(300, 6, 0.05),
        hgb(500, 4, 0.05),
        hgb(600, 3, 0.05),
        forest(8),
        forest(16),
        {
            "name": "Logistic Regression (balanced)",
            "builder": lambda: LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
            "params": {
                "model_type": "logistic_regression",
                "class_weight": "balanced",
                "max_iter": 1000,
            },
        },
    ]

    results: List[Dict[str, Any]] = []
    for var in variants:
        res = run_single_experiment(
            variant_name=var["name"],
            model_builder=var["builder"],
            params=var["params"],
            X_train_raw=X_train_raw,
            y_train=y_train,
            X_train_proc=X_train_proc,
            X_test_proc=X_test_proc,
            y_test=y_test,
            preprocessor_dict=preprocessor_dict,
            total_rows=len(df),
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
        )
        results.append(res)

    print("\n" + "=" * 80)
    print("                    MLFLOW EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))
    print("=" * 80)
    print("\nNote: All experiment runs were logged with promote=False.")
    print("Production artifacts in 'models/' were not touched.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MLflow churn prediction benchmark experiments.")
    parser.add_argument("--db-path", default="data/churn.db", help="Path to SQLite database")
    args = parser.parse_args()

    run_all_experiments(db_path=args.db_path)
