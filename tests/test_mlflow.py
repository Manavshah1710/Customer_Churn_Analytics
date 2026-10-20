"""
Fast unit and integration tests for MLflow experiment tracking and model registry.
Uses isolated SQLite tracking database and artifact directory via pytest's tmp_path.
"""

from pathlib import Path
from typing import Generator

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from src.train import train_model


@pytest.fixture
def isolated_mlflow(tmp_path: Path) -> Generator[str, None, None]:
    """
    Configures an isolated SQLite tracking URI built from tmp_path with forward slashes.
    Restores the previous tracking URI after the test concludes.
    """
    orig_uri = mlflow.get_tracking_uri()
    db_file = (tmp_path / "test_tracking.db").as_posix()
    test_uri = f"sqlite:///{db_file}"
    mlflow.set_tracking_uri(test_uri)
    yield test_uri
    mlflow.set_tracking_uri(orig_uri)


def test_mlflow_fast_run_creates_expected_params_metrics_artifacts(
    tmp_path: Path, isolated_mlflow: str
) -> None:
    """
    Fast test verifying that train_model logs expected params, metrics, tags,
    and artifacts under the 'churn-2.0' experiment.
    Uses n_estimators=5, max_depth=3 for sub-second execution.
    """
    output_dir = tmp_path / "models_scratch"
    test_tracking_uri = isolated_mlflow

    model, preprocessor, metrics = train_model(
        db_path="data/churn.db",
        output_dir=str(output_dir),
        use_smote=False,
        promote=False,
        tracking_uri=test_tracking_uri,
        experiment_name="churn-2.0",
        n_estimators=5,
        max_depth=3,
    )

    client = MlflowClient(tracking_uri=test_tracking_uri)
    experiment = client.get_experiment_by_name("churn-2.0")
    assert experiment is not None, "Experiment 'churn-2.0' was not created."

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 1, f"Expected 1 run, got {len(runs)}"

    run = runs[0]

    # Verify expected parameters
    expected_params = [
        "model_type",
        "n_estimators",
        "max_depth",
        "class_weight",
        "use_smote",
        "test_size",
        "random_state",
        "number_of_features",
        "train_row_count",
        "test_row_count",
    ]
    for param in expected_params:
        assert param in run.data.params, f"Missing logged parameter: '{param}'"

    assert run.data.params["n_estimators"] == "5"
    assert run.data.params["max_depth"] == "3"

    # Verify expected metrics
    expected_metrics = [
        "auc",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "top_10_decile_capture_rate",
        "threshold_high_risk",
        "threshold_moderate_risk",
    ]
    for metric in expected_metrics:
        assert metric in run.data.metrics, f"Missing logged metric: '{metric}'"
        val = run.data.metrics[metric]
        assert isinstance(val, (int, float)), f"Metric '{metric}' is not numeric: {val}"

    assert 0.0 <= run.data.metrics["auc"] <= 1.0, f"AUC out of range: {run.data.metrics['auc']}"

    # Verify tags
    assert "git_commit" in run.data.tags
    assert "data_row_count" in run.data.tags
    assert run.data.tags["promote"].lower() == "false"

    # Verify logged artifacts
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert "model" in artifacts, "Logged sklearn model artifact missing"
    assert "metrics.json" in artifacts, "metrics.json artifact missing"
    assert "feature_names.json" in artifacts, "feature_names.json artifact missing"
    assert any("confusion_matrix" in a for a in artifacts), "confusion matrix artifact missing"

    # Verify promote=False does NOT write to output_dir
    assert not (output_dir / "churn_model.pkl").exists()


def test_mlflow_promotion_registers_champion(
    tmp_path: Path, isolated_mlflow: str
) -> None:
    """
    Verifies that when promote=True:
    1. The model is registered as 'churn-rf'.
    2. The 'champion' alias is assigned to the registered version.
    3. Production artifacts (churn_model.pkl, preprocessor.pkl, metrics.json) are written to output_dir.
    """
    output_dir = tmp_path / "models_prod"
    test_tracking_uri = isolated_mlflow

    model, preprocessor, metrics = train_model(
        db_path="data/churn.db",
        output_dir=str(output_dir),
        use_smote=False,
        promote=True,
        tracking_uri=test_tracking_uri,
        experiment_name="churn-2.0",
        n_estimators=5,
        max_depth=3,
    )

    client = MlflowClient(tracking_uri=test_tracking_uri)

    # Verify model registry & alias
    champion_model = client.get_model_version_by_alias(name="churn-rf", alias="champion")
    assert champion_model is not None, "Model alias 'champion' not found on 'churn-rf'."
    assert champion_model.name == "churn-rf"

    # Verify production files written
    assert (output_dir / "churn_model.pkl").exists()
    assert (output_dir / "preprocessor.pkl").exists()
    assert (output_dir / "metrics.json").exists()
