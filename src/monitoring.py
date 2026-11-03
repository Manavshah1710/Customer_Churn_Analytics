"""
Data drift monitoring with Evidently: compares recent customer inputs with the training data.
Run `python -m src.monitoring` to check the prediction log. It saves an HTML report and exits 1 on an alert.
Evidently is imported only when a check runs, and is not part of the API image.
"""

import sys
from typing import Any, Dict, Tuple

import pandas as pd

from src import PROJECT_ROOT
from src.prediction_log import load_predictions
from src.preprocess import load_data, split_data
from src.schema import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERICAL_FEATURES

# Measured over 20 healthy held-out samples: 3+ columns drift in 11/20 runs at 200 rows, 2/20 at 300, 0/20 at 500.
MIN_RECORDS = 500
# shortcut: 1 or 2 drifted columns show in the results but do not alert; lower this once real traffic shows the noise level
ALERT_DRIFTED_COLUMNS = 3
# Recent rows only, so old traffic cannot hide new drift
RECENT_RECORDS = 1000

REPORT_PATH = PROJECT_ROOT / "evidently_report.html"


def load_reference_and_holdout() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Training inputs (the reference) and held-out inputs (a healthy baseline), from the model's own split."""
    df = load_data(str(PROJECT_ROOT / "data" / "churn.db"))
    X_train, X_test, _, _ = split_data(df, target_col="churn", test_size=0.2, random_state=42)
    return X_train, X_test


def simulate_drift(df: pd.DataFrame) -> pd.DataFrame:
    """Demo only: lighter users on older handsets, because real drift does not exist in this static dataset."""
    return df.assign(
        eqpdays=df["eqpdays"] * 1.5,
        mou_Mean=df["mou_Mean"] * 0.5,
        mou_cvce_Mean=df["mou_cvce_Mean"] * 0.5,
    )


def check_drift(current: pd.DataFrame, reference: pd.DataFrame) -> Tuple[Dict[str, Any], Any]:
    """Returns a summary (records, alert, drifted_columns, columns) and the Evidently snapshot behind it."""
    if len(current) < MIN_RECORDS:
        raise ValueError(f"Need at least {MIN_RECORDS} records to check drift, got {len(current)}")

    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    definition = DataDefinition(numerical_columns=NUMERICAL_FEATURES, categorical_columns=CATEGORICAL_FEATURES)
    snapshot = Report([DataDriftPreset()], include_tests=True).run(
        Dataset.from_pandas(current[MODEL_FEATURES], data_definition=definition),
        Dataset.from_pandas(reference[MODEL_FEATURES], data_definition=definition),
    )

    results = snapshot.dict()
    drifted = {
        test["metric_config"]["params"]["column"]
        for test in results["tests"]
        if test["status"].value == "FAIL" and "column" in test["metric_config"]["params"]
    }
    columns = [
        {
            "column": metric["config"]["column"],
            "method": metric["config"]["method"],
            "score": round(metric["value"], 4),
            "threshold": metric["config"]["threshold"],
            "drifted": metric["config"]["column"] in drifted,
        }
        for metric in results["metrics"]
        if "column" in metric["config"]
    ]
    columns.sort(key=lambda c: c["score"], reverse=True)

    summary = {
        "records": len(current),
        "alert": len(drifted) >= ALERT_DRIFTED_COLUMNS,
        "drifted_columns": sorted(drifted),
        "columns": columns,
    }
    return summary, snapshot


def main() -> int:
    current = load_predictions(limit=RECENT_RECORDS)
    if len(current) < MIN_RECORDS:
        print(f"Not enough data: {len(current)} logged predictions, need {MIN_RECORDS}.")
        return 0

    reference, _ = load_reference_and_holdout()
    summary, snapshot = check_drift(current, reference)
    snapshot.save_html(str(REPORT_PATH))

    print(f"Checked {summary['records']} predictions. Drifted columns: {summary['drifted_columns'] or 'none'}.")
    print(f"Report saved to {REPORT_PATH}")
    if summary["alert"]:
        print("ALERT: data drift detected.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
