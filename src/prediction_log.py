"""
Prediction log for drift monitoring.
Keeps its own SQLite file because data_ingest.py deletes and rebuilds churn.db.
"""

import hashlib
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src import PROJECT_ROOT
from src.explain import ALL_RAW_FEATURES

COLUMNS = ALL_RAW_FEATURES + ["probability", "risk_level", "model_version", "created_at"]


@lru_cache(maxsize=1)
def _model_version() -> str:
    """Short hash of the model file, cached like the model itself. Changes when a new model is promoted."""
    return hashlib.sha256((PROJECT_ROOT / "models" / "churn_model.pkl").read_bytes()).hexdigest()[:12]


def _log_path() -> Path:
    """Set PREDICTION_LOG_PATH to store the log somewhere else."""
    return Path(os.environ.get("PREDICTION_LOG_PATH", PROJECT_ROOT / "data" / "predictions.db"))


def load_predictions(limit: int) -> pd.DataFrame:
    """Return the most recent predictions, oldest first. Empty if nothing has been logged yet."""
    db_path = _log_path()
    if not db_path.exists():
        return pd.DataFrame(columns=COLUMNS)
    with closing(sqlite3.connect(db_path)) as conn:
        recent = pd.read_sql_query("SELECT * FROM predictions ORDER BY rowid DESC LIMIT ?", conn, params=(limit,))
    return recent.iloc[::-1].reset_index(drop=True)


def log_prediction(raw_input: Dict[str, Any], result: Dict[str, Any]) -> None:
    """Append one prediction."""
    db_path = _log_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    values = [raw_input[feature] for feature in ALL_RAW_FEATURES] + [
        result["probability"],
        result["risk_level"],
        _model_version(),
        datetime.now(timezone.utc).isoformat(),
    ]
    columns = ", ".join(COLUMNS)
    placeholders = ", ".join("?" for _ in COLUMNS)

    # shortcut: blocking write per request (about 3 ms), but up to seconds under heavy concurrent writes; move to a background writer if traffic grows
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute(f"CREATE TABLE IF NOT EXISTS predictions ({columns})")
        conn.execute(f"INSERT INTO predictions ({columns}) VALUES ({placeholders})", values)
