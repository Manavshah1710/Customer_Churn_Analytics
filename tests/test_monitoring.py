"""
Tests for data drift monitoring and the prediction log reader.
"""

import pytest

from src import monitoring
from src.explain import ALL_RAW_FEATURES
from src.monitoring import MIN_RECORDS, check_drift, load_reference_and_holdout, simulate_drift
from src.prediction_log import load_predictions, log_prediction


@pytest.fixture(scope="module")
def reference_and_holdout():
    return load_reference_and_holdout()


def test_healthy_holdout_does_not_alert(reference_and_holdout):
    reference, holdout = reference_and_holdout
    summary, _ = check_drift(holdout, reference)

    assert summary["records"] == len(holdout)
    assert summary["alert"] is False
    assert summary["drifted_columns"] == []
    assert {c["column"] for c in summary["columns"]} == set(ALL_RAW_FEATURES)


def test_simulated_drift_alerts_on_the_shifted_columns(reference_and_holdout):
    reference, holdout = reference_and_holdout
    summary, _ = check_drift(simulate_drift(holdout), reference)

    assert summary["alert"] is True
    assert summary["drifted_columns"] == ["monthly_charges", "tenure", "total_charges"]


def test_alert_threshold_holds_at_the_minimum_sample_size(reference_and_holdout):
    """The smallest allowed sample must still separate healthy data from drift."""
    reference, holdout = reference_and_holdout
    sample = holdout.sample(MIN_RECORDS, random_state=0)

    assert check_drift(sample, reference)[0]["alert"] is False
    assert check_drift(simulate_drift(sample), reference)[0]["alert"] is True


def test_too_few_records_are_refused(reference_and_holdout):
    """Small samples produce false alarms, so they must not get a result at all."""
    reference, holdout = reference_and_holdout
    with pytest.raises(ValueError, match=str(MIN_RECORDS)):
        check_drift(holdout.head(MIN_RECORDS - 1), reference)


def test_load_predictions_returns_most_recent_oldest_first(tmp_path, monkeypatch, reference_and_holdout):
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(tmp_path / "log.db"))
    assert load_predictions(limit=2).empty

    _, holdout = reference_and_holdout
    customer = holdout.iloc[0].to_dict()
    for tenure in (1, 2, 3):
        log_prediction({**customer, "tenure": tenure}, {"probability": 0.5, "risk_level": "low"})

    assert load_predictions(limit=2)["tenure"].tolist() == [2, 3]


def test_main_does_not_alert_without_enough_data(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(tmp_path / "missing.db"))

    assert monitoring.main() == 0
    assert "Not enough data" in capsys.readouterr().out


def test_main_exits_1_and_saves_report_on_alert(tmp_path, monkeypatch, reference_and_holdout):
    """Automation relies on the exit code, so drift must produce a failure and a report."""
    _, holdout = reference_and_holdout
    report_path = tmp_path / "report.html"
    monkeypatch.setattr(monitoring, "REPORT_PATH", report_path)
    monkeypatch.setattr(monitoring, "load_predictions", lambda limit: simulate_drift(holdout))

    assert monitoring.main() == 1
    assert report_path.stat().st_size > 0


def test_main_exits_0_on_healthy_data(tmp_path, monkeypatch, reference_and_holdout):
    _, holdout = reference_and_holdout
    monkeypatch.setattr(monitoring, "REPORT_PATH", tmp_path / "report.html")
    monkeypatch.setattr(monitoring, "load_predictions", lambda limit: holdout)

    assert monitoring.main() == 0
