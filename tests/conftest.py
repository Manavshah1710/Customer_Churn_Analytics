"""
Shared pytest setup.
"""

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def isolated_prediction_log(tmp_path_factory):
    """Keep test requests out of the real prediction log."""
    os.environ["PREDICTION_LOG_PATH"] = str(tmp_path_factory.mktemp("logs") / "predictions.db")
