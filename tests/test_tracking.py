import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mlflow
from mlflow.exceptions import MlflowException

from triage import tracking
from triage.config import Settings


def test_start_run_falls_back_to_local_when_dagshub_refuses(monkeypatch):
    settings = Settings(dagshub_token="token-sem-escrita")
    real_start_run = mlflow.start_run
    calls = []

    def refuse_first(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise MlflowException("API request to endpoint /api/2.0/mlflow/runs/create failed with error code 403")
        return real_start_run(*args, **kwargs)

    monkeypatch.setattr(tracking.mlflow, "start_run", refuse_first)

    with tracking.start_run(settings):
        assert mlflow.get_tracking_uri().startswith("sqlite:///")

    assert settings.dagshub_token is None
    assert len(calls) == 2
