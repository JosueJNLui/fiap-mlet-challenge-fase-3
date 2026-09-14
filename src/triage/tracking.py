"""Integração com MLflow no DagsHub (tracking + registro de modelos).

O pipeline tenta usar DagsHub; se credenciais não estiverem disponíveis, faz fallback
para tracking local (arquivo SQLite) para não quebrar o CI.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from triage.config import Settings, load_settings


class ClassifierPyfunc(mlflow.pyfunc.PythonModel):
    """Adaptador pyfunc: expõe Pipeline sklearn para o MLflow."""

    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    def predict(self, context, model_input: pd.DataFrame, params=None):  # noqa: ARG002
        """Prevê classes a partir de DataFrame com coluna 'texto'."""
        texts = model_input["texto"].tolist()
        preds = self.pipeline.predict(texts)
        proba = self.pipeline.predict_proba(texts)
        return pd.DataFrame({
            "prediction": preds,
            "probabilities": [dict(zip(self.pipeline.classes_, p)) for p in proba]
        })


def init_mlflow(settings: Settings | None = None) -> str:
    """Aponta o MLflow para o servidor DagsHub ou fallback local.

    Args:
        settings: Configurações do pipeline. Se None, carrega via load_settings().

    Returns:
        URI do tracking server configurado.
    """
    settings = settings or load_settings()

    # Tenta DagsHub se token disponível
    if settings.dagshub_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = settings.dagshub_user or settings.dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = settings.dagshub_token
        uri = f"https://dagshub.com/{settings.dagshub_repo_owner}/{settings.dagshub_repo_name}.mlflow"
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(settings.mlflow.experiment_name)
        return uri

    # Fallback local: SQLite em pasta temporária (não persistente)
    local_dir = Path(tempfile.gettempdir()) / "mlflow_local"
    local_dir.mkdir(parents=True, exist_ok=True)
    local_uri = f"sqlite:///{local_dir}/mlflow.db"
    mlflow.set_tracking_uri(local_uri)
    mlflow.set_experiment(settings.mlflow.experiment_name)
    return local_uri


def start_run(settings: Settings) -> mlflow.ActiveRun:
    """Abre a run no tracking configurado por ``init_mlflow``.

    Se o DagsHub recusar a criação (ex.: 403 de token sem permissão de escrita no
    repositório), descarta o token e refaz a run no fallback local, para que o treino
    não quebre por causa de credencial.
    """
    try:
        return mlflow.start_run()
    except MlflowException as exc:
        if not settings.dagshub_token:
            raise
        print(f"DagsHub recusou a criação da run ({exc}). Usando MLflow local.")
        settings.dagshub_token = None
        print(f"MLflow tracking: {init_mlflow(settings)}")
        return mlflow.start_run()


def log_classifier(pipeline: Pipeline, example: pd.DataFrame, registered_model_name: str) -> None:
    """Loga o classificador como pyfunc, com signature/input_example, e registra no Registry."""
    preds = pipeline.predict(example["texto"].tolist())
    signature = infer_signature(example, pd.DataFrame({"prediction": preds}))
    mlflow.pyfunc.log_model(
        name="model",
        python_model=ClassifierPyfunc(pipeline),
        signature=signature,
        input_example=example,
        registered_model_name=registered_model_name,
    )


def _alias_metric(client: MlflowClient, name: str, alias: str, metric: str) -> float | None:
    """Valor de ``metric`` da versão apontada por ``alias``, ou ``None`` se não existe."""
    try:
        mv = client.get_model_version_by_alias(name, alias)
    except Exception:
        return None
    if mv.run_id is None:
        return None
    return client.get_run(mv.run_id).data.metrics.get(metric)


def promote_to_production(
    registered_model_name: str,
    metric_value: float,
    metric_name: str = "f1_macro",
    settings: Settings | None = None,
) -> None:
    """Promove a última versão do modelo no Model Registry.

    Toda versão vira ``staging``; o alias ``production`` só migra para ela se a
    ``metric_value`` informada bater a da produção atual (ou se ainda não houver produção).
    """
    settings = settings or load_settings()

    # Configura tracking URI se tiver token (DagsHub)
    if settings.dagshub_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = settings.dagshub_user or settings.dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = settings.dagshub_token
        uri = f"https://dagshub.com/{settings.dagshub_repo_owner}/{settings.dagshub_repo_name}.mlflow"
        client = MlflowClient(tracking_uri=uri)
    else:
        client = MlflowClient()

    versions = client.search_model_versions(f"name='{registered_model_name}'")
    if not versions:
        return
    latest = max(versions, key=lambda v: int(v.version))
    client.set_registered_model_alias(registered_model_name, "staging", latest.version)
    current = _alias_metric(client, registered_model_name, "production", metric_name)
    if current is None or metric_value > current:
        client.set_registered_model_alias(registered_model_name, "production", latest.version)
