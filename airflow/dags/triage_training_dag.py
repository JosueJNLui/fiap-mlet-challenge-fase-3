"""
DAG Airflow: pipeline de (re)treino do classificador de triagem de laudos.

Fluxo:
    load_data >> train_model >> export_onnx >> validate_model

Cada task é implementada com PythonOperator chamando os scripts em src/,
simulando um pipeline real de MLOps (ingestão -> treino -> exportação ->
validação de artefatos antes de disponibilizar o novo modelo para a API).

Para rodar localmente:
    export AIRFLOW_HOME=~/airflow
    airflow db init
    cp airflow/dags/triage_training_dag.py $AIRFLOW_HOME/dags/
    airflow standalone
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

# Ajuste conforme onde o repositório é montado no worker do Airflow.
PROJECT_ROOT = Path("/opt/airflow/project")


def _run(script: str, args: list[str] | None = None):
    args = args or []
    cmd = [sys.executable, str(PROJECT_ROOT / script), *args]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Falha ao executar {script}: {result.stderr}")


def load_data():
    """Task 1: gera/carrega o CSV de dados de treino."""
    _run("data/generate_data.py")


def train_model():
    """Task 2: treina o pipeline TF-IDF + RandomForest e salva o .joblib."""
    _run("src/train.py", ["--data", "data/laudos.csv", "--out", "models/model.joblib"])


def export_onnx():
    """Task 3: converte o classificador treinado para ONNX (otimização)."""
    _run(
        "src/export_onnx.py",
        ["--model", "models/model.joblib", "--out", "models/model.onnx"],
    )


def validate_model():
    """Task 4: valida que os artefatos foram gerados antes de liberar o deploy."""
    required = [
        PROJECT_ROOT / "models" / "model.joblib",
        PROJECT_ROOT / "models" / "model.onnx",
        PROJECT_ROOT / "models" / "tfidf_vectorizer.joblib",
        PROJECT_ROOT / "models" / "classes.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Artefatos ausentes após treino: {missing}")
    print("Todos os artefatos do modelo foram gerados com sucesso.")


default_args = {
    "owner": "mlops-team",
    "retries": 1,
}

with DAG(
    dag_id="triage_model_training_pipeline",
    description="Pipeline de treino/retreino do classificador de triagem de laudos",
    default_args=default_args,
    schedule="@weekly",  # retreino periódico; pode ser disparado manualmente também
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mlops", "triagem", "nlp"],
) as dag:

    t1_load_data = PythonOperator(task_id="load_data", python_callable=load_data)

    t2_train_model = PythonOperator(task_id="train_model", python_callable=train_model)

    t3_export_onnx = PythonOperator(task_id="export_onnx", python_callable=export_onnx)

    t4_validate_model = PythonOperator(task_id="validate_model", python_callable=validate_model)

    t1_load_data >> t2_train_model >> t3_export_onnx >> t4_validate_model
