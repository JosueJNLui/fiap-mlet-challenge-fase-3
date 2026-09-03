"""Garante que /metrics expõe exatamente as séries usadas pelos dashboards.

Se um nome de métrica mudar no código, este teste quebra antes do painel do
Grafana virar "No data".
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# Nomes consultados em docker/grafana/dashboards/triage_dashboard.json.
SERIES_ESPERADAS = [
    "http_requests_total",
    "http_errors_total",
    "http_request_duration_seconds_bucket",
    "http_server_active_requests",
    "predictions_total",
    "model_predict_duration_seconds_bucket",
    "model_predict_confidence_bucket",
]


def test_metrics_expoe_series_dos_dashboards():
    client.post("/predict", json={"texto": "Paciente estável, exame sem alterações"})
    client.post("/predict", json={"texto": "   "})  # alimenta http_errors_total

    body = client.get("/metrics").text
    for serie in SERIES_ESPERADAS:
        assert serie in body, f"série ausente em /metrics: {serie}"


def test_metrics_nao_conta_o_proprio_scrape():
    client.get("/metrics")
    body = client.get("/metrics").text
    assert 'endpoint="/metrics"' not in body
