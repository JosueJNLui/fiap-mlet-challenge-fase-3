import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_urgente():
    response = client.post(
        "/predict",
        json={"texto": "Paciente com dor torácica intensa e falta de ar súbita"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["classificacao"] in {"normal", "atencao", "urgente"}
    assert 0.0 <= body["confianca"] <= 1.0
    assert "probabilidades" in body


def test_predict_texto_vazio():
    response = client.post("/predict", json={"texto": "   "})
    assert response.status_code == 400


def test_predict_campo_ausente():
    response = client.post("/predict", json={})
    assert response.status_code == 422
