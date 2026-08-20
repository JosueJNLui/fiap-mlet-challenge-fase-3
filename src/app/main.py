"""
API de triagem automática de laudos médicos.

Endpoints:
  POST /predict   -> classifica um laudo em normal / atencao / urgente
  GET  /health     -> healthcheck simples
  GET  /metrics    -> métricas Prometheus (contagem de requisições, latência, erros)
"""
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from app.model_loader import ModelService
from app.schemas import LaudoRequest, LaudoResponse

app = FastAPI(
    title="API de Triagem de Laudos Médicos",
    description="Classifica laudos médicos em normal / atencao / urgente.",
    version="1.0.0",
)

model_service = ModelService()

# --- Métricas Prometheus ---
REQUEST_COUNT = Counter(
    "http_requests_total", "Total de requisições HTTP", ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "Latência das requisições HTTP", ["endpoint"]
)
PREDICTION_COUNT = Counter(
    "predictions_total", "Total de predições realizadas, por classe", ["classe"]
)
ERROR_COUNT = Counter(
    "http_errors_total", "Total de erros HTTP", ["endpoint"]
)


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start = time.perf_counter()
    endpoint = request.url.path
    try:
        response = await call_next(request)
    except Exception:
        ERROR_COUNT.labels(endpoint=endpoint).inc()
        REQUEST_COUNT.labels(method=request.method, endpoint=endpoint, status="500").inc()
        raise
    duration = time.perf_counter() - start
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
    REQUEST_COUNT.labels(
        method=request.method, endpoint=endpoint, status=str(response.status_code)
    ).inc()
    if response.status_code >= 400:
        ERROR_COUNT.labels(endpoint=endpoint).inc()
    return response


@app.get("/health")
def health():
    return {"status": "ok", "modelo": model_service.backend}


@app.post("/predict", response_model=LaudoResponse)
def predict(payload: LaudoRequest):
    if not payload.texto.strip():
        raise HTTPException(status_code=400, detail="Campo 'texto' não pode ser vazio.")
    result = model_service.predict(payload.texto)
    PREDICTION_COUNT.labels(classe=result["classificacao"]).inc()
    return result


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
