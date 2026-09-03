"""
API de triagem automática de laudos médicos.

Endpoints:
  POST /predict  -> classifica um laudo em normal / atencao / urgente
  GET  /health   -> healthcheck simples
  GET  /metrics  -> métricas no formato de exposição do Prometheus

Observabilidade:
  - métricas: prometheus_client em /metrics, coletadas por scrape do Prometheus
  - traces:   spans HTTP automáticos (FastAPI) + span manual de inferência, via OTLP -> Tempo
  - logs:     logging estruturado do Python, via OTLP -> Loki
"""
import time

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from app import telemetry
from app.model_loader import ModelService
from app.schemas import LaudoRequest, LaudoResponse

app = FastAPI(
    title="API de Triagem de Laudos Médicos",
    description="Classifica laudos médicos em normal / atencao / urgente.",
    version="1.0.0",
)

telemetry.setup_telemetry(app)

model_service = ModelService()

# --- Instrumentos de métrica (prometheus_client) ---
# Counter("x") é exposto como "x_total"; Histogram("x") gera x_bucket/_sum/_count.
# Os nomes abaixo são os mesmos usados nas queries de docker/grafana/dashboards/.
http_requests_total = Counter(
    "http_requests",
    "Total de requisições HTTP por método/endpoint/status.",
    ["method", "endpoint", "status"],
)
http_errors_total = Counter(
    "http_errors",
    "Total de erros HTTP por endpoint.",
    ["endpoint"],
)
http_request_duration = Histogram(
    "http_request_duration_seconds",
    "Latência das requisições HTTP.",
    ["method", "endpoint"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
http_active_requests = Gauge(
    "http_server_active_requests",
    "Requisições HTTP em andamento.",
)
predictions_total = Counter(
    "predictions",
    "Total de predições realizadas, por classe.",
    ["classe"],
)
prediction_duration = Histogram(
    "model_predict_duration_seconds",
    "Tempo de inferência do modelo.",
    ["backend", "classe"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
prediction_confidence = Histogram(
    "model_predict_confidence",
    "Confiança das predições (0..1).",
    ["classe"],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

_tracer = telemetry.get_tracer()
_logger = telemetry.get_logger(__name__)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    endpoint = request.url.path
    if endpoint == "/metrics":  # o próprio scrape não deve virar métrica
        return await call_next(request)

    http_active_requests.inc()
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        http_errors_total.labels(endpoint=endpoint).inc()
        http_requests_total.labels(
            method=request.method, endpoint=endpoint, status="500"
        ).inc()
        raise
    finally:
        http_active_requests.dec()
        http_request_duration.labels(
            method=request.method, endpoint=endpoint
        ).observe(time.perf_counter() - start)

    http_requests_total.labels(
        method=request.method, endpoint=endpoint, status=str(response.status_code)
    ).inc()
    if response.status_code >= 400:
        http_errors_total.labels(endpoint=endpoint).inc()
    return response


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    return {"status": "ok", "modelo": model_service.backend}


@app.post("/predict", response_model=LaudoResponse)
def predict(payload: LaudoRequest):
    if not payload.texto.strip():
        _logger.warning("Predição rejeitada: texto vazio.")
        raise HTTPException(status_code=400, detail="Campo 'texto' não pode ser vazio.")

    with _tracer.start_as_current_span("predict") as span:
        result = model_service.predict(payload.texto)
        classe = result["classificacao"]
        latency_ms = result["latencia_ms"]

        span.set_attribute("response.classificacao", classe)
        span.set_attribute("response.confianca", result["confianca"])
        span.set_attribute("modelo", result["modelo"])
        span.set_attribute("latencia_ms", latency_ms)

        predictions_total.labels(classe=classe).inc()
        prediction_duration.labels(backend=result["modelo"], classe=classe).observe(
            latency_ms / 1000.0
        )
        prediction_confidence.labels(classe=classe).observe(result["confianca"])

        _logger.info(
            "Predição realizada",
            extra={
                "classe": classe,
                "confianca": result["confianca"],
                "modelo": result["modelo"],
                "latencia_ms": latency_ms,
            },
        )
        return result
