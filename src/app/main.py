"""
API de triagem automática de laudos médicos.

Endpoints:
  POST /predict  -> classifica um laudo em normal / atencao / urgente
  GET  /health   -> healthcheck simples

Observabilidade (OpenTelemetry + OTLP):
  - métricas: enviadas via OTLP para o Prometheus (receiver nativo OTLP)
  - traces:   spans HTTP automáticos (FastAPI) + span manual de inferência
  - logs:     logging estruturado do Python exportado via OTLP para o Loki
"""
import time

from fastapi import FastAPI, HTTPException, Request

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

# --- Instrumentos de métrica (OpenTelemetry) ---
_meter = telemetry.get_meter()

http_requests_total = _meter.create_counter(
    "http.requests.total",
    unit="{request}",
    description="Total de requisições HTTP por método/endpoint/status.",
)
http_request_duration = _meter.create_histogram(
    "http.request.duration",
    unit="s",
    description="Latência das requisições HTTP.",
    explicit_bucket_boundaries_advisory=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
http_errors_total = _meter.create_counter(
    "http.errors.total",
    unit="{error}",
    description="Total de erros HTTP por endpoint.",
)
predictions_total = _meter.create_counter(
    "predictions.total",
    unit="{prediction}",
    description="Total de predições realizadas, por classe.",
)
prediction_duration = _meter.create_histogram(
    "model.predict.duration",
    unit="s",
    description="Tempo de inferência do modelo.",
    explicit_bucket_boundaries_advisory=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
prediction_confidence = _meter.create_histogram(
    "model.predict.confidence",
    description="Confiança das predições (0..1).",
    explicit_bucket_boundaries_advisory=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

_tracer = telemetry.get_tracer()
_logger = telemetry.get_logger(__name__)


@app.middleware("http")
async def otel_metrics_middleware(request: Request, call_next):
    endpoint = request.url.path
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        status = "500"
        http_errors_total.add(1, {"endpoint": endpoint})
        http_requests_total.add(
            1, {"method": request.method, "endpoint": endpoint, "status": status}
        )
        raise
    duration = time.perf_counter() - start
    http_request_duration.record(
        duration, {"method": request.method, "endpoint": endpoint}
    )
    http_requests_total.add(
        1,
        {"method": request.method, "endpoint": endpoint, "status": str(response.status_code)},
    )
    if response.status_code >= 400:
        http_errors_total.add(1, {"endpoint": endpoint})
    return response


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

        predictions_total.add(1, {"classe": classe})
        prediction_duration.record(
            latency_ms / 1000.0, {"backend": result["modelo"], "classe": classe}
        )
        prediction_confidence.record(result["confianca"], {"classe": classe})

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
