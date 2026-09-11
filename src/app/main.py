"""
API de classificação de condições médicas a partir de abstracts.

Endpoints:
  POST /predict  -> classifica um abstract em 5 condições médicas
  GET  /health   -> healthcheck simples
  GET  /metrics  -> métricas no formato de exposição do Prometheus

Observabilidade:
  - métricas: prometheus_client em /metrics, coletadas por scrape do Prometheus
  - traces:   spans HTTP automáticos (FastAPI) + span manual de inferência,
              via OTLP -> Tempo
  - logs:     logging estruturado do Python, via OTLP -> Loki
"""
import time

from fastapi import FastAPI, HTTPException, Request, Response
from opentelemetry import trace as otel_trace
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from app import telemetry, tracing
from app.model_loader import ModelService
from app.schemas import LaudoRequest, LaudoResponse

app = FastAPI(
    title="API de Classificação de Condições Médicas",
    description=(
        "Classifica abstracts médicos em: cardiovascular diseases, "
        "digestive system diseases, general pathological conditions, "
        "neoplasms, nervous system diseases."
    ),
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
    span = otel_trace.get_current_span()
    tracing.add_event(
        "http.request",
        attributes={
            "http.method": request.method,
            "http.path": endpoint,
            "http.content_length": request.headers.get("content-length"),
        },
        span=span,
    )
    _logger.info(
        "Requisição recebida",
        extra={
            "http.method": request.method,
            "http.path": endpoint,
            "http.content_length": request.headers.get("content-length"),
        },
    )
    try:
        response = await call_next(request)
    except Exception:
        http_errors_total.labels(endpoint=endpoint).inc()
        http_requests_total.labels(
            method=request.method, endpoint=endpoint, status="500"
        ).inc()
        tracing.add_event(
            "http.request.unhandled_error",
            level="error",
            attributes={"http.path": endpoint},
            span=span,
        )
        _logger.exception(
            "Erro não tratado na requisição",
            extra={"http.method": request.method, "http.path": endpoint},
        )
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
        tracing.add_event(
            "http.request.client_error",
            level="warning",
            attributes={"http.path": endpoint, "http.status_code": response.status_code},
            span=span,
        )
        _logger.warning(
            "Requisição finalizada com erro de cliente",
            extra={"http.path": endpoint, "http.status_code": response.status_code},
        )
    else:
        duration_ms = round((time.perf_counter() - start) * 1000, 4)
        tracing.add_event(
            "http.request.done",
            attributes={
                "http.path": endpoint,
                "http.status_code": response.status_code,
                "http.duration_ms": duration_ms,
            },
            span=span,
        )
        _logger.info(
            "Requisição concluída",
            extra={
                "http.method": request.method,
                "http.path": endpoint,
                "http.status_code": response.status_code,
                "http.duration_ms": duration_ms,
            },
        )
    return response


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    return {"status": "ok", "modelo": model_service.backend}


@app.post("/predict", response_model=LaudoResponse)
def predict(payload: LaudoRequest):
    with _tracer.start_as_current_span("predict") as span:
        texto = payload.texto.strip()
        span.set_attribute("input.texto_chars", len(texto))
        span.set_attribute("model.backend", model_service.backend)
        preview = tracing.text_preview(texto)
        if preview is not None:
            span.set_attribute("input.texto", preview)

        if not texto:
            span.set_status(otel_trace.StatusCode.ERROR, description="texto vazio")
            tracing.add_event(
                "predict.validation_error",
                level="warning",
                attributes={"reason": "texto_vazio", "input.texto_chars": 0},
                span=span,
            )
            _logger.warning(
                "Predição rejeitada: texto vazio.",
                extra={"reason": "texto_vazio", "input.texto_chars": 0},
            )
            raise HTTPException(status_code=400, detail="Campo 'texto' não pode ser vazio.")

        tracing.add_event(
            "predict.validation_ok",
            attributes={"input.texto_chars": len(texto)},
            span=span,
        )
        _logger.info(
            "Payload validado",
            extra={"input.texto_chars": len(texto), "model.backend": model_service.backend},
        )

        result = model_service.predict(texto)
        classe = result["classificacao"]
        latency_ms = result["latencia_ms"]

        span.set_attribute("response.classificacao", classe)
        span.set_attribute("response.confianca", result["confianca"])
        span.set_attribute("modelo", result["modelo"])
        span.set_attribute("latencia_ms", latency_ms)

        if result["confianca"] < 0.5:
            tracing.add_event(
                "predict.low_confidence",
                level="warning",
                attributes={
                    "classe": classe,
                    "confianca": result["confianca"],
                },
                span=span,
            )
            _logger.warning(
                "Confiança abaixo de 0.5",
                extra={"classe": classe, "confianca": result["confianca"]},
            )

        predictions_total.labels(classe=classe).inc()
        prediction_duration.labels(backend=result["modelo"], classe=classe).observe(
            latency_ms / 1000.0
        )
        prediction_confidence.labels(classe=classe).observe(result["confianca"])

        tracing.add_event(
            "predict.completed",
            attributes={
                "classe": classe,
                "confianca": result["confianca"],
                "modelo": result["modelo"],
                "latencia_ms": latency_ms,
            },
            span=span,
        )
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
