"""
Configuração do OpenTelemetry (tríade de observabilidade: métricas, logs e traces).

Os três sinais são exportados via OTLP HTTP diretamente da API para os backends:

  - Métricas -> Prometheus (receiver OTLP nativo em /api/v1/otlp/v1/metrics)
  - Logs     -> Loki       (ingestão OTLP nativa em /otlp/v1/logs)
  - Traces   -> Tempo      (ingestão OTLP HTTP em /v1/traces)

Os endpoints são lidos das variáveis de ambiente OTEL_EXPORTER_OTLP_<SINAL>_ENDPOINT
(padrão: nomes dos serviços no docker-compose).

A telemetria pode ser desabilitada com OTEL_ENABLED=false (padrão: true).
"""
import atexit
import logging
import os

from fastapi import FastAPI
from opentelemetry import _logs as otel_logs
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.attributes.service_attributes import SERVICE_NAME, SERVICE_VERSION
from opentelemetry.semconv._incubating.attributes.deployment_attributes import (
    DEPLOYMENT_ENVIRONMENT,
)

_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "triagem-api")
_LIBRARY_VERSION = "1.0.0"

# Formato dos logs exportados para o Loki. Inclui trace_id/span_id no corpo da
# linha para permitir o salto log -> trace via derived field no datasource Loki.
_OTEL_LOG_FORMAT = (
    "%(asctime)s %(levelname)s [%(name)s] "
    "[trace_id=%(otelTraceID)s span_id=%(otelSpanID)s] - %(message)s"
)


def _build_resource() -> Resource:
    return Resource.create(
        {
            SERVICE_NAME: _SERVICE_NAME,
            SERVICE_VERSION: _LIBRARY_VERSION,
            DEPLOYMENT_ENVIRONMENT: os.getenv(
                "DEPLOYMENT_ENVIRONMENT", "docker"
            ),
        }
    )


def _setup_traces(resource: Resource) -> None:
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)


def _setup_metrics(resource: Resource) -> None:
    reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(),
        export_interval_millis=int(os.getenv("OTEL_METRIC_EXPORT_INTERVAL", "5000")),
    )
    metrics.set_meter_provider(
        MeterProvider(resource=resource, metric_readers=[reader])
    )


def _setup_logs(resource: Resource) -> None:
    provider = LoggerProvider(resource=resource)
    provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter()))
    otel_logs.set_logger_provider(provider)

    # Injetar trace_id/span_id nos records e enviar os logs para o Loki via
    # handler OTLP instalado no logger raiz do Python.
    LoggingInstrumentor().instrument(
        inject_trace_context=True,
        log_code_attributes=True,
        log_handler_level=logging.INFO,
    )
    handler = LoggingInstrumentor._logging_handler
    if handler is not None:
        handler.formatter = logging.Formatter(_OTEL_LOG_FORMAT)
        # Evita que mensagens internas do SDK reentrem no handler (recursão).
        handler.addFilter(
            lambda record: not record.name.startswith("opentelemetry")
        )


def setup_telemetry(app: FastAPI) -> None:
    """Inicializa traces, métricas e logs e instrumenta o FastAPI.

    No-op quando OTEL_ENABLED=false (útil em testes sem os backends).
    """
    if os.getenv("OTEL_ENABLED", "true").strip().lower() != "true":
        logging.getLogger(_SERVICE_NAME).info(
            "OpenTelemetry desabilitado (OTEL_ENABLED=false)."
        )
        return

    resource = _build_resource()
    _setup_traces(resource)
    _setup_metrics(resource)
    _setup_logs(resource)
    FastAPIInstrumentor.instrument_app(app)
    atexit.register(shutdown_telemetry)


def shutdown_telemetry() -> None:
    """Faz flush e encerra os providers. Idempotente; chamado no exit do processo."""
    for provider in (
        trace.get_tracer_provider(),
        metrics.get_meter_provider(),
        otel_logs.get_logger_provider(),
    ):
        flush = getattr(provider, "force_flush", None)
        if callable(flush):
            flush()
        shutdown = getattr(provider, "shutdown", None)
        if callable(shutdown):
            shutdown()


def get_meter(name: str = _SERVICE_NAME, version: str = _LIBRARY_VERSION):
    """Meter usado para criar instrumentos de métrica."""
    return metrics.get_meter(name, version)


def get_tracer(name: str = _SERVICE_NAME, version: str = _LIBRARY_VERSION):
    """Tracer usado para criar spans manuais."""
    return trace.get_tracer(name, version)


def get_logger(name: str = _SERVICE_NAME) -> logging.Logger:
    """Logger do aplicativo; os records são exportados para o Loki via OTLP.

    O Python default é WARNING na raiz; garanta nível INFO para que os logs
    estruturados (ex.: 'Predição realizada') cheguem ao handler OTLP.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
