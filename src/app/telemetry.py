"""
Configuração do OpenTelemetry (traces e logs).

As métricas NÃO passam por aqui: elas são instrumentadas com `prometheus_client`
e expostas em `GET /metrics` (ver src/app/main.py), que é o caminho pedido pelo
enunciado e o que o Prometheus coleta por scrape.

Os dois sinais restantes são exportados via OTLP HTTP diretamente da API:

  - Logs   -> Loki  (ingestão OTLP nativa em /otlp/v1/logs)
  - Traces -> Tempo (ingestão OTLP HTTP em /v1/traces)

Os endpoints são lidos das variáveis de ambiente OTEL_EXPORTER_OTLP_<SINAL>_ENDPOINT
(padrão: nomes dos serviços no docker-compose).

A telemetria pode ser desabilitada com OTEL_ENABLED=false (padrão: true).
"""
import atexit
import logging
import os

from fastapi import FastAPI
from opentelemetry import _logs as otel_logs
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.attributes.service_attributes import SERVICE_NAME, SERVICE_VERSION
from opentelemetry.semconv._incubating.attributes.deployment_attributes import (
    DEPLOYMENT_ENVIRONMENT,
)

_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "triagem-api")
_LIBRARY_VERSION = "1.0.0"

# O scrape do Prometheus bate em /metrics a cada 5s; sem esta exclusão cada
# coleta viraria um trace no Tempo e afogaria os spans que interessam.
_EXCLUDED_URLS = "metrics"

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
    """Inicializa traces e logs e instrumenta o FastAPI.

    No-op quando OTEL_ENABLED=false (útil em testes sem os backends).
    """
    if os.getenv("OTEL_ENABLED", "true").strip().lower() != "true":
        logging.getLogger(_SERVICE_NAME).info(
            "OpenTelemetry desabilitado (OTEL_ENABLED=false)."
        )
        return

    resource = _build_resource()
    _setup_traces(resource)
    _setup_logs(resource)
    FastAPIInstrumentor.instrument_app(app, excluded_urls=_EXCLUDED_URLS)
    atexit.register(shutdown_telemetry)


def shutdown_telemetry() -> None:
    """Faz flush e encerra os providers. Idempotente; chamado no exit do processo."""
    for provider in (trace.get_tracer_provider(), otel_logs.get_logger_provider()):
        flush = getattr(provider, "force_flush", None)
        if callable(flush):
            flush()
        shutdown = getattr(provider, "shutdown", None)
        if callable(shutdown):
            shutdown()


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
