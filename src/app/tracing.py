"""
Configuração de verbosidade dos traces de aplicação.

Cada "ponto" do fluxo da API anexa um evento e/ou atributo ao span ativo (no
node graph do Tempo/Grafana cada span é um nó; eventos e atributos aparecem no
detalhe do nó, descrevendo o que aconteceu naquela etapa). O quanto entra em
cada nó é controlado por environment variables:

  TRACING_LEVEL            error | warning | info   (padrão: info)
                           - error   -> apenas exceções
                           - warning -> erros + avisos (payload vazio, baixa confiança, ...)
                           - info    -> tudo acima + eventos de progresso por fase
  TRACING_LOG_PAYLOAD      true | false             (padrão: false)
                           inclui o texto do abstract (truncado) como atributo do span
  TRACING_MAX_TEXT_CHARS   int                       (padrão: 200)
                           tamanho máximo do texto armazenado em atributo de span

O switch geral de liga/desliga continua sendo OTEL_ENABLED (ver app/telemetry.py);
quando desabilitado, spans viram NoOp e todas as funções aqui são no-op.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from enum import IntEnum
from typing import Any, Iterator

import opentelemetry.trace as otel_trace


class Level(IntEnum):
    ERROR = 1
    WARNING = 2
    INFO = 3


_LEVEL_NAMES = {"error": Level.ERROR, "warning": Level.WARNING, "info": Level.INFO}

_VERBOSITY = _LEVEL_NAMES.get(
    os.getenv("TRACING_LEVEL", "info").strip().lower(), Level.INFO
)
LOG_PAYLOAD = os.getenv("TRACING_LOG_PAYLOAD", "false").strip().lower() == "true"
MAX_TEXT_CHARS = int(os.getenv("TRACING_MAX_TEXT_CHARS", "200"))

_TRACER_NAME = "triagem-api.tracing"
_TRACER_VERSION = "1.0.0"


def enabled(level: str | Level) -> bool:
    """True se o nível pedido deve ser registrado na verbosidade configurada."""
    if isinstance(level, str):
        numeric = _LEVEL_NAMES.get(level.strip().lower())
        if numeric is None:
            return False
    else:
        numeric = level
    return _VERBOSITY >= numeric


def _level_name(level: str | Level) -> str:
    return level if isinstance(level, str) else level.name.lower()


def text_preview(text: str | None) -> str | None:
    """Retorna o texto truncado para atributo de span, ou None se TRACING_LOG_PAYLOAD=false."""
    if not LOG_PAYLOAD or not text:
        return None
    cleaned = text.strip()
    if len(cleaned) <= MAX_TEXT_CHARS:
        return cleaned
    return cleaned[:MAX_TEXT_CHARS] + f"... [{len(cleaned)} chars]"


def add_event(
    name: str,
    *,
    level: str | Level = "info",
    attributes: dict[str, Any] | None = None,
    span: "otel_trace.Span | None" = None,
) -> dict[str, Any] | None:
    """Anexa um evento ao span (atual ou explícito) respeitando a verbosidade.

    Retorna o payload anexado, ou None se o nível foi suprimido / sem span
    gravando (ex.: OTEL_ENABLED=false).
    """
    if not enabled(level):
        return None
    span = span or otel_trace.get_current_span()
    if span is None or not span.is_recording():
        return None
    payload: dict[str, Any] = {"event.level": _level_name(level)}
    if attributes:
        payload.update(attributes)
    span.add_event(name, attributes=payload)
    return payload


def record_exception(exc: BaseException, span: "otel_trace.Span | None" = None) -> None:
    """Registra a exceção no span no formato padrão (evento `exception`)."""
    span = span or otel_trace.get_current_span()
    if span is None or not span.is_recording():
        return
    span.record_exception(exc)


@contextmanager
def trace_step(
    phase: str,
    *,
    name: str | None = None,
    attributes: dict[str, Any] | None = None,
    level: str | Level = "info",
) -> Iterator["otel_trace.Span"]:
    """Abre um span filho para uma fase da execução (um nó no node graph).

    O span ganha automaticamente:
      - evento `<phase>.start`  (se o nível permitir)
      - atributo `phase.latency_ms` e evento `<phase>.ok`  (em sucesso)
      - status ERROR, evento `exception` e evento `<phase>.error`  (em falha)
    """
    tracer = otel_trace.get_tracer(_TRACER_NAME, _TRACER_VERSION)
    span_name = name or phase
    with tracer.start_as_current_span(span_name) as span:
        if attributes:
            span.set_attributes(attributes)
        add_event(
            f"{phase}.start",
            level=level,
            attributes={"phase": phase},
            span=span,
        )
        _start_t = time.perf_counter()
        try:
            yield span
        except BaseException as exc:
            status_desc = str(exc) or type(exc).__name__
            span.set_status(otel_trace.StatusCode.ERROR, description=status_desc)
            span.record_exception(exc)
            add_event(
                f"{phase}.error",
                level="error",
                attributes={
                    "exception.type": type(exc).__name__,
                    "exception.message": status_desc,
                },
                span=span,
            )
            raise
        else:
            latency_ms = round((time.perf_counter() - _start_t) * 1000, 4)
            span.set_attribute("phase.latency_ms", latency_ms)
            add_event(
                f"{phase}.ok",
                level=level,
                attributes={"phase.latency_ms": latency_ms},
                span=span,
            )
