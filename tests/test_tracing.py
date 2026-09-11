import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app import tracing


@pytest.fixture(scope="module")
def exporter():
    # O SDK só permite definir um TracerProvider global uma única vez; por isso
    # compartilhamos um único provider/exporter em todo o módulo e apenas
    # limpamos entre testes.
    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    previous = trace.get_tracer_provider()
    trace._set_tracer_provider(provider, log=False)
    yield exp
    trace._set_tracer_provider(previous, log=False)


@pytest.fixture()
def clean_exporter(exporter):
    exporter.clear()
    return exporter


def test_enabled_defaults_to_info():
    # TRACING_LEVEL não setado -> verbosidade INFO (eventos de todas as fases).
    assert tracing.enabled("error")
    assert tracing.enabled("warning")
    assert tracing.enabled("info")


def test_enabled_respects_verbosity(monkeypatch):
    monkeypatch.setattr(tracing, "_VERBOSITY", tracing.Level.ERROR)
    assert tracing.enabled("error")
    assert not tracing.enabled("warning")
    assert not tracing.enabled("info")


def test_enabled_unknown_level_is_false():
    assert not tracing.enabled("debug")


def test_add_event_is_noop_when_otel_disabled():
    # OTEL_ENABLED=false (conftest) -> span atual é NoOp e add_event não anexa nada.
    assert tracing.add_event("algo", attributes={"a": 1}) is None


def test_add_event_suppressed_below_verbosity(clean_exporter, monkeypatch):
    monkeypatch.setattr(tracing, "_VERBOSITY", tracing.Level.WARNING)
    with tracing.trace_step("phase") as span:
        add_ok = tracing.add_event("algo", attributes={"a": 1}, span=span)
        warn_ok = tracing.add_event("aviso", level="warning", attributes={"a": 1}, span=span)
    assert warn_ok is not None
    assert add_ok is None
    spans = clean_exporter.get_finished_spans()
    events = [e.name for e in spans[0].events]
    assert "algo" not in events
    assert "aviso" in events


def test_trace_step_emits_start_and_ok_events(clean_exporter):
    with tracing.trace_step("model.vectorize", attributes={"backend": "onnx"}) as span:
        assert span.is_recording()
        span.set_attribute("features", 100)
    spans = clean_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "model.vectorize"
    assert span.attributes["backend"] == "onnx"
    assert span.attributes["features"] == 100
    assert "phase.latency_ms" in span.attributes
    events = {e.name: e.attributes for e in span.events}
    assert "model.vectorize.start" in events
    assert "model.vectorize.ok" in events
    assert events["model.vectorize.ok"]["phase.latency_ms"] == span.attributes["phase.latency_ms"]


def test_trace_step_records_exception(clean_exporter):
    with pytest.raises(ValueError, match="quebrou"):
        with tracing.trace_step("model.inference"):
            raise ValueError("quebrou")
    spans = clean_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code.name == "ERROR"
    names = [e.name for e in span.events]
    assert "model.inference.error" in names
    assert "exception" in names
    exc_event = next(e for e in span.events if e.name == "exception")
    assert exc_event.attributes["exception.message"] == "quebrou"


def test_trace_step_nests_as_child(clean_exporter):
    with tracing.trace_step("predict"):
        with tracing.trace_step("model.postprocess"):
            pass
    spans = clean_exporter.get_finished_spans()
    by_name = {s.name: s for s in spans}
    assert by_name["predict"].parent is None
    assert by_name["model.postprocess"].parent.span_id == by_name["predict"].context.span_id


def test_text_preview_respects_payload_flag(monkeypatch):
    monkeypatch.setattr(tracing, "LOG_PAYLOAD", False)
    assert tracing.text_preview("qualquer texto") is None

    monkeypatch.setattr(tracing, "LOG_PAYLOAD", True)
    monkeypatch.setattr(tracing, "MAX_TEXT_CHARS", 5)
    assert tracing.text_preview("abcdefgh") == "abcde... [8 chars]"
    assert tracing.text_preview("abc") == "abc"
    assert tracing.text_preview(None) is None
