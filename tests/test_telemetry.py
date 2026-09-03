import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opentelemetry import _logs as otel_logs
from opentelemetry import trace

from app import telemetry


def test_otel_disabled_leaves_noop_providers():
    # OTEL_ENABLED=false (definido no conftest) -> a telemetria nunca é inicializada.
    # O SDK expõe providers Proxy/NoOp até set_tracer_provider/set_logger_provider.
    assert _is_noop(trace.get_tracer_provider())
    assert _is_noop(otel_logs.get_logger_provider())


def _is_noop(provider) -> bool:
    name = provider.__class__.__name__
    return "NoOp" in name or "Proxy" in name


def test_helpers_return_objects_in_disabled_mode():
    assert telemetry.get_tracer() is not None
    assert telemetry.get_logger() is not None
    telemetry.shutdown_telemetry()  # deve ser seguro na configuracao NoOp
