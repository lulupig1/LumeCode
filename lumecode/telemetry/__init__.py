"""LumeCode 的本地 telemetry 子系统。

本轮只提供本地 JSONL trace，不引入任何外部依赖。字段命名对齐
OpenTelemetry / Langfuse，后续添加 OTLP 或 Langfuse exporter 时只需要
实现新的导出层，不需要改动埋点。
"""

from lumecode.telemetry.models import (
    TELEMETRY_SCHEMA_VERSION,
    Span,
    SpanKind,
    SpanStatus,
    TraceContext,
    Usage,
    new_span_id,
    new_trace_id,
)
from lumecode.telemetry.pricing import DEFAULT_PRICES, ModelPrice, PricingTable
from lumecode.telemetry.recorder import SpanHandle, TelemetryRecorder
from lumecode.telemetry.store import (
    DEFAULT_RETENTION_DAYS,
    TRACES_DIR,
    TraceFileInfo,
    TraceStore,
    sanitize_session_id,
)
from lumecode.telemetry.view import (
    build_span_tree,
    format_cost_report,
    format_session_list,
    format_span_line,
    format_trace_tree,
)

__all__ = [
    "DEFAULT_PRICES",
    "DEFAULT_RETENTION_DAYS",
    "TELEMETRY_SCHEMA_VERSION",
    "TRACES_DIR",
    "ModelPrice",
    "PricingTable",
    "Span",
    "SpanHandle",
    "SpanKind",
    "SpanStatus",
    "TelemetryRecorder",
    "TraceContext",
    "TraceFileInfo",
    "TraceStore",
    "Usage",
    "build_span_tree",
    "format_cost_report",
    "format_session_list",
    "format_span_line",
    "format_trace_tree",
    "new_span_id",
    "new_trace_id",
    "sanitize_session_id",
]
