"""轻量 telemetry 数据模型。

字段命名刻意对齐 OpenTelemetry / Langfuse 语义，方便后续添加 OTLP 或
Langfuse SDK exporter 时不需要改动埋点：

- trace_id / span_id / parent_id 对应 OTel 的 trace/span 关系；
- kind 对应 Langfuse 的 observation type；
- usage 对应 gen_ai.usage.*；
- attributes 预留给 gen_ai.* 等标准属性。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# 本地 JSONL 的 schema 版本。字段变更时递增，读取方可以据此兼容旧记录。
TELEMETRY_SCHEMA_VERSION = 1


def new_trace_id() -> str:
    """生成 32 位十六进制 trace id（与 OTel 的 trace id 宽度一致）。"""
    return uuid.uuid4().hex


def new_span_id() -> str:
    """生成 16 位十六进制 span id（与 OTel 的 span id 宽度一致）。"""
    return uuid.uuid4().hex[:16]


class SpanKind(StrEnum):
    SESSION = "session"
    AGENT = "agent"
    GENERATION = "generation"
    TOOL = "tool"
    COMPACT = "compact"
    MEMORY_RECALL = "memory_recall"
    MEMORY_EXTRACT = "memory_extract"
    MEMORY_CONSOLIDATION = "memory_consolidation"
    SESSION_SUMMARY = "session_summary"
    HOOK = "hook"


class SpanStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class Usage:
    """一次 LLM 调用的 token 用量。

    input_tokens 不包含 cache token：Anthropic 的 cache_read / cache_creation
    是独立字段，OpenAI 的 cached_tokens 在 client 层已经从头 input 中扣除。
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_creation_tokens
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Usage:
        def _int(key: str) -> int:
            value = data.get(key, 0)
            return int(value) if isinstance(value, (int, float)) else 0

        return cls(
            input_tokens=_int("input_tokens"),
            output_tokens=_int("output_tokens"),
            cache_read_tokens=_int("cache_read_tokens"),
            cache_creation_tokens=_int("cache_creation_tokens"),
        )


@dataclass(frozen=True)
class TraceContext:
    """当前 span 的轻量引用，用于在异步任务/子 Agent 之间显式传递父节点。"""

    trace_id: str
    span_id: str
    session_id: str


@dataclass
class Span:
    """一个已经结束或正在进行的调用节点。"""

    trace_id: str
    span_id: str
    parent_id: str | None
    session_id: str
    kind: SpanKind
    name: str
    start_time_ns: int = field(default_factory=time.time_ns)
    end_time_ns: int | None = None
    status: SpanStatus = SpanStatus.OK
    attributes: dict[str, Any] = field(default_factory=dict)
    model: str | None = None
    usage: Usage | None = None
    cost_usd: float | None = None
    input: Any | None = None
    output: Any | None = None
    error: str | None = None

    @property
    def duration_ms(self) -> float | None:
        if self.end_time_ns is None:
            return None
        return round((self.end_time_ns - self.start_time_ns) / 1_000_000, 3)

    @property
    def context(self) -> TraceContext:
        return TraceContext(
            trace_id=self.trace_id,
            span_id=self.span_id,
            session_id=self.session_id,
        )

    def to_dict(self) -> dict[str, Any]:
        kind = self.kind.value if isinstance(self.kind, SpanKind) else str(self.kind)
        status = self.status.value if isinstance(self.status, SpanStatus) else str(self.status)
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "session_id": self.session_id,
            "kind": kind,
            "name": self.name,
            "start_time_ns": self.start_time_ns,
            "end_time_ns": self.end_time_ns,
            "duration_ms": self.duration_ms,
            "status": status,
            "attributes": self.attributes,
            "model": self.model,
            "usage": self.usage.to_dict() if self.usage is not None else None,
            "cost_usd": self.cost_usd,
            "input": self.input,
            "output": self.output,
            "error": self.error,
        }
