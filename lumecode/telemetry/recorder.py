"""轻量 span 记录器。

设计目标：

1. **零依赖**：不引入 opentelemetry-sdk 或 Langfuse SDK，CLI 启动开销不变。
2. **不阻塞主链路**：span 通过后台线程写入 JSONL，队列满时直接丢弃。
3. **可移植**：字段命名对齐 OTel / Langfuse，后续加 exporter 不用改埋点。

典型用法::

    recorder = TelemetryRecorder(session_id="abc", base_dir=".lumecode/traces")
    with recorder.start_span(SpanKind.GENERATION, "claude-sonnet", model=model) as span:
        ...
        span.end(usage=Usage(input_tokens=100, output_tokens=50))
    recorder.close()
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import queue
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

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
from lumecode.telemetry.pricing import PricingTable
from lumecode.telemetry.store import TraceStore

log = logging.getLogger(__name__)

DEFAULT_MAX_QUEUE = 4096
DEFAULT_MAX_CONTENT_CHARS = 200_000
FLUSH_TIMEOUT_SECONDS = 2.0

_UNSET = object()

_current_context: contextvars.ContextVar[TraceContext | None] = contextvars.ContextVar(
    "lumecode_telemetry_context", default=None
)


def _span_record(span: Span) -> dict[str, Any]:
    return {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "type": "span",
        **span.to_dict(),
    }


def _json_safe(value: Any) -> Any:
    """把任意 input/output 转成可 JSON 序列化的结构。

    telemetry 绝不能因为内容不可序列化而丢 span；退化路径优先保留字符串和
    结构，最后用 ``repr`` 兜底。
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return repr(value)
    return value


class _JsonlWriter:
    """后台线程写入器。所有异常都被吞掉并记录 debug，保证不影响主流程。"""

    def __init__(self, store: TraceStore, max_queue: int) -> None:
        self._store = store
        self._max_queue = max_queue
        self._queue: queue.SimpleQueue[tuple[str, dict[str, Any]] | None] = queue.SimpleQueue()
        self._condition = threading.Condition()
        self._pending = 0
        self._closed = False
        self._drops = 0
        self._thread = threading.Thread(target=self._run, name="lumecode-telemetry", daemon=True)
        self._thread.start()

    @property
    def drops(self) -> int:
        with self._condition:
            return self._drops

    def submit(self, session_id: str, record: dict[str, Any]) -> bool:
        with self._condition:
            if self._closed or self._pending >= self._max_queue:
                self._drops += 1
                return False
            self._pending += 1
            self._queue.put((session_id, record))
            return True

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            session_id, record = item
            try:
                self._store.append_record(session_id, record)
            except Exception as e:
                log.debug("telemetry writer failed: %s", e)
            finally:
                with self._condition:
                    self._pending -= 1
                    self._condition.notify_all()

    def flush(self, timeout: float = FLUSH_TIMEOUT_SECONDS) -> bool:
        with self._condition:
            if self._pending == 0:
                return True
            self._condition.wait_for(lambda: self._pending == 0, timeout=timeout)
            return self._pending == 0

    def close(self, timeout: float = FLUSH_TIMEOUT_SECONDS) -> None:
        # 先让队列排空，再投递哨兵终止线程。flush 超时也继续关闭，避免挂死。
        self.flush(timeout)
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._queue.put(None)
        self._thread.join(timeout)


class SpanHandle:
    """由 ``TelemetryRecorder.start_span`` 返回的可结束句柄。

    既支持 ``with`` 自动结束，也支持手动 ``end()`` 提前结束。重复结束是 no-op。
    """

    def __init__(
        self,
        recorder: TelemetryRecorder | None,
        span: Span | None,
        token: contextvars.Token[TraceContext | None] | None,
    ) -> None:
        self._recorder = recorder
        self._span = span
        self._token = token
        self._ended = False

    @property
    def span(self) -> Span | None:
        return self._span

    @property
    def context(self) -> TraceContext | None:
        return self._span.context if self._span is not None else None

    def end(
        self,
        *,
        status: SpanStatus | None = None,
        attributes: dict[str, Any] | None = None,
        usage: Usage | None = None,
        model: str | None = None,
        input: Any = _UNSET,
        output: Any = _UNSET,
        error: str | None = None,
    ) -> None:
        if self._ended:
            return
        self._ended = True
        if self._token is not None:
            # token 可能已在别的上下文中被 reset（例如跨 task），忽略即可。
            with contextlib.suppress(ValueError):
                _current_context.reset(self._token)
        if self._recorder is not None and self._span is not None:
            self._recorder._finish(
                self._span,
                status=status,
                attributes=attributes,
                usage=usage,
                model=model,
                input=input,
                output=output,
                error=error,
            )

    def __enter__(self) -> SpanHandle:
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> None:
        # 消费方提前 break/close 生成器时也会走这里；GeneratorExit 不是错误。
        if exc is not None and not isinstance(exc, GeneratorExit):
            self.end(status=SpanStatus.ERROR, error=str(exc))
        else:
            self.end()

    def discard(self) -> None:
        """丢弃这个 span（例如“检查了但没触发”的压缩尝试），不落盘。"""
        if self._ended:
            return
        self._ended = True
        if self._token is not None:
            with contextlib.suppress(ValueError):
                _current_context.reset(self._token)


class TelemetryRecorder:
    """一个会话的 telemetry 记录器。

    Args:
        session_id: 会话 ID，写入的文件名由此派生。
        base_dir: trace 目录，默认 ``.lumecode/traces``。
        enabled: 关闭时所有 span 都是 no-op。
        capture_content: False 时不落 prompt / 工具输出等内容。
        max_content_chars: 单条 input/output 的最大字符数，超出截断。
        pricing: 自定义价格表；默认使用 ``DEFAULT_PRICES``。
        max_queue: 待写队列上限，超过后丢弃新 span。
    """

    def __init__(
        self,
        session_id: str,
        *,
        base_dir: str | os.PathLike[str] = ".lumecode/traces",
        store: TraceStore | None = None,
        enabled: bool | None = None,
        capture_content: bool = True,
        max_content_chars: int = DEFAULT_MAX_CONTENT_CHARS,
        pricing: PricingTable | None = None,
        max_queue: int = DEFAULT_MAX_QUEUE,
    ) -> None:
        self.session_id = session_id
        if enabled is None:
            enabled = os.environ.get("LUMECODE_TELEMETRY", "1").strip().lower() not in {
                "0",
                "false",
                "off",
                "no",
            }
        self.enabled = enabled
        self.capture_content = capture_content
        self.max_content_chars = max_content_chars
        self.pricing = pricing or PricingTable()
        self._store = store or TraceStore(base_dir)
        self._writer: _JsonlWriter | None = (
            _JsonlWriter(self._store, max_queue) if enabled else None
        )
        self._trace_id = new_trace_id()
        self._closed = False

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @property
    def store(self) -> TraceStore:
        """底层 trace store，供 /trace 等只读命令使用。"""
        return self._store

    @property
    def dropped_spans(self) -> int:
        return self._writer.drops if self._writer is not None else 0

    def current_context(self) -> TraceContext | None:
        return _current_context.get()

    @contextmanager
    def use_context(self, context: TraceContext | None) -> Iterator[None]:
        """显式绑定父上下文，用于跨 asyncio task / 线程传递父 span。"""
        token = _current_context.set(context)
        try:
            yield
        finally:
            _current_context.reset(token)

    def start_span(
        self,
        kind: SpanKind,
        name: str,
        *,
        parent: TraceContext | SpanHandle | None = None,
        attributes: dict[str, Any] | None = None,
        model: str | None = None,
        input: Any = None,
        set_current: bool = True,
    ) -> SpanHandle:
        """创建一个 span，并把它设为当前上下文。

        ``parent`` 为空时自动使用当前上下文；当前没有父节点时开启一个新 trace。
        返回的 ``SpanHandle`` 同时支持 ``with`` 和手动 ``end()``。

        ``set_current=False`` 用于已经有显式父节点的叶子 span：不修改
        contextvar，避免异常路径把上下文泄漏到后续调用。
        """
        if not self.enabled:
            return SpanHandle(None, None, None)

        if not isinstance(kind, SpanKind):
            kind = SpanKind(kind)

        parent_ctx = _as_context(parent)
        if parent_ctx is None:
            parent_ctx = _current_context.get()
        # 不跨会话复用父节点：同一个进程里可能有多个 recorder（例如远程
        # 多用户），否则会把不同 session 的 trace 拼在一起。
        if parent_ctx is not None and parent_ctx.session_id != self.session_id:
            parent_ctx = None

        if parent_ctx is not None:
            trace_id = parent_ctx.trace_id
            parent_id = parent_ctx.span_id
        else:
            trace_id = new_trace_id()
            parent_id = None

        span = Span(
            trace_id=trace_id,
            span_id=new_span_id(),
            parent_id=parent_id,
            session_id=self.session_id,
            kind=kind,
            name=name,
            attributes=dict(attributes or {}),
            model=model,
            input=self._capture(input),
        )
        token = _current_context.set(span.context) if set_current else None
        return SpanHandle(self, span, token)

    def record_span(self, span: Span) -> bool:
        """直接记录一个已经构造好的 span（用于跨进程/跨线程汇总）。"""
        if not self.enabled or self._writer is None or self._closed:
            return False
        return self._writer.submit(span.session_id, _span_record(span))

    def flush(self, timeout: float = FLUSH_TIMEOUT_SECONDS) -> bool:
        if self._writer is None:
            return True
        return self._writer.flush(timeout)

    def close(self, timeout: float = FLUSH_TIMEOUT_SECONDS) -> None:
        if self._closed:
            return
        self._closed = True
        if self._writer is not None:
            self._writer.close(timeout)

    def __enter__(self) -> TelemetryRecorder:
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _finish(
        self,
        span: Span,
        *,
        status: SpanStatus | None,
        attributes: dict[str, Any] | None,
        usage: Usage | None,
        model: str | None,
        input: Any,
        output: Any,
        error: str | None,
    ) -> None:
        span.end_time_ns = time.time_ns()
        if status is not None:
            span.status = status if isinstance(status, SpanStatus) else SpanStatus(status)
        if error is not None:
            span.error = error
            if span.status == SpanStatus.OK:
                span.status = SpanStatus.ERROR
        if attributes:
            span.attributes.update(attributes)
        if model is not None:
            span.model = model
        if usage is not None:
            span.usage = usage
            span.attributes.setdefault("gen_ai.usage.input_tokens", usage.input_tokens)
            span.attributes.setdefault("gen_ai.usage.output_tokens", usage.output_tokens)
            if usage.cache_read_tokens:
                span.attributes.setdefault(
                    "gen_ai.usage.cache_read_input_tokens", usage.cache_read_tokens
                )
            if usage.cache_creation_tokens:
                span.attributes.setdefault(
                    "gen_ai.usage.cache_creation_input_tokens", usage.cache_creation_tokens
                )
        if span.model:
            span.attributes.setdefault("gen_ai.request.model", span.model)
        if span.usage is not None:
            cost, estimated = self.pricing.estimate(span.model, span.usage)
            span.cost_usd = cost if estimated else None
            span.attributes.setdefault("cost.estimated", estimated)
        if input is not _UNSET:
            span.input = self._capture(input)
        if output is not _UNSET:
            span.output = self._capture(output)

        # 内容来自工具/模型，可能是任意对象；在落盘前统一转成 JSON 安全结构。
        span.attributes = _json_safe(span.attributes)
        span.input = _json_safe(span.input)
        span.output = _json_safe(span.output)

        if self._writer is not None and not self._closed:
            self._writer.submit(span.session_id, _span_record(span))

    def _capture(self, value: Any) -> Any:
        """按配置裁剪内容。capture_content=False 时返回 None。"""
        if not self.capture_content or value is None:
            return None
        if isinstance(value, str) and len(value) > self.max_content_chars:
            return value[: self.max_content_chars] + f"\n...[truncated {len(value)} chars]"
        return value


def _as_context(value: TraceContext | SpanHandle | None) -> TraceContext | None:
    if value is None:
        return None
    if isinstance(value, SpanHandle):
        return value.context
    return value
