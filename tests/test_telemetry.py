from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from lumecode.telemetry import (
    ModelPrice,
    PricingTable,
    Span,
    SpanKind,
    SpanStatus,
    TelemetryRecorder,
    TraceStore,
    Usage,
    sanitize_session_id,
)
from lumecode.telemetry.models import new_span_id, new_trace_id

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> TraceStore:
    return TraceStore(tmp_path / "traces")


@pytest.fixture
def recorder(store: TraceStore) -> Any:
    rec = TelemetryRecorder("sess-1", store=store)
    yield rec
    rec.close()


def _records(store: TraceStore, session_id: str = "sess-1") -> list[dict[str, Any]]:
    return store.read(session_id)


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


class TestModels:
    def test_usage_total_tokens(self) -> None:
        usage = Usage(
            input_tokens=10,
            output_tokens=20,
            cache_read_tokens=30,
            cache_creation_tokens=40,
        )
        assert usage.total_tokens == 100

    def test_usage_dict_roundtrip(self) -> None:
        usage = Usage(1, 2, 3, 4)
        assert Usage.from_dict(usage.to_dict()) == usage

    def test_usage_from_dict_tolerates_bad_values(self) -> None:
        usage = Usage.from_dict({"input_tokens": "oops", "output_tokens": 5})
        assert usage.input_tokens == 0
        assert usage.output_tokens == 5

    def test_span_duration(self) -> None:
        span = Span(
            trace_id=new_trace_id(),
            span_id=new_span_id(),
            parent_id=None,
            session_id="s",
            kind=SpanKind.TOOL,
            name="bash",
            start_time_ns=1_000_000_000,
            end_time_ns=1_250_000_000,
        )
        assert span.duration_ms == 250.0

    def test_span_to_dict(self) -> None:
        span = Span(
            trace_id="t",
            span_id="s",
            parent_id=None,
            session_id="sess",
            kind=SpanKind.GENERATION,
            name="model",
            usage=Usage(1, 2),
        )
        data = span.to_dict()
        assert data["kind"] == "generation"
        assert data["usage"]["total_tokens"] == 3
        assert data["trace_id"] == "t"


# ---------------------------------------------------------------------------
# 价格表
# ---------------------------------------------------------------------------


class TestPricing:
    def test_known_model_cost(self) -> None:
        pricing = PricingTable(overrides={"test-model": ModelPrice(1.0, 2.0, 0.1, 0.5)})
        usage = Usage(1_000_000, 500_000, 100_000, 50_000)
        cost, estimated = pricing.estimate("test-model", usage)
        assert estimated is True
        # 1.0 + 1.0 + 0.01 + 0.025
        assert cost == pytest.approx(2.035)

    def test_unknown_model_is_not_reported_as_free(self) -> None:
        cost, estimated = PricingTable().estimate("totally-unknown-model", Usage(1, 1))
        assert cost == 0.0
        assert estimated is False

    def test_longest_substring_wins(self) -> None:
        pricing = PricingTable(
            overrides={
                "acme": ModelPrice(1.0, 1.0),
                "acme-mini": ModelPrice(9.0, 9.0),
            }
        )
        price = pricing.lookup("acme-mini-2026")
        assert price is not None
        assert price.input_per_mtok == 9.0

    def test_no_usage(self) -> None:
        cost, estimated = PricingTable().estimate("claude-sonnet", None)
        assert cost == 0.0
        assert estimated is False


# ---------------------------------------------------------------------------
# 记录器：span 树、上下文、错误
# ---------------------------------------------------------------------------


class TestRecorder:
    def test_nested_spans_share_trace(self, recorder: TelemetryRecorder, store: TraceStore) -> None:
        with recorder.start_span(SpanKind.AGENT, "agent") as root:
            with recorder.start_span(SpanKind.GENERATION, "generation") as generation:
                with recorder.start_span(SpanKind.TOOL, "read_file") as tool:
                    pass

        recorder.flush()
        records = _records(store)
        assert len(records) == 3

        by_kind = {r["kind"]: r for r in records}
        assert by_kind["agent"]["parent_id"] is None
        assert by_kind["generation"]["parent_id"] == by_kind["agent"]["span_id"]
        assert by_kind["tool"]["parent_id"] == by_kind["generation"]["span_id"]
        assert len({r["trace_id"] for r in records}) == 1

        assert generation.span is not None
        assert tool.span is not None
        assert root.span is not None
        assert generation.span.parent_id == root.span.span_id
        assert tool.span.parent_id == generation.span.span_id

    def test_exception_marks_error(self, recorder: TelemetryRecorder, store: TraceStore) -> None:
        with pytest.raises(RuntimeError, match="boom"):
            with recorder.start_span(SpanKind.TOOL, "bash"):
                raise RuntimeError("boom")

        recorder.flush()
        record = _records(store)[0]
        assert record["status"] == "error"
        assert record["error"] == "boom"

    def test_manual_end_then_context_restores(
        self, recorder: TelemetryRecorder, store: TraceStore
    ) -> None:
        with recorder.start_span(SpanKind.AGENT, "agent") as root:
            with recorder.start_span(SpanKind.TOOL, "outer") as outer:
                outer.end()
                with recorder.start_span(SpanKind.TOOL, "inner") as inner:
                    pass
        recorder.flush()
        assert inner.span is not None
        assert root.span is not None
        # 手动 end 后上下文已回到 root，而不是 outer。
        assert inner.span.parent_id == root.span.span_id

    def test_set_current_false_does_not_leak_context(
        self, recorder: TelemetryRecorder, store: TraceStore
    ) -> None:
        with recorder.start_span(SpanKind.AGENT, "agent") as root:
            root_context = root.context
            assert root_context is not None
            for _ in range(2):
                leaf = recorder.start_span(
                    SpanKind.TOOL,
                    "leaf",
                    parent=root_context,
                    set_current=False,
                )
                leaf.end()

            assert recorder.current_context() == root_context

        recorder.flush()
        records = _records(store)
        leaves = [r for r in records if r["name"] == "leaf"]
        assert len(leaves) == 2
        assert all(r["parent_id"] == root_context.span_id for r in leaves)

    def test_explicit_parent_across_context(
        self, recorder: TelemetryRecorder, store: TraceStore
    ) -> None:
        with recorder.start_span(SpanKind.AGENT, "agent") as root:
            root_ctx = root.context

        assert root_ctx is not None
        with recorder.use_context(root_ctx):
            with recorder.start_span(SpanKind.GENERATION, "generation") as child:
                pass

        recorder.flush()
        assert child.span is not None
        assert child.span.parent_id == root_ctx.span_id
        assert child.span.trace_id == root_ctx.trace_id

    def test_parent_from_another_session_is_ignored(self, store: TraceStore) -> None:
        rec1 = TelemetryRecorder("sess-1", store=store)
        rec2 = TelemetryRecorder("sess-2", store=store)
        try:
            with rec1.start_span(SpanKind.AGENT, "agent") as root:
                root_ctx = root.context
            assert root_ctx is not None

            with rec2.use_context(root_ctx):
                with rec2.start_span(SpanKind.TOOL, "tool") as child:
                    pass
        finally:
            rec1.close()
            rec2.close()

        assert child.span is not None
        assert child.span.parent_id is None
        assert child.span.trace_id != root_ctx.trace_id

    @pytest.mark.asyncio
    async def test_context_propagates_into_task(
        self, recorder: TelemetryRecorder, store: TraceStore
    ) -> None:
        with recorder.start_span(SpanKind.AGENT, "agent") as root:

            async def child() -> str | None:
                with recorder.start_span(SpanKind.TOOL, "child") as span:
                    return span.span.parent_id if span.span else None

            parent_id = await asyncio.create_task(child())

        recorder.flush()
        assert root.span is not None
        assert parent_id == root.span.span_id

    def test_disabled_recorder_is_noop(self, store: TraceStore) -> None:
        with TelemetryRecorder("sess-1", store=store, enabled=False) as recorder:
            with recorder.start_span(SpanKind.AGENT, "agent") as span:
                span.end(usage=Usage(1, 1))
        assert _records(store) == []

    def test_enabled_from_env(self, store: TraceStore, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LUMECODE_TELEMETRY", "0")
        recorder = TelemetryRecorder("sess-1", store=store)
        assert recorder.enabled is False
        recorder.close()

    def test_queue_overflow_is_best_effort(self, store: TraceStore) -> None:
        with TelemetryRecorder("sess-1", store=store, max_queue=0) as recorder:
            with recorder.start_span(SpanKind.AGENT, "agent"):
                pass
            recorder.flush()
            assert recorder.dropped_spans == 1
        assert _records(store) == []

    def test_close_is_idempotent(self, store: TraceStore) -> None:
        recorder = TelemetryRecorder("sess-1", store=store)
        recorder.close()
        recorder.close()


# ---------------------------------------------------------------------------
# 记录器：内容、用量、成本
# ---------------------------------------------------------------------------


class TestContentAndUsage:
    def test_usage_attributes_and_cost(self, store: TraceStore) -> None:
        pricing = PricingTable(overrides={"test-model": ModelPrice(1.0, 2.0, 0.1, 0.5)})
        with TelemetryRecorder("sess-1", store=store, pricing=pricing) as recorder:
            with recorder.start_span(
                SpanKind.GENERATION, "call", model="test-model", input="hello"
            ) as handle:
                handle.end(
                    usage=Usage(1_000_000, 500_000, 100_000, 50_000),
                    output="world",
                )

        record = _records(store)[0]
        assert record["model"] == "test-model"
        assert record["usage"]["total_tokens"] == 1_650_000
        assert record["cost_usd"] == pytest.approx(2.035)
        assert record["attributes"]["gen_ai.request.model"] == "test-model"
        assert record["attributes"]["gen_ai.usage.input_tokens"] == 1_000_000
        assert record["attributes"]["cost.estimated"] is True
        assert record["input"] == "hello"
        assert record["output"] == "world"

    def test_unknown_model_has_no_cost(self, store: TraceStore) -> None:
        with TelemetryRecorder("sess-1", store=store) as recorder:
            with recorder.start_span(SpanKind.GENERATION, "call", model="unknown-model") as handle:
                handle.end(usage=Usage(10, 20))
        record = _records(store)[0]
        assert record["cost_usd"] is None
        assert record["attributes"]["cost.estimated"] is False

    def test_capture_content_disabled(self, store: TraceStore) -> None:
        with TelemetryRecorder("sess-1", store=store, capture_content=False) as recorder:
            with recorder.start_span(SpanKind.TOOL, "bash", input={"command": "ls"}) as handle:
                handle.end(output="file.txt")
        record = _records(store)[0]
        assert record["input"] is None
        assert record["output"] is None

    def test_content_truncation(self, store: TraceStore) -> None:
        with TelemetryRecorder("sess-1", store=store, max_content_chars=5) as recorder:
            with recorder.start_span(SpanKind.TOOL, "bash") as handle:
                handle.end(output="abcdefgh")
        record = _records(store)[0]
        assert record["output"] == "abcde\n...[truncated 8 chars]"

    def test_non_serializable_content_is_repr(self, store: TraceStore) -> None:
        class Weird:
            def __repr__(self) -> str:
                return "<weird>"

        with TelemetryRecorder("sess-1", store=store) as recorder:
            with recorder.start_span(SpanKind.TOOL, "bash", input=Weird()) as handle:
                handle.end(output={"nested": Weird()})
        record = _records(store)[0]
        assert record["input"] == "<weird>"
        assert record["output"] == {"nested": "<weird>"}

    def test_span_status_override(self, store: TraceStore) -> None:
        with TelemetryRecorder("sess-1", store=store) as recorder:
            with recorder.start_span(SpanKind.TOOL, "bash") as handle:
                handle.end(status=SpanStatus.CANCELLED)
        assert _records(store)[0]["status"] == "cancelled"


# ---------------------------------------------------------------------------
# 存储
# ---------------------------------------------------------------------------


class TestTraceStore:
    def test_sanitize_session_id(self) -> None:
        assert sanitize_session_id("abc-123_DEF.4") == "abc-123_DEF.4"
        assert sanitize_session_id("../../etc/passwd") == "etc_passwd"
        assert sanitize_session_id("") == "unknown"
        result = sanitize_session_id("a/b\\c d")
        assert "/" not in result
        assert "\\" not in result

    def test_path_stays_inside_base_dir(self, store: TraceStore) -> None:
        path = store.path_for("../../escape")
        assert path.parent == store.base_dir
        assert path.name == "escape.jsonl"

    def test_read_skips_corrupt_lines(self, store: TraceStore) -> None:
        path = store.path_for("sess-1")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"type": "span", "n": 1})
            + "\n"
            + "not json\n"
            + "\n"
            + json.dumps({"type": "span", "n": 2})
            + "\n",
            encoding="utf-8",
        )
        records = store.read("sess-1")
        assert [r["n"] for r in records] == [1, 2]

    def test_append_and_read_roundtrip(self, store: TraceStore) -> None:
        span = Span(
            trace_id=new_trace_id(),
            span_id=new_span_id(),
            parent_id=None,
            session_id="sess-1",
            kind=SpanKind.AGENT,
            name="agent",
        )
        span.end_time_ns = time.time_ns()
        assert store.append_span(span) is True
        records = store.read("sess-1")
        assert len(records) == 1
        assert records[0]["type"] == "span"
        assert records[0]["schema_version"] == 1
        assert records[0]["span_id"] == span.span_id

    def test_append_failure_returns_false(self, tmp_path: Path) -> None:
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        store = TraceStore(blocker)
        span = Span(
            trace_id=new_trace_id(),
            span_id=new_span_id(),
            parent_id=None,
            session_id="sess-1",
            kind=SpanKind.AGENT,
            name="agent",
        )
        span.end_time_ns = time.time_ns()
        assert store.append_span(span) is False

    def test_non_serializable_record_returns_false(self, store: TraceStore) -> None:
        assert store.append_record("sess-1", {"bad": object()}) is False

    def test_list_sessions_newest_first(self, store: TraceStore) -> None:
        for sid in ("old", "new"):
            path = store.path_for(sid)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")

        now = time.time()
        os.utime(store.path_for("old"), (now - 100, now - 100))
        os.utime(store.path_for("new"), (now, now))

        sessions = store.list_sessions()
        assert [s.session_id for s in sessions] == ["new", "old"]

    def test_cleanup_removes_old_files(self, store: TraceStore) -> None:
        for sid in ("old", "fresh"):
            path = store.path_for(sid)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")

        now = time.time()
        os.utime(store.path_for("old"), (now - 31 * 86_400, now - 31 * 86_400))
        os.utime(store.path_for("fresh"), (now, now))

        assert store.cleanup(max_age_days=30) == 1
        assert not store.path_for("old").exists()
        assert store.path_for("fresh").exists()
