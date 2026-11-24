from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

from lumecode.commands.handlers.trace import create_trace_command
from lumecode.commands.registry import CommandContext
from lumecode.telemetry import TelemetryRecorder, TraceStore
from lumecode.telemetry.models import new_span_id


class MockUI:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def add_system_message(self, text: str) -> None:
        self.messages.append(text)

    def send_user_message(self, text: str) -> None: ...

    def set_plan_mode(self, enabled: bool) -> None: ...

    def get_token_count(self) -> tuple[int, int]:
        return 0, 0

    def refresh_status(self) -> None: ...


class FakeTraceManager:
    _nodes: ClassVar[dict] = {}


def _context(ui: MockUI, args: str = "", agent: object | None = None) -> CommandContext:
    return CommandContext(
        args=args,
        agent=agent,
        conversation=None,
        session=None,
        session_manager=None,
        memory_manager=None,
        ui=ui,
        config={},
    )


def _record(
    kind: str,
    name: str,
    *,
    parent_id: str | None = None,
    usage: dict | None = None,
    cost_usd: float | None = None,
    status: str = "ok",
    duration_ms: float = 10,
    trace_id: str | None = None,
    model: str | None = None,
) -> dict:
    return {
        "type": "span",
        "schema_version": 1,
        "trace_id": trace_id or "trace-1",
        "span_id": new_span_id(),
        "parent_id": parent_id,
        "session_id": "sess-1",
        "kind": kind,
        "name": name,
        "model": model,
        "duration_ms": duration_ms,
        "status": status,
        "usage": usage,
        "cost_usd": cost_usd,
        "attributes": {},
    }


@pytest.fixture
def store(tmp_path: Path) -> TraceStore:
    return TraceStore(tmp_path / "traces")


@pytest.fixture
def recorder(store: TraceStore) -> TelemetryRecorder:
    rec = TelemetryRecorder("sess-1", store=store)
    yield rec
    rec.close()


@pytest.mark.asyncio
async def test_trace_renders_current_session_tree(
    store: TraceStore, recorder: TelemetryRecorder
) -> None:
    agent_record = _record("agent", "agent:abc", duration_ms=120)
    generation = _record(
        "generation",
        "test-model",
        parent_id=agent_record["span_id"],
        usage={
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        },
        cost_usd=0.001,
        duration_ms=80,
    )
    tool = _record(
        "tool",
        "ReadFile",
        parent_id=agent_record["span_id"],
        duration_ms=5,
    )
    store.append_record("sess-1", agent_record)
    store.append_record("sess-1", generation)
    store.append_record("sess-1", tool)

    ui = MockUI()
    command = create_trace_command(
        FakeTraceManager(),
        telemetry_getter=lambda: recorder,
    )
    await command.handler(_context(ui, agent=object()))

    output = ui.messages[-1]
    assert "Trace 树" in output
    assert "agent:abc" in output
    assert "test-model" in output
    assert "ReadFile" in output
    assert "↑100 ↓50" in output
    assert "合计: 3 spans" in output


@pytest.mark.asyncio
async def test_trace_cost_command(store: TraceStore, recorder: TelemetryRecorder) -> None:
    store.append_record("sess-1", _record("agent", "agent"))
    store.append_record(
        "sess-1",
        _record(
            "generation",
            "test-model",
            usage={"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500},
            cost_usd=0.002,
            model="test-model",
        ),
    )

    ui = MockUI()
    command = create_trace_command(FakeTraceManager(), telemetry_getter=lambda: recorder)
    await command.handler(_context(ui, "cost", agent=object()))

    output = ui.messages[-1]
    assert "Trace 成本报告" in output
    assert "总 token" in output
    assert "generation" in output
    assert "test-model" in output


@pytest.mark.asyncio
async def test_zero_usage_does_not_warn_about_unknown_pricing(
    store: TraceStore, recorder: TelemetryRecorder
) -> None:
    """provider 没返回 usage 时，不应误报“无法匹配价格表”。"""
    store.append_record(
        "sess-1",
        _record(
            "memory_recall",
            "memory-recall",
            usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            cost_usd=None,
        ),
    )

    ui = MockUI()
    command = create_trace_command(FakeTraceManager(), telemetry_getter=lambda: recorder)
    await command.handler(_context(ui, "cost", agent=object()))

    assert "无法匹配价格表" not in ui.messages[-1]


@pytest.mark.asyncio
async def test_trace_list_and_cleanup(store: TraceStore, recorder: TelemetryRecorder) -> None:
    store.append_record("sess-1", _record("agent", "agent"))
    store.append_record("other-session", _record("agent", "agent"))

    ui = MockUI()
    command = create_trace_command(FakeTraceManager(), telemetry_getter=lambda: recorder)
    await command.handler(_context(ui, "list", agent=object()))
    assert "sess-1" in ui.messages[-1]
    assert "other-session" in ui.messages[-1]

    await command.handler(_context(ui, "cleanup", agent=object()))
    assert "已清理" in ui.messages[-1]


@pytest.mark.asyncio
async def test_trace_falls_back_to_legacy_view(
    store: TraceStore, recorder: TelemetryRecorder
) -> None:
    ui = MockUI()
    command = create_trace_command(FakeTraceManager(), telemetry_getter=lambda: recorder)
    await command.handler(_context(ui, agent=object()))
    assert "没有 Agent 追踪记录" in ui.messages[-1]


@pytest.mark.asyncio
async def test_trace_specific_session(store: TraceStore, recorder: TelemetryRecorder) -> None:
    store.append_record("other-session", _record("agent", "agent:other"))
    ui = MockUI()
    command = create_trace_command(FakeTraceManager(), telemetry_getter=lambda: recorder)
    await command.handler(_context(ui, "other-session", agent=object()))
    assert "agent:other" in ui.messages[-1]


@pytest.mark.asyncio
async def test_trace_unknown_session(store: TraceStore, recorder: TelemetryRecorder) -> None:
    ui = MockUI()
    command = create_trace_command(FakeTraceManager(), telemetry_getter=lambda: recorder)
    await command.handler(_context(ui, "missing", agent=object()))
    assert "没有找到会话 missing" in ui.messages[-1]


@pytest.mark.asyncio
async def test_trace_without_telemetry_uses_default_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    ui = MockUI()
    command = create_trace_command(FakeTraceManager(), telemetry_getter=lambda: None)
    await command.handler(_context(ui, "cost", agent=object()))
    assert "没有找到会话" in ui.messages[-1] or "没有 Agent 追踪记录" in ui.messages[-1]


@pytest.mark.asyncio
async def test_trace_cleanup_when_nothing_to_remove(
    store: TraceStore, recorder: TelemetryRecorder
) -> None:
    store.append_record("sess-1", _record("agent", "agent"))
    ui = MockUI()
    command = create_trace_command(FakeTraceManager(), telemetry_getter=lambda: recorder)
    await command.handler(_context(ui, "cleanup", agent=object()))
    assert "已清理 0 个" in ui.messages[-1]
