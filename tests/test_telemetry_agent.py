from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from lumecode.agent import Agent, UsageEvent
from lumecode.client import LLMClient
from lumecode.conversation import ConversationManager
from lumecode.telemetry import (
    ModelPrice,
    PricingTable,
    SpanKind,
    TelemetryRecorder,
    TraceStore,
)
from lumecode.tools import create_default_registry
from lumecode.tools.base import (
    StreamEnd,
    StreamEvent,
    TextDelta,
    ToolCallComplete,
)


class ScriptedClient(LLMClient):
    """按脚本返回预设事件，用于验证 agent 埋点。"""

    model = "test-model"

    def __init__(self, responses: list[list[StreamEvent]]) -> None:
        self._responses = list(responses)
        self._index = 0

    async def stream(
        self,
        conversation: ConversationManager,
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        if self._index >= len(self._responses):
            yield TextDelta(text="done")
            yield StreamEnd("end_turn", input_tokens=1, output_tokens=1)
            return
        events = self._responses[self._index]
        self._index += 1
        for event in events:
            yield event


@pytest.fixture
def recorder(tmp_path: Path) -> Any:
    pricing = PricingTable(overrides={"test-model": ModelPrice(1.0, 2.0, 0.1, 0.5)})
    rec = TelemetryRecorder(
        "test-session",
        store=TraceStore(tmp_path / "traces"),
        pricing=pricing,
    )
    yield rec
    rec.close()


def _read(recorder: TelemetryRecorder, tmp_path: Path) -> list[dict[str, Any]]:
    return TraceStore(tmp_path / "traces").read("test-session")


def _span_by_kind(records: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [r for r in records if r["kind"] == kind]


@pytest.mark.asyncio
async def test_run_records_agent_generation_and_tool_spans(
    tmp_path: Path, recorder: TelemetryRecorder
) -> None:
    target = tmp_path / "hello.txt"
    target.write_text("hello", encoding="utf-8")

    client = ScriptedClient(
        [
            [
                TextDelta("Reading file."),
                ToolCallComplete("t1", "ReadFile", {"file_path": str(target)}),
                StreamEnd(
                    "end_turn",
                    input_tokens=10,
                    output_tokens=20,
                    cache_read=100,
                    cache_creation=50,
                ),
            ],
            [
                TextDelta("Done."),
                StreamEnd("end_turn", input_tokens=30, output_tokens=15),
            ],
        ]
    )
    agent = Agent(
        client,
        create_default_registry(),
        "anthropic",
        work_dir=str(tmp_path),
        telemetry=recorder,
    )
    conv = ConversationManager()
    conv.add_user_message("Read hello.txt")

    usage_events: list[UsageEvent] = []
    async for event in agent.run(conv):
        if isinstance(event, UsageEvent):
            usage_events.append(event)

    recorder.flush()
    records = _read(recorder, tmp_path)

    agents = _span_by_kind(records, SpanKind.AGENT.value)
    generations = _span_by_kind(records, SpanKind.GENERATION.value)
    tools = _span_by_kind(records, SpanKind.TOOL.value)

    assert len(agents) == 1
    assert agents[0]["parent_id"] is None
    assert len(generations) == 2
    assert len(tools) == 1

    # 所有子 span 都挂在同一次 run 的根 span 下，且共享同一个 trace。
    root_id = agents[0]["span_id"]
    assert {g["parent_id"] for g in generations} == {root_id}
    assert tools[0]["parent_id"] == root_id
    assert len({r["trace_id"] for r in records}) == 1

    # LLM 用量按“每次调用”记录，而不是累计值。
    assert generations[0]["usage"]["input_tokens"] == 10
    assert generations[0]["usage"]["output_tokens"] == 20
    assert generations[0]["usage"]["cache_read_tokens"] == 100
    assert generations[0]["usage"]["cache_creation_tokens"] == 50
    assert generations[1]["usage"]["input_tokens"] == 30
    assert generations[1]["usage"]["output_tokens"] == 15

    # 成本包含 cache read / creation：10*1 + 20*2 + 100*0.1 + 50*0.5，按百万 token 计。
    assert generations[0]["cost_usd"] == pytest.approx(0.000085)
    assert tools[0]["status"] == "ok"
    assert tools[0]["attributes"]["tool.is_error"] is False
    assert "hello" in tools[0]["output"]

    # UsageEvent 现在是累计值，并且带上了 cache token。
    assert usage_events[-1].input_tokens == 40
    assert usage_events[-1].output_tokens == 35
    assert usage_events[-1].cache_read_tokens == 100
    assert usage_events[-1].cache_creation_tokens == 50
    assert agent.total_cache_read_tokens == 100
    assert agent.total_cache_creation_tokens == 50


@pytest.mark.asyncio
async def test_tool_error_marks_tool_span(tmp_path: Path, recorder: TelemetryRecorder) -> None:
    client = ScriptedClient(
        [
            [
                ToolCallComplete("t1", "NoSuchTool", {}),
                StreamEnd("end_turn", input_tokens=1, output_tokens=1),
            ],
            [TextDelta("done"), StreamEnd("end_turn", input_tokens=1, output_tokens=1)],
        ]
    )
    agent = Agent(
        client,
        create_default_registry(),
        "anthropic",
        work_dir=str(tmp_path),
        telemetry=recorder,
    )
    conv = ConversationManager()
    conv.add_user_message("call unknown tool")

    async for _event in agent.run(conv):
        pass

    recorder.flush()
    tools = _span_by_kind(_read(recorder, tmp_path), SpanKind.TOOL.value)
    assert len(tools) == 1
    assert tools[0]["status"] == "error"
    assert tools[0]["attributes"]["tool.is_error"] is True
    assert "unknown tool" in tools[0]["error"]


@pytest.mark.asyncio
async def test_run_to_completion_records_spans(tmp_path: Path, recorder: TelemetryRecorder) -> None:
    target = tmp_path / "hello.txt"
    target.write_text("hello", encoding="utf-8")
    client = ScriptedClient(
        [
            [
                ToolCallComplete("t1", "ReadFile", {"file_path": str(target)}),
                StreamEnd("end_turn", input_tokens=5, output_tokens=5),
            ],
            [TextDelta("done"), StreamEnd("end_turn", input_tokens=5, output_tokens=5)],
        ]
    )
    agent = Agent(
        client,
        create_default_registry(),
        "anthropic",
        work_dir=str(tmp_path),
        telemetry=recorder,
    )

    result = await agent.run_to_completion("read hello")

    recorder.flush()
    records = _read(recorder, tmp_path)
    assert result == "done"
    assert len(_span_by_kind(records, SpanKind.AGENT.value)) == 1
    assert len(_span_by_kind(records, SpanKind.GENERATION.value)) == 2
    assert len(_span_by_kind(records, SpanKind.TOOL.value)) == 1


@pytest.mark.asyncio
async def test_run_marks_agent_span_error(tmp_path: Path, recorder: TelemetryRecorder) -> None:
    class BoomClient(ScriptedClient):
        async def stream(
            self,
            conversation: ConversationManager,
            system: str = "",
            tools: list[dict[str, Any]] | None = None,
        ) -> AsyncIterator[StreamEvent]:
            raise RuntimeError("provider down")
            yield TextDelta(text="unreachable")

    agent = Agent(
        BoomClient([]),
        create_default_registry(),
        "anthropic",
        work_dir=str(tmp_path),
        telemetry=recorder,
    )
    conv = ConversationManager()
    conv.add_user_message("hi")

    with pytest.raises(RuntimeError, match="provider down"):
        async for _event in agent.run(conv):
            pass

    recorder.flush()
    agents = _span_by_kind(_read(recorder, tmp_path), SpanKind.AGENT.value)
    assert len(agents) == 1
    assert agents[0]["status"] == "error"
    assert agents[0]["error"] == "provider down"
