from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from lumecode.context import auto_compact
from lumecode.conversation import ConversationManager, Message
from lumecode.memory.auto_memory import MemoryManager
from lumecode.memory.session import generate_session_summary
from lumecode.telemetry import SpanKind, TelemetryRecorder, TraceStore
from lumecode.tools.base import StreamEnd, TextDelta


class _FakeClient:
    """返回预设文本和 usage 的最小流式客户端。"""

    model = "test-model"

    def __init__(self, text: str, input_tokens: int = 10, output_tokens: int = 5) -> None:
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    async def stream(
        self,
        conversation: ConversationManager,
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[Any]:
        yield TextDelta(text=self.text)
        yield StreamEnd(
            "end_turn",
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cache_read=3,
            cache_creation=2,
        )


@pytest.fixture
def recorder(tmp_path: Path) -> TelemetryRecorder:
    rec = TelemetryRecorder("sess-1", store=TraceStore(tmp_path / "traces"))
    yield rec
    rec.close()


def _records(tmp_path: Path) -> list[dict[str, Any]]:
    return TraceStore(tmp_path / "traces").read("sess-1")


def _of_kind(records: list[dict[str, Any]], kind: SpanKind) -> list[dict[str, Any]]:
    return [r for r in records if r["kind"] == kind.value]


def _long_conversation() -> ConversationManager:
    conv = ConversationManager()
    for _ in range(8):
        conv.history.append(Message(role="user", content="old user " + "x" * 12_000))
        conv.history.append(Message(role="assistant", content="old assistant " + "y" * 12_000))
    for i in range(6):
        conv.history.append(Message(role="user", content=f"recent_{i}_" + "z" * 4_000))
    conv.record_usage_anchor(input_tokens=200_000)
    return conv


@pytest.mark.asyncio
async def test_auto_compact_records_compact_span(
    tmp_path: Path, recorder: TelemetryRecorder
) -> None:
    conv = _long_conversation()
    client = _FakeClient("<summary>compressed</summary>")

    result = await auto_compact(
        conv,
        client,
        context_window=200_000,
        session_dir=tmp_path,
        manual=True,
        telemetry=recorder,
    )

    recorder.flush()
    records = _records(tmp_path)
    assert records, f"expected a compact span, got {result!r}"
    compacts = _of_kind(records, SpanKind.COMPACT)
    assert len(compacts) == 1
    span = compacts[0]
    assert span["status"] == "ok"
    assert span["attributes"]["compact.manual"] is True
    assert span["attributes"]["compact.before_tokens"] > 0
    assert "compact.after_tokens" in span["attributes"]
    assert "compressed" in (span["output"] or "")


@pytest.mark.asyncio
async def test_auto_compact_below_threshold_does_not_record_span(
    tmp_path: Path, recorder: TelemetryRecorder
) -> None:
    conv = ConversationManager()
    conv.add_user_message("short conversation")

    result = await auto_compact(
        conv,
        _FakeClient("unused"),
        context_window=200_000,
        session_dir=tmp_path,
        telemetry=recorder,
    )

    recorder.flush()
    assert result is None
    assert _of_kind(_records(tmp_path), SpanKind.COMPACT) == []


@pytest.mark.asyncio
async def test_memory_extract_records_span(
    tmp_path: Path, recorder: TelemetryRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_dir = tmp_path / "user-memory"
    monkeypatch.setattr(
        "lumecode.memory.auto_memory.get_user_auto_mem_path",
        lambda: str(user_dir) + os.sep,
    )
    manager = MemoryManager(str(tmp_path / "project"))

    conv = ConversationManager()
    conv.add_user_message("Please always use tabs")
    conv.add_assistant_message("Noted.")

    client = _FakeClient(
        "MEMORY_NAME: tab-preference\n"
        "MEMORY_TYPE: feedback\n"
        "MEMORY_DESC: use tabs\n"
        "MEMORY_BODY: The user prefers tabs over spaces.\n"
        "---\n"
    )
    await manager.extract(client, conv, "anthropic", telemetry=recorder)

    recorder.flush()
    spans = _of_kind(_records(tmp_path), SpanKind.MEMORY_EXTRACT)
    assert len(spans) == 1
    span = spans[0]
    assert span["status"] == "ok"
    assert span["attributes"]["memory.files_written"] == 1
    assert span["usage"]["input_tokens"] == 10
    assert span["usage"]["output_tokens"] == 5
    assert (user_dir / "tab-preference.md").exists()


@pytest.mark.asyncio
async def test_session_summary_records_span(tmp_path: Path, recorder: TelemetryRecorder) -> None:
    conv = ConversationManager()
    conv.add_user_message("hello")
    conv.add_assistant_message("hi")

    summary = await generate_session_summary(
        _FakeClient("A short chat."), conv, "anthropic", telemetry=recorder
    )

    recorder.flush()
    spans = _of_kind(_records(tmp_path), SpanKind.SESSION_SUMMARY)
    assert summary == "A short chat."
    assert len(spans) == 1
    assert spans[0]["usage"]["total_tokens"] == 20
    assert spans[0]["output"] == "A short chat."
