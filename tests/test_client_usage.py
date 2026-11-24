"""OpenAI 兼容客户端的 usage 解析回归测试。

背景：DeepSeek 把 usage 挂在最后一个 **内容 chunk** 上（choices 非空），
而 OpenAI 则发一个 choices 为空的独立 usage chunk。曾经的实现只处理后者，
导致 DeepSeek 下所有 token 统计都是 0，`/trace cost` 因此全为 $0。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

from lumecode.client import OpenAICompatClient, _map_finish_reason, _openai_usage_counts
from lumecode.config import ProviderConfig
from lumecode.conversation import ConversationManager
from lumecode.tools.base import StreamEnd, TextDelta, ToolCallComplete

# ---------------------------------------------------------------------------
# 构造假 chunk
# ---------------------------------------------------------------------------


def _chunk(
    *,
    content: str | None = None,
    finish_reason: str | None = None,
    tool_calls: list[Any] | None = None,
    reasoning: str | None = None,
    usage: Any = None,
) -> Any:
    delta = SimpleNamespace(
        content=content,
        reasoning_content=reasoning,
        tool_calls=tool_calls,
    )
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage)


def _empty_usage_chunk(usage: Any) -> Any:
    """OpenAI 风格：choices 为空的独立 usage chunk。"""
    return SimpleNamespace(choices=[], usage=usage)


def _tool_call_delta(*, index: int, call_id: str, name: str, args: str) -> Any:
    return SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=args),
    )


class FakeStream:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks

    def __aiter__(self) -> AsyncIterator[Any]:
        async def gen() -> AsyncIterator[Any]:
            for chunk in self._chunks:
                yield chunk

        return gen()


class FakeCompletions:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> FakeStream:
        self.calls.append(kwargs)
        return FakeStream(self._chunks)


class FakeChat:
    def __init__(self, chunks: list[Any]) -> None:
        self.completions = FakeCompletions(chunks)


class FakeAsyncOpenAI:
    def __init__(self, chunks: list[Any]) -> None:
        self.chat = FakeChat(chunks)


@pytest.fixture
def make_client(monkeypatch: pytest.MonkeyPatch):
    """返回一个把底层 AsyncOpenAI 替换成假流的 client 工厂。"""

    def factory(chunks: list[Any], base_url: str = "https://api.deepseek.com/v1"):
        config = ProviderConfig(
            name="fake",
            protocol="openai-compat",
            base_url=base_url,
            model="deepseek-flash",
            api_key="test-key",
        )
        client = OpenAICompatClient(config)
        fake = FakeAsyncOpenAI(chunks)
        monkeypatch.setattr(client, "_client", fake)
        return client, fake

    return factory


async def _collect(client: OpenAICompatClient) -> list[Any]:
    events: list[Any] = []
    async for event in client.stream(ConversationManager()):
        events.append(event)
    return events


# ---------------------------------------------------------------------------
# 纯函数
# ---------------------------------------------------------------------------


class TestUsageParsing:
    def test_openai_cached_tokens_details(self) -> None:
        usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            prompt_tokens_details=SimpleNamespace(cached_tokens=40),
        )
        assert _openai_usage_counts(usage) == (60, 40, 20)

    def test_deepseek_cache_hit_field(self) -> None:
        usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
            prompt_cache_hit_tokens=40,
        )
        assert _openai_usage_counts(usage) == (60, 40, 20)

    def test_cache_field_never_exceeds_prompt(self) -> None:
        usage = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=1,
            prompt_cache_hit_tokens=999,
        )
        assert _openai_usage_counts(usage) == (0, 10, 1)

    def test_missing_fields_default_to_zero(self) -> None:
        assert _openai_usage_counts(SimpleNamespace()) == (0, 0, 0)

    def test_finish_reason_mapping(self) -> None:
        assert _map_finish_reason("length") == "max_tokens"
        assert _map_finish_reason("stop") == "end_turn"
        assert _map_finish_reason("tool_calls") == "end_turn"
        assert _map_finish_reason("content_filter") == "content_filter"
        assert _map_finish_reason(None) == "end_turn"


# ---------------------------------------------------------------------------
# 流解析
# ---------------------------------------------------------------------------


class TestStreamUsage:
    @pytest.mark.asyncio
    async def test_usage_on_last_content_chunk(self, make_client) -> None:
        """DeepSeek 风格：usage 挂在最后一个内容 chunk 上。"""
        usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            prompt_cache_hit_tokens=40,
        )
        chunks = [
            _chunk(content="Hel", usage=None),
            _chunk(content="lo", finish_reason="stop", usage=usage),
        ]
        client, _ = make_client(chunks)

        events = await _collect(client)

        text = "".join(e.text for e in events if isinstance(e, TextDelta))
        assert text == "Hello"
        ends = [e for e in events if isinstance(e, StreamEnd)]
        assert len(ends) == 1
        assert ends[0].input_tokens == 60
        assert ends[0].cache_read == 40
        assert ends[0].output_tokens == 20
        assert ends[0].stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_usage_in_separate_empty_choices_chunk(self, make_client) -> None:
        """OpenAI 风格：独立的 choices=[] usage chunk。"""
        usage = SimpleNamespace(
            prompt_tokens=50,
            completion_tokens=10,
            prompt_tokens_details=SimpleNamespace(cached_tokens=10),
        )
        chunks = [
            _chunk(content="Hi"),
            _empty_usage_chunk(usage),
        ]
        client, _ = make_client(chunks, base_url="https://api.openai.com/v1")

        events = await _collect(client)

        ends = [e for e in events if isinstance(e, StreamEnd)]
        assert len(ends) == 1
        assert ends[0].input_tokens == 40
        assert ends[0].cache_read == 10
        assert ends[0].output_tokens == 10

    @pytest.mark.asyncio
    async def test_no_usage_falls_back_to_zero(self, make_client) -> None:
        """provider 完全不返回 usage 时仍要发 StreamEnd，不能挂起。"""
        chunks = [_chunk(content="hi", finish_reason="stop")]
        client, fake = make_client(chunks, base_url="https://api.ollama.local/v1")

        events = await _collect(client)

        ends = [e for e in events if isinstance(e, StreamEnd)]
        assert len(ends) == 1
        assert ends[0].input_tokens == 0
        assert ends[0].output_tokens == 0
        # ollama 在排除列表里，不能发送 stream_options（否则 400）。
        assert "stream_options" not in fake.chat.completions.calls[0]

    @pytest.mark.asyncio
    async def test_deepseek_still_parses_usage_without_stream_options(self, make_client) -> None:
        """DeepSeek 不接收 stream_options，但末内容 chunk 仍带 usage，必须解析。"""
        usage = SimpleNamespace(prompt_tokens=7, completion_tokens=3)
        client, fake = make_client([_chunk(content="x", finish_reason="stop", usage=usage)])

        events = await _collect(client)

        assert "stream_options" not in fake.chat.completions.calls[0]
        end = next(e for e in events if isinstance(e, StreamEnd))
        assert end.input_tokens == 7
        assert end.output_tokens == 3

    @pytest.mark.asyncio
    async def test_openai_compatible_provider_gets_stream_options(self, make_client) -> None:
        client, fake = make_client(
            [_chunk(content="x", finish_reason="stop")],
            base_url="https://api.openai.com/v1",
        )
        await _collect(client)
        assert fake.chat.completions.calls[0]["stream_options"] == {"include_usage": True}

    @pytest.mark.asyncio
    async def test_length_maps_to_max_tokens(self, make_client) -> None:
        """截断必须映射成 max_tokens，否则 Agent 不会续写。"""
        usage = SimpleNamespace(prompt_tokens=1, completion_tokens=2)
        client, _ = make_client([_chunk(content="partial", finish_reason="length", usage=usage)])

        events = await _collect(client)

        end = next(e for e in events if isinstance(e, StreamEnd))
        assert end.stop_reason == "max_tokens"

    @pytest.mark.asyncio
    async def test_tool_calls_still_emitted_with_usage(self, make_client) -> None:
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        args = json.dumps({"path": "a.txt"})
        chunks = [
            _chunk(
                tool_calls=[_tool_call_delta(index=0, call_id="call_1", name="ReadFile", args=args)]
            ),
            _chunk(finish_reason="tool_calls", usage=usage),
        ]
        client, _ = make_client(chunks)

        events = await _collect(client)

        tool_calls = [e for e in events if isinstance(e, ToolCallComplete)]
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == "ReadFile"
        assert tool_calls[0].arguments == {"path": "a.txt"}
        end = next(e for e in events if isinstance(e, StreamEnd))
        assert end.input_tokens == 10
        assert end.output_tokens == 5
