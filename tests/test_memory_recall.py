"""记忆召回升级的单元测试：索引 hook、去重、注入预算、向量选择。

这些测试不访问网络——embedding 客户端用假实现替换。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from typing import Any

from lumecode.config import EmbeddingConfig
from lumecode.memory.embedding import (
    EmbeddingClient,
    VectorMemoryIndex,
    build_embedding_text,
    cosine_similarity,
    select_top_k,
)
from lumecode.memory.recall import (
    MAX_REMINDER_CHARS,
    MemoryHeader,
    RelevantMemory,
    build_recall_result,
    collect_memory_headers,
    find_relevant_memories,
    format_memory_manifest,
    render_reminder,
    scan_memory_files,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _memory_file(path: Path, name: str, description: str = "") -> None:
    fm = (
        f"---\nname: {name}\ndescription: {description}\ntype: project\n---\n"
        if description
        else "---\ntype: project\n---\n"
    )
    _write(path, fm + f"\n# {name}\n\nBody for {name}.\n")


# ---------------------------------------------------------------------------
# A 档：MEMORY.md 索引参与召回
# ---------------------------------------------------------------------------


class TestEntrypointHooks:
    def test_index_hook_is_read(self, tmp_path: Path) -> None:
        mem = tmp_path / "mem"
        _memory_file(mem / "project.md", "project")
        _write(
            mem / "MEMORY.md",
            "- [LumeCode project](project.md) — Python rewrite aligned with Codex\n",
        )

        headers = scan_memory_files(mem, "project")

        assert len(headers) == 1
        assert "Python rewrite aligned with Codex" in headers[0].hook

    def test_frontmatter_description_wins_over_index(self, tmp_path: Path) -> None:
        mem = tmp_path / "mem"
        _memory_file(mem / "a.md", "a", description="frontmatter description")
        _write(mem / "MEMORY.md", "- [Title](a.md) — index hook\n")

        headers = scan_memory_files(mem, "project")

        assert headers[0].hook == "frontmatter description"

    def test_memory_without_description_or_hook_falls_back_to_stem(self, tmp_path: Path) -> None:
        mem = tmp_path / "mem"
        _memory_file(mem / "lonely.md", "lonely")

        headers = scan_memory_files(mem, "project")

        assert headers[0].hook == "lonely"

    def test_manifest_includes_index_hook(self) -> None:
        header = MemoryHeader(
            filename="a.md",
            file_path="/mem/a.md",
            scope="user",
            mtime_ms=0,
            description="",
            type="feedback",
            hook="prefers terse answers — no trailing summaries",
        )

        manifest = format_memory_manifest([header])

        assert "prefers terse answers" in manifest
        # 没有描述时不再出现"只有路径"的死条目
        assert ": prefers" in manifest

    def test_nested_index_path_matches(self, tmp_path: Path) -> None:
        mem = tmp_path / "mem"
        _memory_file(mem / "sub" / "nested.md", "nested")
        _write(mem / "MEMORY.md", "- [Nested](sub/nested.md) — nested hook\n")

        headers = scan_memory_files(mem, "project")

        assert "nested hook" in headers[0].hook


# ---------------------------------------------------------------------------
# A 档：跨轮去重
# ---------------------------------------------------------------------------


class TestSurfacedDedup:
    def test_collect_skips_surfaced(self, tmp_path: Path) -> None:
        mem = tmp_path / "mem"
        _memory_file(mem / "a.md", "a")
        _memory_file(mem / "b.md", "b")
        surfaced = {str((mem / "a.md").resolve())}

        candidates = collect_memory_headers(mem, None, surfaced)

        assert [Path(m.file_path).name for m in candidates] == ["b.md"]

    def test_find_relevant_memories_honours_surfaced(self, tmp_path: Path) -> None:
        mem = tmp_path / "mem"
        _memory_file(mem / "a.md", "a")
        surfaced = {str((mem / "a.md").resolve())}

        async def selector(_system: str, _user: str) -> str:
            return json.dumps({"selected_memories": ["a.md"]})

        results = asyncio.run(
            find_relevant_memories(
                query="q",
                user_mem_dir=None,
                project_mem_dir=mem,
                recent_tools=None,
                already_surfaced=surfaced,
                selector=selector,
            )
        )

        assert results == []

    def test_recall_result_reports_paths(self, tmp_path: Path) -> None:
        mem = tmp_path / "m.md"
        _write(mem, "# M\n\ncontent\n")
        result = build_recall_result([RelevantMemory(path=str(mem), mtime_ms=0)])
        assert result.paths == [str(mem)]
        assert "content" in result.reminder


# ---------------------------------------------------------------------------
# A 档：注入预算
# ---------------------------------------------------------------------------


class TestReminderBudget:
    def test_oversized_file_is_skipped_and_counted(self, tmp_path: Path) -> None:
        big = tmp_path / "big.md"
        _write(big, "x" * (MAX_REMINDER_CHARS + 100))
        small = tmp_path / "small.md"
        _write(small, "small content")

        reminder = render_reminder(
            [
                RelevantMemory(path=str(small), mtime_ms=0),
                RelevantMemory(path=str(big), mtime_ms=0),
            ]
        )

        assert "small content" in reminder
        assert "x" * 100 not in reminder
        assert "omitted" in reminder

    def test_first_oversized_still_included(self, tmp_path: Path) -> None:
        # 第一个文件即使超预算也保留，保证召回永不为空。
        big = tmp_path / "big.md"
        _write(big, "y" * (MAX_REMINDER_CHARS + 100))

        reminder = render_reminder([RelevantMemory(path=str(big), mtime_ms=0)])

        assert "y" * 100 in reminder


# ---------------------------------------------------------------------------
# B 档：向量召回（不访问网络）
# ---------------------------------------------------------------------------


class FakeEmbeddingClient:
    """按文本哈希生成确定性向量的假客户端。"""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.config = EmbeddingConfig(base_url="http://fake", model="fake", dimensions=2)

    async def embed(self, texts: list[str], text_type: str | None = None) -> list[list[float]]:
        self.calls.append(texts)
        return [self._vec(t) for t in texts]

    @staticmethod
    def _vec(text: str) -> list[float]:
        if "food" in text or "tea" in text:
            return [1.0, 0.0]
        if "recall" in text or "memory" in text:
            return [0.0, 1.0]
        return [0.5, 0.5]


class TestCosineAndSelection:
    def test_cosine_basics(self) -> None:
        assert cosine_similarity([1, 0], [1, 0]) == 1.0
        assert cosine_similarity([0, 1], [1, 0]) == 0.0
        assert cosine_similarity([1, 0], [1, 0, 0]) == 0.0

    def test_select_filters_below_min_and_relative(self) -> None:
        entries = [
            {"path": "/a.md", "mtime_ms": 0, "vector": [1.0, 0.0]},
            {"path": "/b.md", "mtime_ms": 0, "vector": [0.9, 0.1]},
            {"path": "/c.md", "mtime_ms": 0, "vector": [0.0, 1.0]},
        ]

        results = select_top_k([1.0, 0.0], entries, top_k=5, min_score=0.45)

        names = [Path(r.path).name for r in results]
        assert names == ["a.md", "b.md"]

    def test_select_returns_empty_for_unrelated_query(self) -> None:
        entries = [{"path": "/a.md", "mtime_ms": 0, "vector": [1.0, 0.0]}]
        assert select_top_k([0.0, 1.0], entries, top_k=5, min_score=0.45) == []


class TestVectorIndex:
    def test_search_embeds_candidates_and_query(self, tmp_path: Path) -> None:
        mem = tmp_path / "mem"
        _write(mem / "food.md", "User likes tea and spicy food.\n")
        _write(mem / "recall.md", "Memory recall runs in app.py.\n")
        client = FakeEmbeddingClient()
        index = VectorMemoryIndex(tmp_path / ".index.json", client)  # type: ignore[arg-type]

        results = asyncio.run(index.search("what food?", mem, None, None, top_k=5))

        assert [Path(r.path).name for r in results] == ["food.md"]

    def test_index_cache_avoids_reembedding(self, tmp_path: Path) -> None:
        mem = tmp_path / "mem"
        _write(mem / "food.md", "User likes tea.\n")
        client = FakeEmbeddingClient()
        index_path = tmp_path / ".index.json"
        index = VectorMemoryIndex(index_path, client)  # type: ignore[arg-type]

        asyncio.run(index.search("food", mem, None, None, top_k=5))
        assert index_path.exists()

        # 新实例从磁盘加载索引，文件未变化时只 embed 查询。
        index2 = VectorMemoryIndex(index_path, client)  # type: ignore[arg-type]
        client.calls.clear()
        asyncio.run(index2.search("food again", mem, None, None, top_k=5))

        assert len(client.calls) == 1

    def test_failed_embedding_falls_back_to_empty(self, tmp_path: Path) -> None:
        mem = tmp_path / "mem"
        _write(mem / "food.md", "User likes tea.\n")

        class Boom:
            config = EmbeddingConfig(base_url="http://fake", model="fake")

            async def embed(self, texts: list[str], text_type: str | None = None) -> Any:
                raise RuntimeError("network down")

        index = VectorMemoryIndex(tmp_path / ".index.json", Boom())  # type: ignore[arg-type]

        # search 把异常交给调用方处理（app 会回退到 LLM selector）。
        with contextlib.suppress(RuntimeError):
            asyncio.run(index.search("food", mem, None, None, top_k=5))
        assert not (tmp_path / ".index.json").exists()

    def test_build_embedding_text_includes_hook_and_body(self, tmp_path: Path) -> None:
        path = tmp_path / "a.md"
        _write(path, "body content here")
        header = MemoryHeader(
            filename="a.md",
            file_path=str(path),
            scope="project",
            mtime_ms=0,
            description="",
            type="project",
            hook="index hook",
        )

        text = build_embedding_text(header, path)

        assert "index hook" in text
        assert "body content here" in text

    def test_disabled_client_is_unavailable(self) -> None:
        index = VectorMemoryIndex(Path("x"), EmbeddingClient(EmbeddingConfig()))
        assert not index.available


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


class TestMemoryConfig:
    def _base(self) -> dict:
        return {
            "providers": [
                {
                    "name": "test",
                    "protocol": "openai-compat",
                    "base_url": "http://localhost/v1",
                    "model": "m",
                }
            ],
        }

    def test_memory_section_is_optional(self) -> None:
        from lumecode.validator import validate_config_structure

        validated = validate_config_structure(self._base())

        assert validated["memory"] == {"embedding": {}, "extraction_interval": 4}

    def test_extraction_interval_parsed_and_validated(self) -> None:
        import pytest

        from lumecode.validator import ConfigError, validate_config_structure

        raw = self._base()
        raw["memory"] = {"extraction_interval": 0}
        assert validate_config_structure(raw)["memory"]["extraction_interval"] == 0

        raw["memory"] = {"extraction_interval": -1}
        with pytest.raises(ConfigError, match="non-negative integer"):
            validate_config_structure(raw)

    def test_embedding_parsed(self) -> None:
        from lumecode.validator import validate_config_structure

        raw = self._base()
        raw["memory"] = {
            "embedding": {
                "base_url": "https://example.com/v1",
                "model": "text-embedding-v3",
                "api_key": "k",
                "dimensions": 1024,
                "top_k": 3,
                "min_score": 0.5,
                "uses_text_type": True,
            }
        }

        validated = validate_config_structure(raw)
        emb = validated["memory"]["embedding"]

        assert emb["model"] == "text-embedding-v3"
        assert emb["top_k"] == 3
        assert emb["min_score"] == 0.5
        assert emb["uses_text_type"] is True

    def test_embedding_missing_model_rejected(self) -> None:
        import pytest

        from lumecode.validator import ConfigError, validate_config_structure

        raw = self._base()
        raw["memory"] = {"embedding": {"base_url": "https://example.com/v1"}}

        with pytest.raises(ConfigError, match="missing fields"):
            validate_config_structure(raw)

    def test_invalid_min_score_rejected(self) -> None:
        import pytest

        from lumecode.validator import ConfigError, validate_config_structure

        raw = self._base()
        raw["memory"] = {
            "embedding": {
                "base_url": "https://example.com/v1",
                "model": "m",
                "min_score": 1.5,
            }
        }

        with pytest.raises(ConfigError, match="between 0 and 1"):
            validate_config_structure(raw)

    def test_embedding_config_env_var_key(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("MY_EMB_KEY", "secret")
        cfg = EmbeddingConfig(base_url="http://x", model="m", api_key="${MY_EMB_KEY}")
        assert cfg.resolve_api_key() == "secret"


# ---------------------------------------------------------------------------
# B 档：Agent 侧注入时序（首轮有界等待 + 跨轮去重）
# ---------------------------------------------------------------------------


class _RecordingClient:
    """记录每次模型调用时对话里已有多少条消息的假客户端。"""

    def __init__(self) -> None:
        self.history_snapshots: list[list[str]] = []

    async def stream(self, conversation: Any, system: str = "", tools: Any = None):
        from lumecode.tools.base import StreamEnd, TextDelta

        self.history_snapshots.append([m.content for m in conversation.history])
        yield TextDelta("ok")
        yield StreamEnd(stop_reason="end_turn", input_tokens=1, output_tokens=1)


class TestAgentRecallInjection:
    def _agent(self) -> Any:
        from lumecode.agent import Agent
        from lumecode.tools import create_default_registry

        return Agent(_RecordingClient(), create_default_registry(), "anthropic", work_dir=".")

    def test_first_turn_waits_for_ready_recall(self) -> None:
        from lumecode.conversation import ConversationManager
        from lumecode.memory.recall import RecallResult

        async def main() -> tuple[Any, list[list[str]]]:
            agent = self._agent()
            agent.recall_first_wait = 1.0

            async def slow_recall() -> RecallResult:
                await asyncio.sleep(0.02)
                return RecallResult(reminder="REMEMBER 42", paths=["/m/42.md"])

            agent.memory_recall_task = asyncio.create_task(slow_recall())
            conv = ConversationManager()
            async for _ in agent.run(conv):
                pass
            return agent, agent.client.history_snapshots

        agent, snapshots = asyncio.run(main())

        # 首轮模型调用前，召回结果已经写入对话历史。
        assert any("REMEMBER 42" in msg for msg in snapshots[0])
        assert agent._surfaced_memory_paths == {"/m/42.md"}

    def test_ready_recall_injected_without_wait(self) -> None:
        from lumecode.conversation import ConversationManager
        from lumecode.memory.recall import RecallResult

        async def main() -> list[list[str]]:
            agent = self._agent()
            agent.recall_first_wait = 0.0

            async def instant() -> RecallResult:
                return RecallResult(reminder="INSTANT", paths=["/m/i.md"])

            task = asyncio.create_task(instant())
            await asyncio.sleep(0)  # 让 task 完成
            agent.memory_recall_task = task
            conv = ConversationManager()
            async for _ in agent.run(conv):
                pass
            return agent.client.history_snapshots

        snapshots = asyncio.run(main())

        assert any("INSTANT" in msg for msg in snapshots[0])

    def test_run_resets_surfaced_paths(self) -> None:
        from lumecode.conversation import ConversationManager

        async def main() -> Any:
            agent = self._agent()
            agent._surfaced_memory_paths = {"/stale.md"}
            agent._memory_recall_consumed = True
            conv = ConversationManager()
            async for _ in agent.run(conv):
                pass
            return agent

        agent = asyncio.run(main())

        assert agent._surfaced_memory_paths == set()
        assert agent._memory_recall_consumed is False
