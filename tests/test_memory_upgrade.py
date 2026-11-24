"""记忆升级的单元测试：混合召回、ADD-only 抽取、observed_at 与历史快照。

全部不访问网络——LLM/embedding 都用假实现替换。
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from lumecode.config import EmbeddingConfig
from lumecode.conversation import ConversationManager, Message
from lumecode.memory.auto_memory import MemoryManager, _append_memory
from lumecode.memory.embedding import VectorMemoryIndex
from lumecode.memory.history import (
    HISTORY_DIR_NAME,
    iter_memory_files,
    snapshot_file,
    snapshot_tree,
)
from lumecode.memory.recall import (
    RelevantMemory,
    memory_age,
    parse_observed_ms,
    render_reminder,
    scan_memory_files,
)
from lumecode.memory.textmatch import (
    bm25_scores,
    build_document,
    extract_entities,
    tokenize,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _memory(path: Path, body: str, *, name: str = "m", extra: str = "") -> None:
    _write(
        path,
        f"---\nname: {name}\ndescription: test memory\n{extra}metadata:\n  type: project\n---\n\n{body}\n",
    )


# ---------------------------------------------------------------------------
# ① 混合召回：分词 / BM25 / 实体
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_splits_paths_and_identifiers(self) -> None:
        tokens = tokenize("lumecode/memory/recall.py AsyncTaskManager snake_case_var")
        assert {"lumecode", "memory", "recall", "py"} <= set(tokens)
        assert {"async", "task", "manager"} <= set(tokens)
        assert {"snake", "case", "var"} <= set(tokens)

    def test_keeps_chinese_bigrams(self) -> None:
        tokens = tokenize("权限沙箱")
        assert {"权限", "限沙", "沙箱"} <= set(tokens)

    def test_drops_stopwords(self) -> None:
        assert "the" not in tokenize("the file")


class TestEntities:
    def test_extracts_paths_quotes_and_symbols(self) -> None:
        entities = extract_entities(
            "改 lumecode/memory/recall.py，跑 `ruff format`，再看 AsyncTaskManager"
        )
        assert "lumecode/memory/recall.py" in entities
        assert "recall.py" in entities
        assert "ruff format" in entities
        assert "asynctaskmanager" in entities

    def test_backslashes_normalized(self) -> None:
        assert "lumecode/memory/recall.py" in extract_entities(r"lumecode\memory\recall.py")


class TestBM25:
    def test_exact_token_match_scores_higher(self) -> None:
        docs = [
            build_document("memory recall lives in app.py, run ruff format"),
            build_document("user likes tea and spicy food"),
            build_document("the file code module"),
        ]
        scores = bm25_scores("app.py ruff format", docs)
        assert scores[0] > 0.5
        assert scores[1] == 0.0

    def test_unrelated_query_scores_zero(self) -> None:
        docs = [build_document("memory recall lives in app.py")]
        assert bm25_scores("how do I cook dinner", docs) == [0.0]

    def test_legacy_entries_without_tokens(self) -> None:
        assert bm25_scores("app.py", [{"vector": [1.0]}]) == [0.0]


class TestHybridSelection:
    def test_lexical_match_lifts_low_cosine_entry(self) -> None:
        entries = [
            {
                "path": "/generic.md",
                "mtime_ms": 0,
                "vector": [0.9, 0.44],  # 余弦 0.9：语义上沾边但没有字面命中
                **build_document("code file module"),
            },
            {
                "path": "/exact.md",
                "mtime_ms": 0,
                "vector": [0.62, 0.78],  # 余弦 0.62，字面/实体精确命中
                **build_document("run ruff format on lumecode/memory/recall.py"),
            },
        ]
        from lumecode.memory.embedding import select_top_k

        results = select_top_k(
            [1.0, 0.0], entries, top_k=5, min_score=0.45, query="ruff format recall.py"
        )

        assert results and Path(results[0].path).name == "exact.md"

    def test_lexical_bonus_cannot_rescue_semantically_unrelated(self) -> None:
        from lumecode.memory.embedding import HYBRID_MIN_COSINE, select_top_k

        entries = [
            {
                "path": "/unrelated.md",
                "mtime_ms": 0,
                "vector": [0.0, 1.0],  # 余弦 0，低于 HYBRID_MIN_COSINE
                **build_document("ruff format recall.py"),
            }
        ]
        assert HYBRID_MIN_COSINE > 0
        assert select_top_k([1.0, 0.0], entries, top_k=5, min_score=0.45, query="ruff format") == []

    def test_entries_without_lexical_payload_still_work(self) -> None:
        from lumecode.memory.embedding import select_top_k

        entries = [{"path": "/a.md", "mtime_ms": 0, "vector": [1.0, 0.0]}]
        results = select_top_k([1.0, 0.0], entries, top_k=5, min_score=0.45, query="anything")
        assert [Path(r.path).name for r in results] == ["a.md"]


class _KeywordEmbedder:
    """按关键词返回确定性向量的假 embedder（不访问网络）。"""

    config = EmbeddingConfig(base_url="http://fake", model="fake", dimensions=2)

    async def embed(self, texts: list[str], text_type: str | None = None) -> list[list[float]]:
        out = []
        for t in texts:
            if "ruff" in t or "format" in t:
                out.append([0.62, 0.78])
            elif "tea" in t or "food" in t:
                out.append([1.0, 0.0])
            else:
                out.append([0.0, 1.0])
        return out


class TestIndexStoresLexicalPayload:
    def test_refresh_writes_tokens_and_entities(self, tmp_path: Path) -> None:
        mem = tmp_path / "mem"
        _memory(mem / "tools.md", "Use `ruff format` on lumecode/memory/recall.py")
        index = VectorMemoryIndex(tmp_path / ".index.json", _KeywordEmbedder())  # type: ignore[arg-type]

        asyncio.run(index.search("ruff format", mem, None, None, top_k=5))

        payload = index._entries[str((mem / "tools.md").resolve())]
        assert isinstance(payload["tokens"], list)
        assert "lumecode/memory/recall.py" in payload["entities"]


# ---------------------------------------------------------------------------
# ③ observed_at 年龄
# ---------------------------------------------------------------------------


class TestObservedAt:
    def test_parse_formats(self) -> None:
        assert parse_observed_ms("2026-10-27T21:05:00Z") > 0
        assert parse_observed_ms("2026-10-27") > 0
        assert parse_observed_ms("2026-10-27 21:05") > 0
        assert parse_observed_ms("") == 0
        assert parse_observed_ms("not a date") == 0

    def test_header_reads_observed_at(self, tmp_path: Path) -> None:
        mem = tmp_path / "mem"
        _memory(mem / "a.md", "old fact", extra="observed_at: 2026-01-01\n")
        headers = scan_memory_files(mem, "project")
        assert headers[0].observed_ms == parse_observed_ms("2026-01-01")

    def test_reminder_age_uses_observed_at_not_mtime(self, tmp_path: Path) -> None:
        old = datetime.now(UTC) - timedelta(days=30)
        path = tmp_path / "old.md"
        _memory(path, "fact", extra=f"observed_at: {old.strftime('%Y-%m-%d')}\n")
        old_ms = parse_observed_ms(old.strftime("%Y-%m-%d"))
        now_ms = int(datetime.now(UTC).timestamp() * 1000)

        # 文件刚被 dream 重写过（mtime = 现在），但事实观察于 30 天前。
        with_observed = render_reminder(
            [RelevantMemory(path=str(path), mtime_ms=now_ms, observed_ms=old_ms)]
        )
        assert "memory is 30 days old" in with_observed
        assert memory_age(old_ms) == "30 days ago"

        # 没有 observed_ms 时退回 mtime（这里也是现在），不会把重写误判为旧记忆。
        without_observed = render_reminder([RelevantMemory(path=str(path), mtime_ms=now_ms)])
        assert "days old" not in without_observed


# ---------------------------------------------------------------------------
# ③ 历史快照
# ---------------------------------------------------------------------------


class TestSnapshots:
    def test_snapshot_file_and_prune(self, tmp_path: Path) -> None:
        mem = tmp_path / "mem"
        path = mem / "a.md"
        _memory(path, "v1")

        for i in range(7):
            _write(path, f"v{i}")
            assert snapshot_file(mem, path, keep=3) is not None

        snapshots = list((mem / HISTORY_DIR_NAME / "a").glob("*.md"))
        assert len(snapshots) == 3
        # 最新一份快照保留的是最近一次改写前的内容
        assert "v6" in max(snapshots, key=lambda p: p.name).read_text(encoding="utf-8")

    def test_iter_memory_files_skips_history(self, tmp_path: Path) -> None:
        mem = tmp_path / "mem"
        _memory(mem / "a.md", "x")
        _write(mem / HISTORY_DIR_NAME / "a" / "20260101T000000Z.md", "old")
        _memory(mem / HISTORY_DIR_NAME / "dream" / "x" / "a.md", "old")

        names = [p.name for p in iter_memory_files(mem)]

        assert names == ["a.md"]

    def test_snapshot_tree_copies_index_and_dreams(self, tmp_path: Path) -> None:
        mem = tmp_path / "mem"
        _memory(mem / "a.md", "x")
        _write(mem / "MEMORY.md", "- [a](a.md) — hook\n")

        dest = snapshot_tree(mem)

        assert dest is not None
        assert (dest / "a.md").read_text(encoding="utf-8").startswith("---")
        assert "hook" in (dest / "MEMORY.md").read_text(encoding="utf-8")

    def test_scan_ignores_history(self, tmp_path: Path) -> None:
        mem = tmp_path / "mem"
        _memory(mem / "a.md", "x")
        _memory(mem / HISTORY_DIR_NAME / "a" / "snap.md", "old")

        assert [Path(h.filename).name for h in scan_memory_files(mem, "project")] == ["a.md"]


# ---------------------------------------------------------------------------
# ② ADD-only 抽取
# ---------------------------------------------------------------------------


class TestAppendMemory:
    def test_new_file_records_observed_at_and_source(self, tmp_path: Path) -> None:
        target = tmp_path / "mem"
        target.mkdir()
        path = target / "user-style.md"
        now = int(datetime(2026, 10, 27, 12, 0, tzinfo=UTC).timestamp() * 1000)

        assert _append_memory(str(target), path, "user", "prefers uv", "Uses uv.", now, "session_a")

        content = path.read_text(encoding="utf-8")
        assert "observed_at: 2026-10-27T12:00:00Z" in content
        assert "source: session session_a" in content
        assert "metadata:\n  type: user" in content
        assert "Uses uv." in content
        assert "user-style.md" in (target / "MEMORY.md").read_text(encoding="utf-8")

    def test_duplicate_body_is_skipped(self, tmp_path: Path) -> None:
        target = tmp_path / "mem"
        target.mkdir()
        path = target / "a.md"
        now = int(datetime.now(UTC).timestamp() * 1000)

        assert _append_memory(str(target), path, "project", "d", "Same fact.", now, "")
        assert not _append_memory(str(target), path, "project", "d", " same   FACT. ", now + 1, "")

    def test_same_name_appends_and_snapshots_never_overwrites(self, tmp_path: Path) -> None:
        target = tmp_path / "mem"
        target.mkdir()
        path = target / "a.md"
        now = int(datetime.now(UTC).timestamp() * 1000)

        _append_memory(str(target), path, "project", "first", "First fact.", now, "s1")
        _append_memory(str(target), path, "project", "second", "Second fact.", now + 60_000, "s2")

        content = path.read_text(encoding="utf-8")
        assert "First fact." in content
        assert "Second fact." in content
        assert "## Update —" in content
        assert "(session s2)" in content
        # 追加前保留了旧版本的快照
        assert list((target / HISTORY_DIR_NAME / "a").glob("*.md"))

    def test_unsafe_name_cannot_escape_memory_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "mem"
        target.mkdir()
        path = target / "x.md"
        now = int(datetime.now(UTC).timestamp() * 1000)

        _append_memory(str(target), path, "project", "d", "fact", now, "")

        assert set(target.rglob("*.md")) == {path, target / "MEMORY.md"}


class _ScriptedClient:
    """按队列返回固定抽取文本的假 LLM 客户端，并记录收到的 prompt。"""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.prompts: list[str] = []
        self.model = "fake"

    async def stream(self, conversation: Any, system: str = ""):
        from lumecode.tools.base import StreamEnd, TextDelta

        self.prompts.append("\n".join(str(m.content) for m in conversation.history))
        yield TextDelta(self.outputs.pop(0))
        yield StreamEnd(stop_reason="end_turn", input_tokens=1, output_tokens=1)


def _extract(manager: MemoryManager, client: Any, text: str, session_id: str = "") -> None:
    conv = ConversationManager()
    conv.history = [Message(role="user", content=text)]
    asyncio.run(manager.extract(client, conv, "anthropic", session_id=session_id))


class TestExtraction:
    def _manager(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryManager:
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
        return MemoryManager(str(tmp_path / "project"))

    def test_extracts_new_file_with_provenance(self, tmp_path: Path, monkeypatch) -> None:
        mgr = self._manager(tmp_path, monkeypatch)
        output = (
            "MEMORY_NAME: Prefer uv\n"
            "MEMORY_TYPE: user\n"
            "MEMORY_DESC: user installs tools with uv\n"
            "MEMORY_BODY: Use uv pip install, not pip.\n"
            "---\n"
        )
        _extract(mgr, _ScriptedClient([output]), "install deps with uv", session_id="s1")

        path = mgr.user_mem_dir / "prefer-uv.md"
        content = path.read_text(encoding="utf-8")
        assert "observed_at:" in content
        assert "source: session s1" in content
        assert "Use uv pip install" in content

    def test_prompt_only_includes_related_memories(self, tmp_path: Path, monkeypatch) -> None:
        mgr = self._manager(tmp_path, monkeypatch)
        mgr.project_mem_dir.mkdir(parents=True)
        _write(
            mgr.project_mem_dir / "ruff.md",
            "---\nname: ruff\ndescription: formatting\ntype: project\n---\n\nRun ruff format.\n",
        )
        _write(
            mgr.project_mem_dir / "tea.md",
            "---\nname: tea\ndescription: drinks\ntype: project\n---\n\nLikes tea.\n",
        )
        client = _ScriptedClient(["NONE"])
        _extract(mgr, client, "ruff format failing on recall.py")

        assert client.prompts
        prompt = client.prompts[0]
        assert "ruff.md" in prompt
        assert "tea.md" not in prompt

    def test_extraction_prompt_is_add_only(self, tmp_path: Path, monkeypatch) -> None:
        mgr = self._manager(tmp_path, monkeypatch)
        client = _ScriptedClient(["NONE"])
        _extract(mgr, client, "some conversation")

        assert "Only extract facts NOT already covered" in client.prompts[0]
        assert "Do not rewrite or merge existing memories" in client.prompts[0]

    def test_second_identical_extraction_is_deduped(self, tmp_path: Path, monkeypatch) -> None:
        mgr = self._manager(tmp_path, monkeypatch)
        output = (
            "MEMORY_NAME: uv-tooling\n"
            "MEMORY_TYPE: project\n"
            "MEMORY_DESC: uv tooling\n"
            "MEMORY_BODY: Always use uv.\n"
            "---\n"
        )
        _extract(mgr, _ScriptedClient([output]), "use uv", session_id="s1")
        _extract(mgr, _ScriptedClient([output]), "use uv again", session_id="s2")

        content = (mgr.project_mem_dir / "uv-tooling.md").read_text(encoding="utf-8")
        assert content.count("Always use uv.") == 1
        assert "## Update —" not in content

    def test_unsafe_memory_name_is_sanitized(self, tmp_path: Path, monkeypatch) -> None:
        mgr = self._manager(tmp_path, monkeypatch)
        output = (
            "MEMORY_NAME: ../../evil\n"
            "MEMORY_TYPE: project\n"
            "MEMORY_DESC: evil\n"
            "MEMORY_BODY: nope\n"
            "---\n"
        )
        _extract(mgr, _ScriptedClient([output]), "x")

        files = list(mgr.project_mem_dir.glob("*.md"))
        assert not (tmp_path / "evil.md").exists()
        assert any(f.name != "MEMORY.md" for f in files)


# ---------------------------------------------------------------------------
# 提取频率门槛
# ---------------------------------------------------------------------------


class TestExtractionGating:
    def _agent(self, interval: int) -> Any:
        from lumecode.agent import Agent
        from lumecode.memory.auto_memory import MemoryManager
        from lumecode.tools import create_default_registry

        class _Client:
            async def stream(self, conversation: Any, system: str = "", tools: Any = None):
                from lumecode.tools.base import StreamEnd, TextDelta

                yield TextDelta("ok")
                yield StreamEnd(stop_reason="end_turn", input_tokens=1, output_tokens=1)

        mgr = MemoryManager(".")
        # 只关心门控，不真正调 LLM：用无会话历史时提前 return。
        return Agent(
            _Client(),  # type: ignore[arg-type]
            create_default_registry(),
            "anthropic",
            work_dir=".",
            memory_manager=mgr,
            memory_extraction_interval=interval,
        )

    def _conv(self, messages: int) -> Any:
        conv = ConversationManager()
        for i in range(messages):
            conv.history.append(Message(role="user", content=f"m{i}"))
        return conv

    def test_interval_zero_never_triggers_at_turn_end(self) -> None:
        agent = self._agent(0)
        agent._loop_count = 8
        assert not agent._should_extract(self._conv(20))

    def test_waits_for_interval(self) -> None:
        agent = self._agent(4)
        conv = self._conv(10)
        agent.memory_manager.extracted_message_count = 0

        agent._loop_count = 1
        assert not agent._should_extract(conv)
        agent._loop_count = 4
        assert agent._should_extract(conv)

    def test_requires_enough_new_messages(self) -> None:
        from lumecode.agent import MEMORY_EXTRACTION_MIN_MESSAGES

        agent = self._agent(1)
        agent._loop_count = 1
        conv = self._conv(MEMORY_EXTRACTION_MIN_MESSAGES - 1)
        assert not agent._should_extract(conv)

        conv = self._conv(MEMORY_EXTRACTION_MIN_MESSAGES)
        assert agent._should_extract(conv)

    def test_no_unseen_messages_is_noop(self) -> None:
        agent = self._agent(1)
        agent._loop_count = 1
        conv = self._conv(10)
        agent.memory_manager.extracted_message_count = 10
        assert not agent._should_extract(conv)

    def test_no_memory_manager_never_extracts(self) -> None:
        from lumecode.agent import Agent
        from lumecode.tools import create_default_registry

        agent = Agent(object(), create_default_registry(), "anthropic", work_dir=".")  # type: ignore[arg-type]
        agent._loop_count = 1
        assert not agent._should_extract(self._conv(20))
