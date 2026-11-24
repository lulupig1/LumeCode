"""基于 embedding 的记忆混合召回。

用 OpenAI 兼容的 /v1/embeddings 端点把记忆文件向量化，索引以项目为单位
缓存在 `.lumecode/memory/.index.json`，只在文件 mtime / 大小变化时重算
对应文件的向量。

查询时不是纯余弦排序，而是融合三路信号（对齐 Mem0 v3 的 hybrid retrieval）：

- 向量语义相似度（主信号，阈值仍按其原有分布校准）；
- BM25 字面匹配（文件名、命令、符号名等精确 token）；
- 实体重叠加分（文件路径、引号内容、标识符）。

设计约束：
- 任何失败（端点不可达、维度不匹配、索引损坏）都只记日志并返回空结果，
  由调用方回退到 LLM selector。召回绝不能阻塞或打断主对话。
- 候选文本 = frontmatter 描述 + MEMORY.md 索引 hook + 正文前段，兼顾标题、
  索引意图和正文关键信息，同时控制嵌入成本。
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path

import httpx

from lumecode.config import EmbeddingConfig
from lumecode.memory.recall import MemoryHeader, RelevantMemory, collect_memory_headers
from lumecode.memory.textmatch import (
    bm25_scores,
    build_document,
    extract_entities,
    fuse_scores,
)

log = logging.getLogger(__name__)

INDEX_FILE_NAME = ".index.json"
# v2：条目新增词法载荷（tokens / doc_len / entities）。旧索引自动重建。
INDEX_VERSION = 2
EMBEDDING_BATCH_SIZE = 10
EMBEDDING_TIMEOUT = 10.0
# 正文参与 embedding 的字符数上限。太短会丢语义，太长会推高索引成本。
EMBEDDING_BODY_CHARS = 1500
# 余弦相似度下限：向量排序总有 top-k，没有阈值会把无关记忆也注入。
# 实测 text-embedding-v3 上真正相关的记忆得分约 0.63~0.80，无关查询的
# 最高分约 0.43，因此取 0.45 作为默认分界；不同模型分布不同，可通过
# memory.embedding.min_score 覆盖。
DEFAULT_MIN_SIMILARITY = 0.45
# 相对阈值：得分低于 top-1 的这个比例时丢弃。避免 top-1 很强时把
# 明显弱一档的条目也带进来。
RELATIVE_SCORE_RATIO = 0.7
# 混合召回里"纯字面信号"能救回的最低语义分。低于它说明语义上已经无关，
# 再强的 BM25/实体命中也不该注入——防止记忆库很小、IDF 失效时把
# 只共享常见词的记忆误召回。
HYBRID_MIN_COSINE = 0.25


class EmbeddingClient:
    """OpenAI 兼容的 embedding 客户端（目前只实现了 batch embed）。"""

    def __init__(
        self,
        config: EmbeddingConfig,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self._client = client
        self._owns_client = client is None

    async def __aenter__(self) -> EmbeddingClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def embed(self, texts: list[str], text_type: str | None = None) -> list[list[float]]:
        """批量取向量。调用方负责把批大小控制在供应商上限内。

        text_type 只在 config.uses_text_type 打开时发送（'query' / 'document'）。
        """
        if not texts:
            return []
        if not self.config.enabled:
            raise ValueError("embedding config is not enabled")
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=EMBEDDING_TIMEOUT)

        payload: dict[str, object] = {"model": self.config.model, "input": texts}
        if self.config.dimensions > 0:
            payload["dimensions"] = self.config.dimensions
        if text_type and self.config.uses_text_type:
            payload["text_type"] = text_type
        headers = {"Content-Type": "application/json"}
        api_key = self.config.resolve_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        url = self.config.base_url.rstrip("/") + "/embeddings"
        resp = await self._client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        entries = data.get("data")
        if not isinstance(entries, list) or len(entries) != len(texts):
            count = len(entries) if isinstance(entries, list) else 0
            raise ValueError(f"embedding response has {count} vectors for {len(texts)} inputs")
        # 供应商不保证返回顺序，按 index 对齐。
        ordered = sorted(entries, key=lambda e: e.get("index", 0))
        return [list(e["embedding"]) for e in ordered]


def build_embedding_text(header: MemoryHeader, file_path: Path) -> str:
    """组合用于 embed 的文本：描述 + hook + 正文前段。"""
    parts = [header.hook or header.description]
    parts.append(header.filename.replace("\\", "/"))
    try:
        body = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        body = ""
    if body:
        parts.append(body[:EMBEDDING_BODY_CHARS])
    return "\n".join(p for p in parts if p)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def select_top_k(
    query_vec: list[float],
    entries: list[dict],
    top_k: int,
    min_score: float = 0.0,
    query: str = "",
) -> list[RelevantMemory]:
    """对索引条目做混合排序，返回 top-k 个 RelevantMemory。

    分数 = 余弦 + 0.4×BM25 + 实体重叠加分（query 为空或条目缺少词法载荷时
    退化为纯余弦）。低于 min_score、低于 top-1 得分 RELATIVE_SCORE_RATIO 倍、
    或纯字面信号救回但余弦低于 HYBRID_MIN_COSINE 的条目会被丢弃——排序总有
    结果，没有阈值就会把与查询无关的记忆也注入进去。
    """
    valid = [entry for entry in entries if _vector_matches(entry, query_vec)]
    if not valid:
        return []

    cosines = [cosine_similarity(query_vec, entry["vector"]) for entry in valid]
    use_lexical = bool(query) and any(isinstance(e.get("tokens"), list) for e in valid)
    if use_lexical:
        lex_scores = bm25_scores(query, valid)
        query_entities = extract_entities(query)
        fused = [
            fuse_scores(
                cos,
                lex,
                _entity_overlap(query_entities, entry.get("entities")),
            )
            for cos, lex, entry in zip(cosines, lex_scores, valid, strict=True)
        ]
    else:
        fused = cosines

    scored = [
        (score, cos, entry)
        for score, cos, entry in zip(fused, cosines, valid, strict=True)
        if score >= min_score and cos >= HYBRID_MIN_COSINE
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    if scored:
        floor = scored[0][0] * RELATIVE_SCORE_RATIO
        scored = [pair for pair in scored if pair[0] >= floor]

    results: list[RelevantMemory] = []
    for _score, _cos, entry in scored[:top_k]:
        path = entry.get("path")
        if not path:
            continue
        mtime_ms = entry.get("mtime_ms", 0)
        observed_ms = entry.get("observed_ms", 0)
        results.append(
            RelevantMemory(
                path=str(path),
                mtime_ms=int(mtime_ms),
                observed_ms=int(observed_ms) if isinstance(observed_ms, int) else 0,
            )
        )
    return results


def _vector_matches(entry: dict, query_vec: list[float]) -> bool:
    vec = entry.get("vector")
    return isinstance(vec, list) and len(vec) == len(query_vec)


def _entity_overlap(query_entities: set[str], entry_entities: object) -> int:
    if not query_entities or not isinstance(entry_entities, list):
        return 0
    return sum(1 for e in entry_entities if isinstance(e, str) and e in query_entities)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class VectorMemoryIndex:
    """记忆文件 -> 向量的项目级缓存。"""

    def __init__(self, index_path: Path, client: EmbeddingClient | None) -> None:
        self.index_path = index_path
        self.client = client
        self._entries: dict[str, dict] = {}
        self._loaded = False

    @property
    def available(self) -> bool:
        return self.client is not None and self.client.config.enabled

    def load(self) -> None:
        """读取索引文件。损坏或版本不符时放弃缓存，下次重建。"""
        self._entries = {}
        self._loaded = True
        try:
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(raw, dict) or raw.get("version") != INDEX_VERSION:
            return
        entries = raw.get("entries")
        if isinstance(entries, dict):
            self._entries = entries

    def save(self) -> None:
        if not self._loaded:
            return
        payload = {"version": INDEX_VERSION, "entries": self._entries}
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.index_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(tmp, self.index_path)
        except OSError as e:
            log.debug("failed to persist memory index at %s: %s", self.index_path, e)

    def clear(self) -> None:
        self._entries = {}
        self.save()

    async def search(
        self,
        query: str,
        user_mem_dir: Path | None,
        project_mem_dir: Path | None,
        already_surfaced: set[str] | None,
        top_k: int,
    ) -> list[RelevantMemory]:
        """同步索引并返回与 query 最相似的 top-k 条记忆。

        返回空列表表示"不可用 / 无候选"，调用方应回退到 LLM selector。
        抛出异常同样视为不可用。
        """
        if not self.available or self.client is None:
            return []
        if not self._loaded:
            self.load()

        candidates = collect_memory_headers(user_mem_dir, project_mem_dir, already_surfaced)
        if not candidates:
            return []

        up_to_date = await self._refresh(candidates)
        if not up_to_date:
            return []

        query_vec = (await self.client.embed([query], text_type="query"))[0]
        entries = [
            entry for entry in self._entries.values() if Path(entry.get("path", "")).exists()
        ]
        return select_top_k(
            query_vec,
            entries,
            top_k,
            min_score=self.client.config.min_score,
            query=query,
        )

    async def _refresh(self, candidates: list[MemoryHeader]) -> bool:
        """为发生变化的候选重算向量，并清理已删除文件的索引条目。

        返回 True 表示索引与候选一致。任何 embedding 失败返回 False，
        让调用方回退，而不是用半新半旧的索引给出不可靠的排序。
        """
        assert self.client is not None

        dirty: list[tuple[str, MemoryHeader, str, str]] = []
        seen: set[str] = set()
        for header in candidates:
            path = header.file_path
            seen.add(path)
            try:
                stat = Path(path).stat()
            except OSError:
                continue
            text = build_embedding_text(header, Path(path))
            text_hash = _hash_text(text)
            cached = self._entries.get(path)
            if (
                isinstance(cached, dict)
                and cached.get("mtime_ms") == int(stat.st_mtime * 1000)
                and cached.get("size") == stat.st_size
                and cached.get("hash") == text_hash
            ):
                continue
            dirty.append((path, header, text, text_hash))

        # 清理已删除文件的陈旧条目。索引本身是项目级缓存，整个缓存都可以重建，
        # 因此不必按目录前缀做局部清理。
        for path in list(self._entries):
            if path not in seen and not Path(path).exists():
                del self._entries[path]

        if not dirty:
            return True

        try:
            for start in range(0, len(dirty), EMBEDDING_BATCH_SIZE):
                batch = dirty[start : start + EMBEDDING_BATCH_SIZE]
                vectors = await self.client.embed(
                    [text for _, _, text, _ in batch], text_type="document"
                )
                for (path, _header, _text, text_hash), vec in zip(batch, vectors, strict=True):
                    try:
                        stat = Path(path).stat()
                    except OSError:
                        continue
                    self._entries[path] = {
                        "path": path,
                        "mtime_ms": int(stat.st_mtime * 1000),
                        "observed_ms": _header.observed_ms,
                        "size": stat.st_size,
                        "hash": text_hash,
                        "vector": vec,
                        **build_document(text),
                    }
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as e:
            log.warning("memory embedding refresh failed, falling back to LLM selector: %s", e)
            return False

        self.save()
        return True
