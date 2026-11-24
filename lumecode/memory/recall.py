from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from lumecode.memory.history import iter_memory_files

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_MEMORY_FILES = 200
FRONTMATTER_MAX_LINES = 30
ENTRYPOINT_NAME = "MEMORY.md"
VALID_TYPES = {"user", "feedback", "project", "reference"}

# 注入給模型的记忆总量上限（字符）。超过时按选择顺序保留，并标注截断。
MAX_REMINDER_CHARS = 24_000

FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# MEMORY.md 索引行：`- [Title](file.md) — hook`。
# 索引是召回时的重要线索——标题和 hook 往往是正文里最凝练的一句话。
_ENTRYPOINT_LINE_RE = re.compile(r"^\s*[-*]\s+\[([^\]]*)\]\(([^)]+)\)\s*(?:[—–-]\s*(.+))?$")

SELECTOR_SYSTEM_PROMPT = (
    "You are selecting memories that will be useful to LumeCode as it processes "
    "a user's query. You will be given the user's query and a list of available "
    "memory files with their filenames and descriptions.\n\n"
    "Return a list of filenames for the memories that will clearly be useful to "
    "LumeCode as it processes the user's query (up to 5). Only include memories "
    "that you are certain will be helpful based on their name and description.\n"
    "- If you are unsure if a memory will be useful in processing the user's "
    "query, then do not include it in your list. Be selective and discerning.\n"
    "- If there are no memories in the list that would clearly be useful, feel "
    "free to return an empty list.\n"
    "- If a list of recently-used tools is provided, do not select memories "
    "that are usage reference or API documentation for those tools (LumeCode is "
    "already exercising them). DO still select memories containing warnings, "
    "gotchas, or known issues about those tools — active use is exactly when "
    "those matter.\n\n"
    "Respond with valid JSON only, no markdown, in this exact shape: "
    '{"selected_memories": ["filename1.md", "filename2.md"]}'
)

# Type alias for the side-query selector function.
SelectorFn = Callable[[str, str], Awaitable[str]]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MemoryHeader:
    filename: str  # path relative to memory_dir
    file_path: str  # absolute path
    scope: str  # "user" or "project"
    mtime_ms: int  # file modification time, ms since epoch（仅用于缓存失效）
    description: str  # frontmatter description; "" if absent
    type: str  # frontmatter type; "" if unrecognized
    hook: str = ""  # 一行摘要：frontmatter description 优先，否则取 MEMORY.md 索引 hook
    # 事实被观察到的时间（frontmatter `observed_at`），ms since epoch；缺失为 0。
    # 年龄/时效判断必须用它：dream 整理会重写文件，mtime 不代表事实新旧。
    observed_ms: int = 0


@dataclass
class RelevantMemory:
    path: str
    mtime_ms: int
    observed_ms: int = 0

    @property
    def saved_ms(self) -> int:
        """用于计算年龄的时间：优先 observed_at，缺失时退回文件 mtime。"""
        return self.observed_ms or self.mtime_ms


@dataclass
class RecallResult:
    reminder: str  # 渲染好的 system-reminder 正文；"" 表示没有可注入的记忆
    paths: list[str]  # 本次注入的记忆文件绝对路径，用于跨轮去重


# ---------------------------------------------------------------------------
# Memory age helpers
# ---------------------------------------------------------------------------


def memory_age_days(mtime_ms: int) -> int:
    """Floor-rounded days since mtime. 0 for today, 1 for yesterday, etc."""
    d = (int(time.time() * 1000) - mtime_ms) // 86_400_000
    return max(d, 0)


def memory_age(mtime_ms: int) -> str:
    """Human-readable age: 'today', 'yesterday', or 'N days ago'."""
    d = memory_age_days(mtime_ms)
    if d == 0:
        return "today"
    if d == 1:
        return "yesterday"
    return f"{d} days ago"


def memory_freshness_text(mtime_ms: int) -> str:
    """Staleness warning for memories older than 1 day. Returns '' for fresh."""
    d = memory_age_days(mtime_ms)
    if d <= 1:
        return ""
    return (
        f"This memory is {d} days old. "
        "Memories are point-in-time observations, not live state — "
        "claims about code behavior or file:line citations may be outdated. "
        "Verify against current code before asserting as fact."
    )


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------


def parse_frontmatter(content: str) -> dict[str, str]:
    """Extract name/description/type/observed_at from YAML-ish frontmatter.

    Only the known fields are read; everything else is ignored.
    Files without frontmatter return empty fields.
    """
    keys = ("name", "description", "type", "observed_at", "source")
    m = FRONTMATTER_RE.match(content)
    if not m:
        return dict.fromkeys(keys, "")

    block = m.group(1)
    result: dict[str, str] = dict.fromkeys(keys, "")
    for line in block.split("\n"):
        colon = line.find(":")
        if colon < 0:
            continue
        key = line[:colon].strip()
        val = line[colon + 1 :].strip()
        # Strip quotes.
        if len(val) >= 2 and (
            (val.startswith('"') and val.endswith('"'))
            or (val.startswith("'") and val.endswith("'"))
        ):
            val = val[1:-1]
        if key == "name":
            result["name"] = val
        elif key == "description":
            result["description"] = val
        elif key == "type" and val in VALID_TYPES:
            result["type"] = val
        elif key == "observed_at":
            result["observed_at"] = val
        elif key == "source":
            result["source"] = val
    return result


def parse_observed_ms(value: str) -> int:
    """把 frontmatter ``observed_at`` 解析为毫秒时间戳；无法解析返回 0。

    接受 ISO-8601（``2026-10-27``、``2026-10-27T21:05:00Z``）和
    ``YYYY-MM-DD HH:MM``。无时区的日期按当日 00:00 处理。
    """
    text = (value or "").strip()
    if not text:
        return 0
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    for fmt in (None, "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            dt = datetime.fromisoformat(normalized) if fmt is None else datetime.strptime(text, fmt)
        except ValueError:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return int(dt.timestamp() * 1000)
    return 0


def format_observed_at(ms: int) -> str:
    """毫秒时间戳 → frontmatter 用的 UTC ISO-8601（秒精度）。"""
    dt = datetime.fromtimestamp(ms / 1000, tz=UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


def scan_memory_files(memory_dir: Path, scope: str) -> list[MemoryHeader]:
    """Walk memory_dir for .md files (excluding MEMORY.md and .history),
    read frontmatter and the matching MEMORY.md index hook from each, and
    return a header list sorted newest-first, capped at MAX_MEMORY_FILES.
    """
    if not memory_dir.is_dir():
        return []

    hooks = _read_entrypoint_hooks(memory_dir / ENTRYPOINT_NAME)
    results: list[MemoryHeader] = []
    for fp in iter_memory_files(memory_dir):
        hdr = _read_memory_header(fp, memory_dir, scope)
        if hdr is not None:
            index_hook = hooks.get(hdr.filename) or hooks.get(fp.name, "")
            hdr.hook = hdr.description or index_hook or fp.stem
            results.append(hdr)

    # Sort newest-first（按观察时间；旧文件没有 observed_at 时退回 mtime）。
    results.sort(key=lambda h: h.observed_ms or h.mtime_ms, reverse=True)
    if len(results) > MAX_MEMORY_FILES:
        log.warning(
            "memory recall candidates capped: %d files found, oldest %d dropped (%s scope, %s)",
            len(results),
            len(results) - MAX_MEMORY_FILES,
            scope,
            memory_dir,
        )
        results = results[:MAX_MEMORY_FILES]
    return results


def _read_entrypoint_hooks(entrypoint: Path) -> dict[str, str]:
    """解析 MEMORY.md 索引行，返回 {相对路径或文件名: "标题 — hook"}。

    解析失败或文件不存在时返回空字典——索引只是召回线索，不影响主流程。
    """
    try:
        text = entrypoint.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}

    hooks: dict[str, str] = {}
    for line in text.splitlines():
        m = _ENTRYPOINT_LINE_RE.match(line)
        if not m:
            continue
        title = m.group(1).strip()
        target = m.group(2).strip()
        hook = (m.group(3) or "").strip()
        # 只取文件名做键：索引里有时写 `memory/foo.md`，有时只写 `foo.md`。
        key = Path(target.replace("\\", "/")).name
        text_hook = f"{title} — {hook}" if hook else title
        hooks.setdefault(key, text_hook)
        hooks.setdefault(target, text_hook)
    return hooks


def _read_memory_header(file_path: Path, memory_dir: Path, scope: str) -> MemoryHeader | None:
    try:
        mtime_ms = int(file_path.stat().st_mtime * 1000)
    except OSError:
        return None

    # Read first FRONTMATTER_MAX_LINES for frontmatter parsing.
    try:
        lines: list[str] = []
        with file_path.open(encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= FRONTMATTER_MAX_LINES:
                    break
                lines.append(line)
        content = "".join(lines)
    except OSError:
        return None

    fm = parse_frontmatter(content)
    try:
        rel = str(file_path.relative_to(memory_dir))
    except ValueError:
        rel = file_path.name

    return MemoryHeader(
        filename=rel,
        file_path=str(file_path.resolve()),
        scope=scope,
        mtime_ms=mtime_ms,
        description=fm["description"],
        type=fm["type"],
        observed_ms=parse_observed_ms(fm.get("observed_at", "")),
    )


# ---------------------------------------------------------------------------
# Manifest formatting
# ---------------------------------------------------------------------------


def format_memory_manifest(memories: list[MemoryHeader]) -> str:
    """Format memory headers as a text manifest for the selector prompt."""
    if not memories:
        return ""
    lines: list[str] = []
    for m in memories:
        scope_tag = f"[{m.scope}-scope] " if m.scope else ""
        type_tag = f"[{m.type}] " if m.type else ""
        ts = (
            datetime.fromtimestamp(m.mtime_ms / 1000, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.")
            + f"{m.mtime_ms % 1000:03d}Z"
        )
        path = m.file_path if m.file_path else m.filename
        # hook 优先取 frontmatter 描述，其次取 MEMORY.md 索引里的一行摘要。
        # 两者都没有时退化为文件名，避免出现"只有路径"的死条目。
        label = m.hook or m.description
        if label:
            lines.append(f"- {scope_tag}{type_tag}{path} ({ts}): {label}")
        else:
            lines.append(f"- {scope_tag}{type_tag}{path} ({ts})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Find relevant memories
# ---------------------------------------------------------------------------


def collect_memory_headers(
    user_mem_dir: Path | None,
    project_mem_dir: Path | None,
    already_surfaced: set[str] | None = None,
) -> list[MemoryHeader]:
    """扫描两个记忆目录，返回尚未注入过的候选（用户级在前）。"""
    all_headers: list[MemoryHeader] = []
    if user_mem_dir is not None:
        all_headers.extend(scan_memory_files(user_mem_dir, "user"))
    if project_mem_dir is not None:
        all_headers.extend(scan_memory_files(project_mem_dir, "project"))

    surfaced = already_surfaced or set()
    return [m for m in all_headers if m.file_path not in surfaced]


def build_recall_result(memories: list[RelevantMemory]) -> RecallResult:
    """把已选中的记忆渲染成 system-reminder，并记录路径供跨轮去重。"""
    return RecallResult(reminder=render_reminder(memories), paths=[m.path for m in memories])


async def find_relevant_memories(
    query: str,
    user_mem_dir: Path | None,
    project_mem_dir: Path | None,
    recent_tools: list[str] | None,
    already_surfaced: set[str] | None,
    selector: SelectorFn,
) -> list[RelevantMemory]:
    """Scan both dirs, filter already-surfaced, ask selector to pick up to 5
    relevant filenames, and return the corresponding paths + mtimes.

    Selector failures are silent — recall is best-effort and must never block
    the main conversation.
    """
    candidates = collect_memory_headers(user_mem_dir, project_mem_dir, already_surfaced)
    if not candidates:
        return []

    selected_filenames = await _select_relevant_memories(query, candidates, recent_tools, selector)

    # Build lookup from both file_path and filename to header.
    by_key: dict[str, MemoryHeader] = {}
    for m in candidates:
        by_key[m.file_path] = m
        by_key.setdefault(m.filename, m)

    result: list[RelevantMemory] = []
    for fn in selected_filenames:
        hdr = by_key.get(fn)
        if hdr is not None:
            result.append(
                RelevantMemory(
                    path=hdr.file_path,
                    mtime_ms=hdr.mtime_ms,
                    observed_ms=hdr.observed_ms,
                )
            )
    return result


async def _select_relevant_memories(
    query: str,
    memories: list[MemoryHeader],
    recent_tools: list[str] | None,
    selector: SelectorFn,
) -> list[str]:
    """Format manifest, call selector, parse JSON, return valid filenames."""
    valid_filenames = {m.filename for m in memories}

    manifest = format_memory_manifest(memories)

    tools_section = ""
    if recent_tools:
        tools_section = "\n\nRecently used tools: " + ", ".join(recent_tools)

    user_message = f"Query: {query}\n\nAvailable memories:\n{manifest}{tools_section}"

    try:
        raw = await selector(SELECTOR_SYSTEM_PROMPT, user_message)
    except Exception as e:
        log.debug("memory selector failed: %s", e)
        return []

    clean = _extract_json_object(raw)
    if not clean:
        return []

    try:
        parsed = json.loads(clean)
        arr = parsed.get("selected_memories", [])
        if not isinstance(arr, list):
            return []
        return [f for f in arr if isinstance(f, str) and f in valid_filenames]
    except (json.JSONDecodeError, AttributeError):
        return []


def _extract_json_object(raw: str) -> str:
    """Return the first {...} substring found in raw. Tolerates markdown
    fences or prose around the JSON.
    """
    trimmed = raw.strip()
    if trimmed.startswith("{"):
        return trimmed
    start = trimmed.find("{")
    if start < 0:
        return ""
    end = trimmed.rfind("}")
    if end < start:
        return ""
    return trimmed[start : end + 1]


# ---------------------------------------------------------------------------
# Reminder rendering
# ---------------------------------------------------------------------------


def render_reminder(memories: list[RelevantMemory]) -> str:
    """Read each selected memory file's full content and format a single
    system-reminder body with freshness headers.

    注入总量受 MAX_REMINDER_CHARS 限制：按选择顺序保留完整文件，放不下的
    整个跳过并在末尾标注数量，避免静默截断或半截内容。
    """
    if not memories:
        return ""

    parts: list[str] = []
    parts.append("The following relevant memories from prior conversations may help:\n")
    used = len(parts[0])
    included = 0
    omitted = 0
    for mem in memories:
        try:
            content = Path(mem.path).read_text(encoding="utf-8")
        except OSError:
            continue  # skip unreadable files
        basename = Path(mem.path).name
        chunk_parts = [f"## Memory: {basename} (saved {memory_age(mem.saved_ms)})\n"]
        note = memory_freshness_text(mem.saved_ms)
        if note:
            chunk_parts.append(note + "\n")
        chunk_parts.append(content + "\n\n---\n")
        chunk = "\n".join(chunk_parts)

        if included > 0 and used + len(chunk) > MAX_REMINDER_CHARS:
            omitted += 1
            continue
        parts.append(chunk)
        used += len(chunk)
        included += 1

    if included == 0:
        return ""
    if omitted:
        parts.append(
            f"({omitted} additional selected memories omitted to stay within "
            f"the {MAX_REMINDER_CHARS}-char recall budget.)\n"
        )
    return "\n".join(parts)
