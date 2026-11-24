"""自动记忆管理器（对齐 Go 版 memory.Manager + memdir + paths）。

使用独立 .md 文件 + frontmatter + MEMORY.md 索引的存储格式，
替代旧版集中式 memories.md。每条记忆存为一个文件，MEMORY.md
只保存索引指针。
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lumecode.conversation import ConversationManager, Message
from lumecode.memory.history import iter_memory_files, snapshot_file
from lumecode.memory.recall import format_observed_at
from lumecode.telemetry import SpanKind, SpanStatus, Usage

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 记忆索引文件名
ENTRYPOINT_NAME = "MEMORY.md"

# 四种记忆类型（对齐 Go 版 MemoryType）
VALID_TYPES = {"user", "feedback", "project", "reference"}

# 记忆类型到存储目录的路由：user/feedback → 用户级，project/reference → 项目级
_USER_LEVEL_TYPES = {"user", "feedback"}
_PROJECT_LEVEL_TYPES = {"project", "reference"}

# MEMORY.md 截断限制
MAX_ENTRYPOINT_LINES = 200
MAX_ENTRYPOINT_BYTES = 25_000

# 抽取时提供给 LLM 的"相关已有记忆"条数。只挑与本次对话词法相关的记忆做
# 去重上下文（对齐 Mem0 v3：top-k 相关记忆，而不是全量 manifest）。
RELATED_MEMORY_LIMIT = 10
RELATED_BODY_CHARS = 240

# frontmatter 正则
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_UNSAFE_NAME_RE = re.compile(r"[^a-z0-9._-]+")


# ---------------------------------------------------------------------------
# 路径工具函数（对齐 Go 版 paths.go）
# ---------------------------------------------------------------------------


def get_auto_mem_path(project_root: str) -> str:
    """返回项目级记忆目录路径：<projectRoot>/.lumecode/memory/。

    保留尾部分隔符，确保前缀匹配不会误命中类似 memoryxyz 的路径。
    支持 LUMECODE_REMOTE_MEMORY_DIR 环境变量覆盖。
    """
    override = os.environ.get("LUMECODE_REMOTE_MEMORY_DIR", "")
    if override:
        return override.rstrip(os.sep) + os.sep
    abs_root = os.path.abspath(project_root)
    return os.path.join(abs_root, ".lumecode", "memory") + os.sep


def get_user_auto_mem_path() -> str:
    """返回用户级记忆目录路径：~/.lumecode/memory/。

    用于存储 type=user / type=feedback 的记忆，跨项目跟随用户。
    如果 HOME 无法解析则返回空字符串。
    """
    try:
        home = str(Path.home())
    except RuntimeError:
        return ""
    if not home:
        return ""
    return os.path.join(home, ".lumecode", "memory") + os.sep


def is_auto_mem_path(absolute_path: str, project_root: str) -> bool:
    """检查路径是否在项目级或用户级记忆目录内。"""
    abs_p = os.path.normpath(absolute_path) + os.sep
    project_dir = get_auto_mem_path(project_root)
    if project_dir and abs_p.startswith(project_dir):
        return True
    user_dir = get_user_auto_mem_path()
    return bool(user_dir and abs_p.startswith(user_dir))


def ensure_memory_dir_exists(memory_dir: str) -> None:
    """确保记忆目录存在，agent 可直接写入无需先 mkdir。"""
    if memory_dir:
        os.makedirs(memory_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# Frontmatter 解析（对齐 Go 版 parseFrontmatter）
# ---------------------------------------------------------------------------


@dataclass
class MemoryFile:
    """一个记忆文件的元信息。"""

    path: str = ""
    name: str = ""
    description: str = ""
    type: str = ""


def parse_frontmatter(content: str) -> MemoryFile:
    """从 YAML-ish frontmatter 中提取 name/description/type。

    只读取三个已知字段，未知字段忽略。没有 frontmatter 的文件返回空字段。
    """
    mf = MemoryFile()
    m = _FRONTMATTER_RE.match(content)
    if not m:
        return mf
    for line in m.group(1).split("\n"):
        colon = line.find(":")
        if colon < 0:
            continue
        key = line[:colon].strip()
        val = line[colon + 1 :].strip()
        # 去除引号
        if len(val) >= 2 and (
            (val.startswith('"') and val.endswith('"'))
            or (val.startswith("'") and val.endswith("'"))
        ):
            val = val[1:-1]
        if key == "name":
            mf.name = val
        elif key == "description":
            mf.description = val
        elif key == "type" and val in VALID_TYPES:
            mf.type = val
    return mf


# ---------------------------------------------------------------------------
# MEMORY.md 截断（对齐 Go 版 TruncateEntrypointContent）
# ---------------------------------------------------------------------------


def truncate_entrypoint_content(raw: str) -> str:
    """截断 MEMORY.md 内容，超过行数或字节限制时添加警告。"""
    trimmed = raw.strip()
    lines = trimmed.split("\n")
    line_count = len(lines)
    byte_count = len(trimmed.encode("utf-8"))

    over_lines = line_count > MAX_ENTRYPOINT_LINES
    over_bytes = byte_count > MAX_ENTRYPOINT_BYTES

    if not over_lines and not over_bytes:
        return trimmed

    result = trimmed
    if over_lines:
        result = "\n".join(lines[:MAX_ENTRYPOINT_LINES])

    result_bytes = result.encode("utf-8")
    if len(result_bytes) > MAX_ENTRYPOINT_BYTES:
        cut = result[:MAX_ENTRYPOINT_BYTES].rfind("\n")
        if cut > 0:
            result = result[:cut]
        else:
            result = result[:MAX_ENTRYPOINT_BYTES]

    # 构建警告信息
    if over_bytes and not over_lines:
        reason = f"{_format_size(byte_count)} (limit: {_format_size(MAX_ENTRYPOINT_BYTES)}) — index entries are too long"
    elif over_lines and not over_bytes:
        reason = f"{line_count} lines (limit: {MAX_ENTRYPOINT_LINES})"
    else:
        reason = f"{line_count} lines and {_format_size(byte_count)}"

    result += (
        f"\n\n> WARNING: {ENTRYPOINT_NAME} is {reason}. "
        "Only part of it was loaded. Keep index entries to one line "
        "under ~200 chars; move detail into topic files."
    )
    return result


def _format_size(byte_count: int) -> str:
    if byte_count < 1024:
        return f"{byte_count}B"
    if byte_count < 1024 * 1024:
        return f"{byte_count / 1024:.1f}KB"
    return f"{byte_count / (1024 * 1024):.1f}MB"


# ---------------------------------------------------------------------------
# 构建记忆系统提示（对齐 Go 版 BuildMemoryPrompt）
# ---------------------------------------------------------------------------


def build_memory_prompt(user_mem_dir: str, project_mem_dir: str) -> str:
    """构建记忆系统提示，包含行为指令和 MEMORY.md 索引内容。

    对齐 Go 版 BuildMemoryPrompt：组合类型化记忆行为指令 + 两个 MEMORY.md
    的内容，生成完整的 '# auto memory' 系统提示段。
    """
    lines = _build_memory_lines(user_mem_dir, project_mem_dir)
    parts = [lines]

    if user_mem_dir:
        ep_path = os.path.join(user_mem_dir, ENTRYPOINT_NAME)
        parts.append("")
        parts.append(_build_entrypoint_section("User-level", ep_path))

    if project_mem_dir:
        ep_path = os.path.join(project_mem_dir, ENTRYPOINT_NAME)
        parts.append("")
        parts.append(_build_entrypoint_section("Project-level", ep_path))

    return "\n".join(parts)


def _build_entrypoint_section(scope_label: str, entrypoint_path: str) -> str:
    """读取一个 MEMORY.md 文件并格式化为系统提示段。"""
    header = f"## {scope_label} {ENTRYPOINT_NAME} (`{entrypoint_path}`)\n"
    try:
        data = Path(entrypoint_path).read_text(encoding="utf-8")
        if data.strip():
            return header + "\n" + truncate_entrypoint_content(data)
    except OSError:
        pass
    return (
        header
        + f"\nThis {ENTRYPOINT_NAME} is currently empty. When you save new {scope_label.lower()}-level memories, add their pointers here."
    )


def _build_memory_lines(user_mem_dir: str, project_mem_dir: str) -> str:
    """构建类型化记忆的行为指令文本（不含 MEMORY.md 内容）。"""
    dir_exists_guidance = (
        "This directory already exists — write to it directly with the Write tool "
        "(do not run mkdir or check for its existence)."
    )

    parts = ["# auto memory\n"]
    parts.append(
        "You have a persistent, file-based memory system organized into two locations by content type:\n"
    )

    if user_mem_dir:
        parts.append(
            f"- **User-level** (`{user_mem_dir}`) — memories with `type: user` or `type: feedback`. "
            f"These follow you across all projects, because they describe the human or how the human likes to work. "
            f"{dir_exists_guidance}"
        )
    if project_mem_dir:
        parts.append(
            f"- **Project-level** (`{project_mem_dir}`) — memories with `type: project` or `type: reference`. "
            f"These belong to the current repo, can be committed for team sharing or git-ignored for personal use. "
            f"{dir_exists_guidance}"
        )

    parts.append(
        "\nThe `type` field in each memory file's frontmatter determines which directory it belongs to "
        "— pick the type first, then write to the matching directory."
    )
    parts.append(
        "\nYou should build up this memory system over time so that future conversations can have "
        "a complete picture of who the user is, how they'd like to collaborate with you, what behaviors "
        "to avoid or repeat, and the context behind the work the user gives you."
    )

    # frontmatter 格式示例
    parts.append("\n## How to save memories\n")
    parts.append(
        "Saving a memory is a two-step process:\n\n"
        "**Step 1** — write the memory to its own file (e.g., `user_role.md`, `feedback_testing.md`) "
        "using this frontmatter format:\n\n"
        "```markdown\n"
        "---\n"
        "name: {{memory name}}\n"
        "description: {{one-line description}}\n"
        "observed_at: {{YYYY-MM-DD, when the fact was observed — NOT when the file was edited}}\n"
        "source: {{optional: session id or origin}}\n"
        "metadata:\n"
        "  type: {{user, feedback, project, reference}}\n"
        "---\n\n"
        "{{memory content}}\n"
        "```\n\n"
        f"**Step 2** — add a pointer to that file in the `{ENTRYPOINT_NAME}` index in the SAME directory "
        f"as the memory file. `{ENTRYPOINT_NAME}` is an index, not a memory — each entry should be one line, "
        "under ~150 characters: `- [Title](file.md) — one-line hook`. It has no frontmatter. "
        f"Never write memory content directly into `{ENTRYPOINT_NAME}`.\n\n"
        f"- Both `{ENTRYPOINT_NAME}` files are always loaded into your conversation context"
        f" — lines after {MAX_ENTRYPOINT_LINES} each will be truncated, so keep each index concise\n"
        "- Keep the name, description, and type fields in memory files up-to-date with the content\n"
        "- Organize memory semantically by topic, not chronologically\n"
        "- Update or remove memories that turn out to be wrong or outdated\n"
        "- Do not write duplicate memories. First check if there is an existing memory you can update "
        "before writing a new one."
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# MemoryManager（对齐 Go 版 Manager）
# ---------------------------------------------------------------------------


class MemoryManager:
    """管理双路径自动记忆目录（用户级 + 项目级）。

    对齐 Go 版 memory.Manager：使用独立 .md 文件 + frontmatter + MEMORY.md 索引。
    实际的写入/读取通过 agent 的 Write/Read 工具完成（参考架构），
    此类提供系统提示构建和 /memory 斜杠命令支持。
    """

    def __init__(self, project_root: str) -> None:
        abs_root = os.path.abspath(project_root)
        self._project_root = abs_root
        # 用户级：~/.lumecode/memory/ — user/feedback 类型记忆
        self._user_mem_dir = get_user_auto_mem_path()
        # 项目级：<projectRoot>/.lumecode/memory/ — project/reference 类型记忆
        self._mem_dir = get_auto_mem_path(abs_root)
        self.extracted_message_count = 0

    @property
    def user_path(self) -> Path:
        """用户级 MEMORY.md 的路径（兼容旧接口）。"""
        if self._user_mem_dir:
            return Path(os.path.join(self._user_mem_dir, ENTRYPOINT_NAME))
        return Path.home() / ".lumecode" / "memory" / ENTRYPOINT_NAME

    @property
    def project_path(self) -> Path:
        """项目级 MEMORY.md 的路径（兼容旧接口）。"""
        return Path(os.path.join(self._mem_dir, ENTRYPOINT_NAME))

    @property
    def user_mem_dir(self) -> Path:
        """用户级记忆目录（~/.lumecode/memory/）。"""
        return (
            Path(self._user_mem_dir.rstrip(os.sep))
            if self._user_mem_dir
            else Path.home() / ".lumecode" / "memory"
        )

    @property
    def project_mem_dir(self) -> Path:
        """项目级记忆目录（<project>/.lumecode/memory/）。"""
        return Path(self._mem_dir.rstrip(os.sep))

    def load(self) -> str:
        """构建完整的记忆系统提示（对齐 Go 版 BuildSystemReminder）。

        确保两个目录存在后，返回包含行为指令和 MEMORY.md 索引内容的
        '# auto memory' 段，用于注入系统提示。
        """
        if not self._mem_dir and not self._user_mem_dir:
            return ""
        # 确保目录存在
        if self._user_mem_dir:
            ensure_memory_dir_exists(self._user_mem_dir)
        if self._mem_dir:
            ensure_memory_dir_exists(self._mem_dir)
        return build_memory_prompt(self._user_mem_dir, self._mem_dir)

    def load_all(self) -> list[MemoryFile]:
        """扫描两个目录中所有 .md 文件（排除 MEMORY.md），解析 frontmatter。

        对齐 Go 版 LoadAll：用户级文件在前，项目级在后。
        """
        result = _load_dir(self._user_mem_dir)
        result.extend(_load_dir(self._mem_dir))
        return result

    def get_memories(self) -> list[str]:
        """返回所有记忆文件的单行摘要，用于 /memory list。

        对齐 Go 版 GetMemories。
        """
        files = self.load_all()
        out: list[str] = []
        for f in files:
            type_tag = f.type if f.type else "?"
            desc = f.description if f.description else Path(f.path).name
            out.append(f"[{type_tag}] {f.name} — {desc}")
        return out

    def _scan_existing_memories(
        self,
        conversation_text: str = "",
        limit: int = RELATED_MEMORY_LIMIT,
    ) -> str:
        """挑出与本次对话最相关的已有记忆，作为去重上下文。

        不再把全部记忆的 manifest 塞进 prompt（记忆一多，固定的 prompt 成本
        高，模型在大列表里做 diff 也不可靠）。按 BM25 词法相关度取 top-k，
        再把正文前段截给模型判断"这条是否已经被记录"。
        """
        from lumecode.memory.textmatch import bm25_scores, tokenize

        docs: list[tuple[Path, str]] = []
        for dir_path in (self._user_mem_dir, self._mem_dir):
            if not dir_path:
                continue
            for f in iter_memory_files(dir_path):
                try:
                    docs.append((f, f.read_text(encoding="utf-8")))
                except OSError:
                    continue
        if not docs:
            return ""

        terms = set(tokenize(conversation_text))
        if terms:
            lexical = [
                {
                    "tokens": [t for t in tokenize(body) if t in terms],
                    "doc_len": 1,
                }
                for _f, body in docs
            ]
            scores = bm25_scores(conversation_text, lexical)
            ranked = sorted(zip(scores, docs, strict=True), key=lambda pair: pair[0], reverse=True)
            picked = [(f, body) for score, (f, body) in ranked if score > 0][:limit]
        else:
            picked = docs[:limit]

        entries: list[str] = []
        for f, body in picked:
            mf = parse_frontmatter(body)
            type_tag = mf.type or "?"
            desc = mf.description or f.stem
            snippet = _strip_frontmatter(body)[:RELATED_BODY_CHARS].strip()
            entry = f"- [{type_tag}] {f.name}: {desc}"
            if snippet:
                entry += f"\n  excerpt: {snippet}"
            entries.append(entry)
        return "\n".join(entries)

    async def extract(
        self,
        client: Any,
        conversation: ConversationManager,
        protocol: str,
        telemetry: Any | None = None,
        session_id: str = "",
    ) -> None:
        """触发记忆提取（对齐 Mem0 v3 的 ADD-only 抽取）。

        热路径只负责"把新事实加进来"：给模型看与本次对话相关的 top-k 已有
        记忆做去重参考，要求它只输出尚未被记录的新事实。合并、修正、删除
        交给后台 dream 整理（``consolidation.py``），两边职责不重叠。
        写入时按内容指纹去重；同名文件不会被静默覆盖，而是追加带日期的
        小节并保留旧版本快照。
        ``telemetry`` 仅用于记录这次“隐形” LLM 调用；为空时行为不变。
        """
        from lumecode.tools.base import StreamEnd, TextDelta

        recent = conversation.history[self.extracted_message_count :]
        if not recent:
            return

        conv_lines: list[str] = []
        for msg in recent:
            if msg.role == "user" and msg.content:
                conv_lines.append(f"[user]: {msg.content}")
            elif msg.role == "assistant" and msg.content:
                conv_lines.append(f"[assistant]: {msg.content}")
        if not conv_lines:
            return
        conversation_text = "\n".join(conv_lines)

        # 只取相关已有记忆做去重上下文（而不是全量 manifest）。
        manifest = self._scan_existing_memories(conversation_text)
        manifest_section = ""
        if manifest:
            manifest_section = (
                f"\n\n## Possibly related existing memories\n\n{manifest}\n\n"
                "These are for de-duplication reference only. Do NOT output a memory that "
                "duplicates one of these — including entries others may already cover."
            )

        prompt = (
            f"Analyze the conversation below and extract NEW durable memories worth saving.\n\n"
            f"For each memory, output in this exact format:\n"
            f"MEMORY_NAME: <kebab-case-name>\n"
            f"MEMORY_TYPE: <user|feedback|project|reference>\n"
            f"MEMORY_DESC: <one-line description>\n"
            f"MEMORY_BODY: <content>\n"
            f"---\n\n"
            f"Types:\n"
            f"- user/feedback → save to {self._user_mem_dir}\n"
            f"- project/reference → save to {self._mem_dir}\n\n"
            f"Rules:\n"
            f"- Only extract facts NOT already covered by the related memories above. "
            f"Do not rewrite or merge existing memories — only add what is new.\n"
            f"- Convert relative dates (yesterday, last week) to absolute dates "
            f"using today's date ({datetime.now(UTC).date().isoformat()}).\n"
            f"- If a related memory is contradicted by the conversation, still output the "
            f"corrected fact under a NEW name; background consolidation will reconcile it.\n\n"
            f"What NOT to save:\n"
            f"- Code patterns derivable from reading the project\n"
            f"- Git history, debugging solutions\n"
            f"- Ephemeral task details\n\n"
            f"If nothing is worth saving, output NONE.{manifest_section}\n\n"
            f"Conversation:\n{conversation_text}"
        )

        extract_conv = ConversationManager()
        extract_conv.history = [Message(role="user", content=prompt)]

        extract_span = None
        if telemetry is not None:
            model = str(getattr(client, "model", "") or "") or None
            extract_span = telemetry.start_span(
                SpanKind.MEMORY_EXTRACT,
                "memory-extract",
                model=model,
                input=prompt,
                set_current=False,
            )

        collected = ""
        usage: Usage | None = None
        try:
            async for event in client.stream(
                extract_conv, system="You are a memory extraction assistant."
            ):
                if isinstance(event, TextDelta):
                    collected += event.text
                elif isinstance(event, StreamEnd):
                    usage = Usage(
                        input_tokens=event.input_tokens,
                        output_tokens=event.output_tokens,
                        cache_read_tokens=event.cache_read,
                        cache_creation_tokens=event.cache_creation,
                    )
        except Exception as e:
            if extract_span is not None:
                extract_span.end(status=SpanStatus.ERROR, error=str(e))
            log.debug("memory extraction failed: %s", e)
            return

        self.extracted_message_count = len(conversation.history)

        # 解析并写入记忆文件
        if not collected or collected.strip() == "NONE" or "MEMORY_NAME:" not in collected:
            if extract_span is not None:
                extract_span.end(
                    usage=usage,
                    output=collected,
                    attributes={"memory.files_written": 0},
                )
            return

        written = 0
        blocks = [b for b in collected.split("---") if "MEMORY_NAME:" in b]
        now_ms = int(time.time() * 1000)
        for block in blocks:
            name = _extract_field(block, "MEMORY_NAME")
            mtype = _extract_field(block, "MEMORY_TYPE") or "reference"
            desc = _extract_field(block, "MEMORY_DESC")
            body = _extract_field(block, "MEMORY_BODY")
            if not name or not body:
                continue
            if mtype not in VALID_TYPES:
                mtype = "reference"

            # 路由到正确的目录
            target_dir = self._user_mem_dir if mtype in _USER_LEVEL_TYPES else self._mem_dir
            if not target_dir:
                continue
            ensure_memory_dir_exists(target_dir)

            safe_name = _sanitize_name(name)
            if not safe_name:
                continue
            file_path = Path(target_dir) / f"{safe_name}.md"

            if _append_memory(target_dir, file_path, mtype, desc, body, now_ms, session_id):
                written += 1

        if extract_span is not None:
            extract_span.end(
                usage=usage,
                output=collected,
                attributes={"memory.files_written": written},
            )

    def clear(self) -> None:
        """清除两个目录中所有 .md 文件（对齐 Go 版 Clear）。"""
        _clear_dir(self._user_mem_dir)
        _clear_dir(self._mem_dir)

    def get_display_text(self) -> str:
        """返回记忆摘要文本（/memory 命令显示）。"""
        memories = self.get_memories()
        if not memories:
            return "当前没有任何自动记忆。"

        parts: list[str] = []
        parts.append("记忆目录：")
        parts.append(f"  用户级: {self._user_mem_dir}")
        parts.append(f"  项目级: {self._mem_dir}")
        parts.append("")
        for line in memories:
            parts.append(f"  {line}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------


def _load_dir(dir_path: str) -> list[MemoryFile]:
    """扫描目录中的 .md 文件，解析 frontmatter 并返回 MemoryFile 列表。"""
    if not dir_path:
        return []

    result: list[MemoryFile] = []
    for entry in iter_memory_files(dir_path):
        try:
            data = entry.read_text(encoding="utf-8")
        except OSError:
            continue
        mf = parse_frontmatter(data)
        mf.path = str(entry)
        if not mf.name:
            mf.name = entry.stem  # 去掉 .md 后缀
        result.append(mf)
    return result


def _strip_frontmatter(content: str) -> str:
    m = _FRONTMATTER_RE.match(content)
    return content[m.end() :] if m else content


def _extract_field(block: str, field: str) -> str:
    """从记忆提取输出的一个 block 中提取指定字段值。"""
    m = re.search(rf"{field}:\s*(.+?)(?:\n|$)", block)
    return m.group(1).strip() if m else ""


def _sanitize_name(name: str) -> str:
    """记忆名 → 安全的文件名 stem（防止 `../x` 这类路径逃逸）。"""
    return _UNSAFE_NAME_RE.sub("-", name.strip().lower()).strip("-.")


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _memory_timestamp(now_ms: int, session_id: str) -> str:
    """追加小节/快照用的时间戳：人类可读的绝对时间 + 来源会话。"""
    stamp = format_observed_at(now_ms).replace("T", " ").replace("Z", " UTC")
    return f"{stamp} (session {session_id})" if session_id else stamp


def _append_memory(
    target_dir: str,
    file_path: Path,
    mtype: str,
    desc: str,
    body: str,
    now_ms: int,
    session_id: str,
) -> bool:
    """ADD-only 写入一条记忆；返回是否真的写入了内容。

    - 内容已经出现在文件里（逐字，忽略空白差异）→ 跳过；
    - 文件不存在 / 是旧格式（没有 ``observed_at``）→ 写入完整新文件；
    - 同名文件已存在 → 先快照，再追加带日期与来源会话的小节，
      **绝不覆盖**已有内容。
    """
    observed = format_observed_at(now_ms)
    norm_body = _normalize_text(body)
    if not norm_body:
        return False

    try:
        existing = file_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        existing = ""
    except OSError:
        return False

    source = f"source: session {session_id}" if session_id else None

    if existing:
        if norm_body in _normalize_text(existing):
            return False

        old_fm = parse_frontmatter(existing)
        snapshot_file(target_dir, file_path)
        if old_fm.description and old_fm.description not in desc:
            desc = f"{desc}; {old_fm.description}" if desc else old_fm.description

        content = (
            existing.rstrip()
            + f"\n\n## Update — {_memory_timestamp(now_ms, session_id)}\n\n{body}\n"
        )
    else:
        header = [
            "---",
            f"name: {file_path.stem}",
            f"description: {desc}",
            f"observed_at: {observed}",
        ]
        if source:
            header.append(source)
        header.append(f"metadata:\n  type: {mtype}")
        content = "\n".join(header) + f"\n---\n\n{body}\n"

    try:
        file_path.write_text(content, encoding="utf-8")
    except OSError:
        return False

    _update_entrypoint(target_dir, file_path.name, desc)
    return True


def _update_entrypoint(target_dir: str, filename: str, desc: str) -> None:
    """确保 MEMORY.md 索引里有一行指向该记忆文件。"""
    idx_path = Path(target_dir) / ENTRYPOINT_NAME
    title = filename.removesuffix(".md")
    idx_line = f"- [{title}]({filename}) — {desc}\n"
    try:
        existing = idx_path.read_text(encoding="utf-8") if idx_path.exists() else ""
        if filename not in existing:
            idx_path.write_text(existing + idx_line, encoding="utf-8")
    except OSError:
        pass


def _clear_dir(dir_path: str) -> None:
    """删除目录中所有 .md 文件（包括 MEMORY.md）及历史快照。"""
    if not dir_path:
        return
    from lumecode.memory.history import clear_history

    for entry in iter_memory_files(dir_path, include_index=True):
        with contextlib.suppress(OSError):
            entry.unlink()
    clear_history(dir_path)
