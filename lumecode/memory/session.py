from __future__ import annotations

import json
import logging
import random
import re
import string
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import IO, Any

from lumecode.conversation import (
    ConversationManager,
    Message,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)

log = logging.getLogger(__name__)

SESSIONS_DIR = ".lumecode/sessions"
DEFAULT_MAX_AGE_DAYS = 30
TITLE_MAX_LENGTH = 50

# 会话文件头。写入第一行，读取时不要求存在（v0 旧文件没有头）。
SCHEMA_VERSION = 1
HEADER_TYPE = "_meta"

# 每次 /fork、/compact 后重新注入的派生上下文；恢复时若重新注入成功，
# 就丢弃从磁盘重放出来的旧副本，避免同一段上下文在历史里出现两次。
_INJECTED_CONTEXT_PREFIXES = ("Current working directory:",)

SESSION_SUMMARY_PROMPT = (
    "你是一个对话摘要助手。请根据下面的对话内容，用一句话总结这个会话的主要内容。"
    "只输出摘要文本，不要加任何前缀或标点符号外的修饰。不要调用任何工具。"
)


# ---------------------------------------------------------------------------
# RecordType & SessionRecord
# ---------------------------------------------------------------------------


class RecordType(str, Enum):
    SYSTEM_PROMPT = "system_prompt"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_RESULT = "tool_result"
    COMPRESSION = "compression"
    # Layer-2 compact 标记。auto_compact 压缩对话记录时写入。
    # 内容为结构化载荷（参见 make_compact_boundary / parse_compact_boundary），
    # 包含摘要文本和原样保留的 keep 尾部（以序列化 record 形式内联），
    # 使 resume 可以仅凭此标记重建压缩后的状态，无需重放标记之前的原始前缀。
    COMPACT_BOUNDARY = "compact_boundary"
    # /rewind 标记。只记录"回退到第几条消息"，不改写已有行；
    # 重放时应用，保证内存回退和磁盘重放结果一致。
    ROLLBACK = "rollback"


@dataclass
class SessionRecord:
    type: RecordType
    content: Any
    timestamp: datetime
    tool_use_id: str | None = None
    is_error: bool = False
    # 结构性标记（compact_boundary / rollback）不更新 message_count 与标题。
    structural: bool = False
    # thinking 随所属 assistant 记录一起落盘（Gemini CLI 的做法），
    # 避免为思考内容单开一种 record 类型，也保持"一条消息一行"的形态。
    thinking: list[dict[str, str]] = field(default_factory=list)
    ordinal: int = 0
    turn_id: str = ""

    def to_jsonl(self) -> str:
        data: dict[str, Any] = {
            "type": self.type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.tool_use_id is not None:
            data["tool_use_id"] = self.tool_use_id
        if self.type == RecordType.TOOL_RESULT:
            data["is_error"] = self.is_error
        if self.thinking:
            data["thinking"] = self.thinking
        if self.ordinal:
            data["ordinal"] = self.ordinal
        if self.turn_id:
            data["turn_id"] = self.turn_id
        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def from_jsonl(cls, line: str) -> SessionRecord | None:
        try:
            data = json.loads(line)
            return cls(
                type=RecordType(data["type"]),
                content=data["content"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
                tool_use_id=data.get("tool_use_id"),
                is_error=data.get("is_error", False),
                thinking=list(data.get("thinking") or []),
                ordinal=int(data.get("ordinal") or 0),
                turn_id=data.get("turn_id") or "",
            )
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            return None

    @classmethod
    def from_message(cls, message: Message) -> list[SessionRecord]:
        now = datetime.now(UTC)
        records: list[SessionRecord] = []

        if message.tool_results:
            for tr in message.tool_results:
                records.append(
                    cls(
                        type=RecordType.TOOL_RESULT,
                        content=tr.content,
                        timestamp=now,
                        tool_use_id=tr.tool_use_id,
                        is_error=tr.is_error,
                    )
                )
        elif message.role == "assistant":
            thinking = _thinking_to_dicts(message.thinking_blocks)
            if message.tool_uses:
                content_blocks: list[dict[str, Any]] = []
                if message.content or thinking:
                    content_blocks.append({"type": "text", "text": message.content})
                for tu in message.tool_uses:
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tu.tool_use_id,
                            "name": tu.tool_name,
                            "input": tu.arguments,
                        }
                    )
                records.append(
                    cls(
                        type=RecordType.ASSISTANT,
                        content=content_blocks,
                        timestamp=now,
                        thinking=thinking,
                    )
                )
            else:
                records.append(
                    cls(
                        type=RecordType.ASSISTANT,
                        content=message.content,
                        timestamp=now,
                        thinking=thinking,
                    )
                )
        else:
            records.append(cls(type=RecordType.USER, content=message.content, timestamp=now))

        return records


def _thinking_to_dicts(blocks: list[ThinkingBlock]) -> list[dict[str, str]]:
    return [
        {"thinking": tb.thinking, "signature": tb.signature}
        for tb in blocks
        if tb.thinking or tb.signature
    ]


# ---------------------------------------------------------------------------
# Compact boundary 载荷（摘要 + 内联的 keep 尾部）
# ---------------------------------------------------------------------------


def _message_to_record_dicts(message: Message) -> list[dict[str, Any]]:
    """将单条 Message 序列化为与磁盘存储格式一致的 record-dict 列表。

    复用 SessionRecord.from_message，使内联的 keep 尾部与正常追加消息的持久化
    结果逐字节一致（assistant 的 tool_uses 变为 content-blocks 列表，每个
    tool_result 独立成一条 record）。这保证了 tool_use↔tool_result 配对的
    无损往返——不像纯 role+content 文本导出那样会丢失 tool call 的关联关系。
    """
    dicts: list[dict[str, Any]] = []
    for rec in SessionRecord.from_message(message):
        data: dict[str, Any] = {"type": rec.type.value, "content": rec.content}
        if rec.tool_use_id is not None:
            data["tool_use_id"] = rec.tool_use_id
        if rec.type == RecordType.TOOL_RESULT:
            data["is_error"] = rec.is_error
        if rec.thinking:
            data["thinking"] = rec.thinking
        dicts.append(data)
    return dicts


def make_compact_boundary(
    summary: str,
    keep: list[Message],
    keep_includes_summary: bool = False,
) -> SessionRecord:
    """构建一条 COMPACT_BOUNDARY record，内联摘要和原样保留的 keep 尾部。

    `keep` 是 auto_compact 原样保留的近期尾部消息。将其存储在 boundary record
    内部（而不是依赖它在文件中的物理位置），意味着 resume 可以仅凭 boundary
    重建压缩后的状态——boundary 之前的原始前缀保留在磁盘上但不会被重放。

    keep_includes_summary=True 表示 `keep` 的第一条就是压缩后历史里的摘要消息
    （可能携带恢复附件）。此时重放直接展开 keep，不再生成摘要文本，保证恢复
    出的历史与压缩后的内存历史逐条一致。旧的 boundary 没有这个标记，重放时
    仍按"生成摘要消息 + keep"处理（向后兼容）。
    """
    keep_dicts: list[dict[str, Any]] = []
    for msg in keep:
        keep_dicts.extend(_message_to_record_dicts(msg))
    payload: dict[str, Any] = {"summary": summary, "keep": keep_dicts}
    if keep_includes_summary:
        payload["keep_includes_summary"] = True
    return SessionRecord(
        type=RecordType.COMPACT_BOUNDARY,
        content=payload,
        timestamp=datetime.now(UTC),
        structural=True,
    )


def make_rollback(message_index: int, reason: str = "rewind") -> SessionRecord:
    """构建一条 ROLLBACK 标记：重放时把历史截断到 `message_index` 条消息。

    与"原地删除记录"不同，标记保留在文件里，因此：
    - resume 重放出的历史与内存里回退后的历史一致（回退不会因重启而复活）
    - 回退本身可审计
    """
    return SessionRecord(
        type=RecordType.ROLLBACK,
        content={"message_index": max(0, int(message_index)), "reason": reason},
        timestamp=datetime.now(UTC),
        structural=True,
    )


def parse_compact_boundary(record: SessionRecord) -> tuple[str, list[Message]]:
    """make_compact_boundary 的逆操作：返回 (summary, keep_messages)。

    对遗留或格式异常的 payload 降级返回 ("", [])，确保单条损坏的 boundary
    不会导致 resume 崩溃。
    """
    content = record.content
    if not isinstance(content, dict):
        return "", []
    summary = content.get("summary", "")
    keep_raw = content.get("keep", [])
    keep_records: list[SessionRecord] = []
    for item in keep_raw if isinstance(keep_raw, list) else []:
        if not isinstance(item, dict) or "type" not in item:
            continue
        try:
            keep_records.append(
                SessionRecord(
                    type=RecordType(item["type"]),
                    content=item.get("content"),
                    timestamp=record.timestamp,
                    tool_use_id=item.get("tool_use_id"),
                    is_error=item.get("is_error", False),
                    thinking=list(item.get("thinking") or []),
                )
            )
        except ValueError:
            continue
    return summary, records_to_messages(keep_records)


# ---------------------------------------------------------------------------
# Record ↔ Message 转换
# ---------------------------------------------------------------------------


def records_to_messages(records: list[SessionRecord]) -> list[Message]:
    messages: list[Message] = []
    pending_tool_results: list[ToolResultBlock] = []

    for record in records:
        if record.type == RecordType.TOOL_RESULT:
            pending_tool_results.append(
                ToolResultBlock(
                    tool_use_id=record.tool_use_id or "",
                    content=(
                        record.content
                        if isinstance(record.content, str)
                        else json.dumps(record.content)
                    ),
                    is_error=record.is_error,
                )
            )
            continue

        if pending_tool_results:
            messages.append(Message(role="user", content="", tool_results=pending_tool_results))
            pending_tool_results = []

        if record.type == RecordType.SYSTEM_PROMPT:
            continue

        if record.type == RecordType.ROLLBACK:
            if isinstance(record.content, dict):
                limit = int(record.content.get("message_index", len(messages)) or 0)
            else:
                limit = len(messages)
            del messages[max(0, limit) :]
            continue

        if record.type == RecordType.COMPRESSION:
            messages.append(
                Message(
                    role="user",
                    content="本次会话延续自之前的对话，因上下文空间不足进行了压缩。以下是早期对话的摘要：\n\n"
                    + (record.content or ""),
                )
            )
            continue

        if record.type == RecordType.COMPACT_BOUNDARY:
            # 内联展开：摘要作为 user 消息，后接原样保留的 keep 尾部。
            # resume() 通常已预裁剪到最后一个 boundary，所以这里只会处理
            # 权威的那一条；但在此展开可以保证 records_to_messages 对任何
            # 直接调用者都保持自洽。
            summary, keep_messages = parse_compact_boundary(record)
            includes_summary = isinstance(record.content, dict) and bool(
                record.content.get("keep_includes_summary")
            )
            if includes_summary:
                messages.extend(keep_messages)
                continue
            messages.append(
                Message(
                    role="user",
                    content="本次会话延续自之前的对话，因上下文空间不足进行了压缩。以下是早期对话的摘要：\n\n"
                    + summary,
                )
            )
            messages.extend(keep_messages)
            continue

        if record.type == RecordType.USER:
            messages.append(Message(role="user", content=record.content or ""))
        elif record.type == RecordType.ASSISTANT:
            if isinstance(record.content, list):
                text = ""
                tool_uses: list[ToolUseBlock] = []
                for block in record.content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        text += block.get("text", "")
                    elif block.get("type") == "tool_use":
                        tool_uses.append(
                            ToolUseBlock(
                                tool_use_id=block.get("id", ""),
                                tool_name=block.get("name", ""),
                                arguments=block.get("input", {}),
                            )
                        )
                messages.append(
                    Message(
                        role="assistant",
                        content=text,
                        tool_uses=tool_uses,
                        thinking_blocks=_thinking_from_dicts(record.thinking),
                    )
                )
            else:
                messages.append(
                    Message(
                        role="assistant",
                        content=record.content or "",
                        thinking_blocks=_thinking_from_dicts(record.thinking),
                    )
                )

    if pending_tool_results:
        messages.append(Message(role="user", content="", tool_results=pending_tool_results))

    return messages


def _thinking_from_dicts(raw: list[dict[str, str]]) -> list[ThinkingBlock]:
    out: list[ThinkingBlock] = []
    for item in raw or []:
        if isinstance(item, dict):
            out.append(
                ThinkingBlock(
                    thinking=item.get("thinking", ""),
                    signature=item.get("signature", ""),
                )
            )
    return out


# ---------------------------------------------------------------------------
# 消息链校验
# ---------------------------------------------------------------------------


def validate_message_chain(records: list[SessionRecord]) -> int:
    last_valid = 0
    pending_tool_uses: set[str] = set()

    for i, record in enumerate(records):
        if record.type == RecordType.ASSISTANT and isinstance(record.content, list):
            for block in record.content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_id = block.get("id", "")
                    if tool_id:
                        pending_tool_uses.add(tool_id)

        if record.type == RecordType.TOOL_RESULT and record.tool_use_id:
            pending_tool_uses.discard(record.tool_use_id)

        if record.type in (RecordType.COMPACT_BOUNDARY, RecordType.ROLLBACK):
            # 结构性标记自带完整语义（boundary 内联 keep 尾部，rollback 截断），
            # 它们不是"等待配对的 tool_use"，遇到时视为一个完整边界。
            pending_tool_uses.clear()

        if not pending_tool_uses:
            last_valid = i + 1

    return last_valid


# ---------------------------------------------------------------------------
# SessionMeta
# ---------------------------------------------------------------------------


@dataclass
class SessionMeta:
    id: str
    title: str = ""
    summary: str = ""
    message_count: int = 0
    total_tokens: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_active: datetime = field(default_factory=lambda: datetime.now(UTC))

    def save(self, path: Path) -> None:
        data = {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "message_count": self.message_count,
            "total_tokens": self.total_tokens,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> SessionMeta | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                id=data["id"],
                title=data.get("title", ""),
                summary=data.get("summary", ""),
                message_count=data.get("message_count", 0),
                total_tokens=data.get("total_tokens", 0),
                created_at=datetime.fromisoformat(data["created_at"]),
                last_active=datetime.fromisoformat(data["last_active"]),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return None


# ---------------------------------------------------------------------------
# Session（活跃会话句柄）
# ---------------------------------------------------------------------------


def _scan_next_ordinal(file: IO[str]) -> int:
    """Resume 场景下从已有文件推出下一个可用序号。

    只在打开会话时读一次（会话文件量级很小）；找不到 ordinal 的旧文件
    从 1 开始编号，保持单调即可。
    """
    if not (file.mode and "r" in file.mode):
        return 1
    try:
        file.seek(0)
        last = 0
        for line in file:
            line = line.strip()
            if not line or HEADER_TYPE in line[:16]:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            ordinal = data.get("ordinal")
            if isinstance(ordinal, int) and ordinal > last:
                last = ordinal
        file.seek(0, 2)
        return last + 1
    except (OSError, ValueError):
        return 1


def _is_injected_context(message: Message) -> bool:
    """识别每次启动/压缩后重新注入的派生上下文（环境信息、长期记忆）。

    这类内容由程序生成并在注入时带上"当前时间"等易变信息，落盘会在恢复时
    产生重复或过期的副本。写穿钩子安装后新会话不会再记录它们；这里负责
    清掉旧文件里已经存在的副本。
    """
    if message.role != "user" or message.tool_results:
        return False
    content = message.content or ""
    if content.startswith(_INJECTED_CONTEXT_PREFIXES):
        return True
    return "<system-reminder>" in content and (
        "As you answer the user's questions" in content or "# lumecodeMd" in content
    )


def strip_persisted_injected_context(messages: list[Message]) -> list[Message]:
    return [m for m in messages if not _is_injected_context(m)]


class Session:
    def __init__(
        self,
        session_id: str,
        file: IO[str],
        meta: SessionMeta,
        sessions_dir: Path,
    ) -> None:
        self.session_id = session_id
        self._file = file
        self.meta = meta
        self._sessions_dir = sessions_dir
        # 序号与轮次号由 Session 统一分配，保证同一次运行内单调递增，
        # 也保证从磁盘恢复后继续追加的序号不会和已有行冲突。
        self._next_ordinal = _scan_next_ordinal(file)
        self.turn_id = ""

    def begin_turn(self, turn_id: str) -> None:
        """标记后续追加的记录属于哪一轮（Agent 在每轮开始时调用）。"""
        self.turn_id = turn_id

    def _stamp(self, record: SessionRecord) -> None:
        record.ordinal = self._next_ordinal
        self._next_ordinal += 1
        if not record.turn_id:
            record.turn_id = self.turn_id

    def append(self, message: Message) -> None:
        records = SessionRecord.from_message(message)
        for record in records:
            self._stamp(record)
            self._file.write(record.to_jsonl() + "\n")
        self._file.flush()

        self.meta.message_count += 1
        self.meta.last_active = datetime.now(UTC)

        if not self.meta.title and message.role == "user" and message.content:
            self.meta.title = message.content[:TITLE_MAX_LENGTH]

        self.meta.save(self._sessions_dir / f"{self.session_id}.meta")

    def append_record(self, record: SessionRecord) -> None:
        """追加一条原始 SessionRecord（例如 compact_boundary 标记）。

        与 append() 不同，此方法不会更新 message_count/title——boundary 是
        结构性标记而非对话轮次。last_active 仍会更新，以保证 session 按最近
        使用排序。
        """
        self._stamp(record)
        self._file.write(record.to_jsonl() + "\n")
        self._file.flush()
        self.meta.last_active = datetime.now(UTC)
        self.meta.save(self._sessions_dir / f"{self.session_id}.meta")

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()

    # ------------------------------------------------------------------
    # ConversationManager 写穿钩子
    # ------------------------------------------------------------------

    def on_message(self, message: Message, kind: str = "message") -> None:
        """ConversationManager 的 append sink：内存追加的同时立即落盘。"""
        self.append(message)

    def on_replace(self, replacement: Any) -> None:
        """ConversationManager 的 replace sink：把历史替换写成结构性标记。

        - compact：boundary 内联的尾部就是内存里重建后的历史（摘要消息 + 保留尾部），
          因此 resume 重放出的 history 与压缩后的内存 history 逐条一致。
        - rollback：只记截断位置，不改写已有行。
        """
        kind = getattr(replacement, "kind", "other")
        index = int(getattr(replacement, "message_index", 0) or 0)
        if kind == "compact":
            summary = getattr(replacement, "summary", "") or ""
            summary_message = getattr(replacement, "summary_message", None)
            keep = list(getattr(replacement, "keep", []))
            tail = ([summary_message] if summary_message is not None else []) + keep
            if summary:
                self.append_record(
                    make_compact_boundary(
                        summary,
                        tail,
                        keep_includes_summary=summary_message is not None,
                    )
                )
                return
        self.append_record(make_rollback(index, reason=kind if kind != "other" else "replace"))

    # ------------------------------------------------------------------
    # 结构性标记
    # ------------------------------------------------------------------

    def mark_rollback(self, message_index: int, reason: str = "rewind") -> None:
        self.append_record(make_rollback(message_index, reason))

    def mark_compaction(self, summary: str, keep: list[Message]) -> None:
        if summary:
            self.append_record(make_compact_boundary(summary, keep))


# ---------------------------------------------------------------------------
# ResumeResult
# ---------------------------------------------------------------------------


@dataclass
class ResumeResult:
    session: Session
    messages: list[Message]
    last_active: datetime


# ---------------------------------------------------------------------------
# Session 摘要生成
# ---------------------------------------------------------------------------


async def generate_session_summary(
    client: Any,
    conversation: ConversationManager,
    protocol: str,
    telemetry: Any | None = None,
) -> str:
    from lumecode.tools.base import StreamEnd, TextDelta

    recent = conversation.history[-10:]
    if not recent:
        return ""

    summary_conv = ConversationManager()
    summary_conv.history = [Message(role="user", content=SESSION_SUMMARY_PROMPT)]
    for msg in recent:
        summary_conv.history.append(msg)
    summary_conv.history.append(
        Message(role="user", content="请用一句话总结上面的对话内容。不要调用工具。")
    )

    summary_span = None
    if telemetry is not None:
        from lumecode.telemetry import SpanKind

        summary_span = telemetry.start_span(
            SpanKind.SESSION_SUMMARY,
            "session-summary",
            model=getattr(client, "model", None),
            set_current=False,
        )

    collected = ""
    usage = None
    try:
        async for event in client.stream(summary_conv, system=SESSION_SUMMARY_PROMPT):
            if isinstance(event, TextDelta):
                collected += event.text
            elif isinstance(event, StreamEnd):
                from lumecode.telemetry import Usage

                usage = Usage(
                    input_tokens=event.input_tokens,
                    output_tokens=event.output_tokens,
                    cache_read_tokens=event.cache_read,
                    cache_creation_tokens=event.cache_creation,
                )
    except Exception as e:
        if summary_span is not None:
            summary_span.end(status="error", error=str(e))
        log.debug("session summary generation failed: %s", e)
        return ""

    if summary_span is not None:
        summary_span.end(usage=usage, output=collected)

    return collected.strip()


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------


def _generate_session_id() -> str:
    now = datetime.now()
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"session_{now.strftime('%Y%m%d_%H%M%S')}_{suffix}"


# 合法的 session id 只允许字母/数字/下划线，且必须以 "session_" 开头。
# resume/delete 会用用户传入的 id 直接拼路径，不校验的话
# `/session delete ../../foo` 就能删掉任意文件。
_SESSION_ID_RE = re.compile(r"^session_[A-Za-z0-9_]+$")


def _is_valid_session_id(session_id: str) -> bool:
    return bool(session_id) and _SESSION_ID_RE.match(session_id) is not None


class SessionManager:
    def __init__(self, work_dir: str) -> None:
        self._sessions_dir = Path(work_dir) / SESSIONS_DIR
        self._sessions_dir.mkdir(parents=True, exist_ok=True)

    def create(self) -> Session:
        session_id = _generate_session_id()
        jsonl_path = self._sessions_dir / f"{session_id}.jsonl"
        meta = SessionMeta(id=session_id)
        meta.save(self._sessions_dir / f"{session_id}.meta")

        file = open(jsonl_path, "a", encoding="utf-8")  # noqa: SIM115
        # 新文件写一行版本头；旧文件没有头，读取时按 v0 处理（向后兼容，
        # 不在 resume 时改写用户已有文件）。
        header = json.dumps(
            {"type": HEADER_TYPE, "version": SCHEMA_VERSION},
            ensure_ascii=False,
        )
        file.write(header + "\n")
        file.flush()
        return Session(
            session_id=session_id,
            file=file,
            meta=meta,
            sessions_dir=self._sessions_dir,
        )

    def list(self) -> list[SessionMeta]:
        metas: list[SessionMeta] = []
        for meta_path in self._sessions_dir.glob("*.meta"):
            meta = SessionMeta.load(meta_path)
            if meta is not None:
                metas.append(meta)
        metas.sort(key=lambda m: m.last_active, reverse=True)
        return metas

    def resume(self, session_id: str) -> ResumeResult | None:
        if not _is_valid_session_id(session_id):
            return None
        jsonl_path = self._sessions_dir / f"{session_id}.jsonl"
        meta_path = self._sessions_dir / f"{session_id}.meta"

        if not jsonl_path.exists():
            return None

        meta = SessionMeta.load(meta_path)
        if meta is None:
            return None

        records: list[SessionRecord] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = SessionRecord.from_jsonl(line)
                if record is not None:
                    records.append(record)

        # 重建压缩后的状态：仅从最后一个 compact_boundary 开始重放。
        # 该标记之前的 record 是已被摘要过的原始前缀——保留在磁盘上供审计，
        # 但不再重放。标记本身内联了摘要 + 原样 keep 尾部，标记之后追加的
        # 普通消息（续写）照常重放。没有 boundary 则全量重放（兼容旧 session）。
        last_boundary = -1
        for i, rec in enumerate(records):
            if rec.type == RecordType.COMPACT_BOUNDARY:
                last_boundary = i
        if last_boundary >= 0:
            records = records[last_boundary:]

        valid_count = validate_message_chain(records)
        records = records[:valid_count]
        messages = records_to_messages(records)
        # 旧文件里可能混有环境信息/记忆注入副本；恢复流程会重新注入，先剔除。
        messages = strip_persisted_injected_context(messages)

        # 打开为 r+ 而不是 a：_scan_next_ordinal 需要从头读出最后一个序号，
        # 读完再 seek 回文件末尾继续追加。
        file = open(jsonl_path, "r+", encoding="utf-8")  # noqa: SIM115
        session = Session(
            session_id=session_id,
            file=file,
            meta=meta,
            sessions_dir=self._sessions_dir,
        )

        return ResumeResult(
            session=session,
            messages=messages,
            last_active=meta.last_active,
        )

    def delete(self, session_id: str) -> bool:
        if not _is_valid_session_id(session_id):
            return False
        jsonl_path = self._sessions_dir / f"{session_id}.jsonl"
        meta_path = self._sessions_dir / f"{session_id}.meta"

        deleted = False
        if jsonl_path.exists():
            jsonl_path.unlink()
            deleted = True
        if meta_path.exists():
            meta_path.unlink()
            deleted = True
        return deleted

    def cleanup(self, max_age_days: int = DEFAULT_MAX_AGE_DAYS) -> int:
        cutoff = datetime.now(UTC) - timedelta(days=max_age_days)
        removed = 0

        for meta_path in list(self._sessions_dir.glob("*.meta")):
            meta = SessionMeta.load(meta_path)
            if meta is not None and meta.last_active < cutoff:
                self.delete(meta.id)
                removed += 1

        return removed
