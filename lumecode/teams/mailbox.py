from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from lumecode.fileutil import atomic_write_text, file_lock


@dataclass
class MailboxMessage:
    id: str
    from_agent: str
    to_agent: str
    content: str
    summary: str = ""
    message_type: str = "text"  # text | shutdown_request | shutdown_response
    timestamp: float = 0.0
    read: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MailboxMessage:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class Mailbox:
    """Single-file mailbox with file locking, one JSON array per agent.

    Each agent's inbox is stored as ``{agent_id}.json`` under *base_dir*.
    A companion ``.lock`` file is used for mutual exclusion (matching the
    Go/Java/TS implementation).
    """

    def __init__(self, base_dir: str | Path) -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    # ── path helpers ─────────────────────────────────────────────

    def _inbox_path(self, agent_id: str) -> Path:
        return self._base_dir / f"{agent_id}.json"

    def _lock_path(self, agent_id: str) -> Path:
        return self._base_dir / f"{agent_id}.json.lock"

    # ── file lock ────────────────────────────────────────────────

    def _with_lock(
        self,
        agent_id: str,
        fn: Callable[[list[MailboxMessage]], list[MailboxMessage]],
    ) -> Any:
        """Acquire a file lock, read inbox, apply *fn* mutation, write back."""
        with file_lock(self._lock_path(agent_id)):
            messages = self._read_inbox(agent_id)
            messages = fn(messages)
            self._write_inbox(agent_id, messages)

    # ── inbox I/O ────────────────────────────────────────────────

    def _read_inbox(self, agent_id: str) -> list[MailboxMessage]:
        path = self._inbox_path(agent_id)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return [MailboxMessage.from_dict(item) for item in data]
        except (json.JSONDecodeError, KeyError, TypeError):
            return []

    def _write_inbox(self, agent_id: str, messages: list[MailboxMessage]) -> None:
        path = self._inbox_path(agent_id)
        data = json.dumps(
            [m.to_dict() for m in messages],
            ensure_ascii=False,
            indent=2,
        )
        # 原子替换：并发读者要么读到旧的完整内容，要么读到新的完整内容，
        # 绝不会读到写了一半的 JSON（旧实现直接 write_text 截断重写，
        # 无锁的 read() 会因此解析失败并静默返回空列表，即丢消息）。
        atomic_write_text(path, data)

    # ── public API ───────────────────────────────────────────────

    def write(self, agent_id: str, message: MailboxMessage) -> None:
        """Append a message to *agent_id*'s inbox (thread-safe)."""

        def _append(msgs: list[MailboxMessage]) -> list[MailboxMessage]:
            message.read = False
            if message.timestamp == 0.0:
                message.timestamp = time.time()
            msgs.append(message)
            return msgs

        self._with_lock(agent_id, _append)

    def read(self, agent_id: str) -> list[MailboxMessage]:
        """Return all unread messages without marking them as read."""
        messages = self._read_inbox(agent_id)
        return [m for m in messages if not m.read]

    def consume(self, agent_id: str) -> list[MailboxMessage]:
        """Return all unread messages and mark them as read (thread-safe)."""
        result: list[MailboxMessage] = []

        def _mark_read(msgs: list[MailboxMessage]) -> list[MailboxMessage]:
            for m in msgs:
                if not m.read:
                    result.append(m)
                    m.read = True
            return msgs

        self._with_lock(agent_id, _mark_read)
        return result

    def broadcast(
        self,
        team_members: list[str],
        message: MailboxMessage,
        exclude: str = "",
    ) -> None:
        for agent_id in team_members:
            if agent_id == exclude:
                continue
            self.write(agent_id, message)

    def cleanup(self, agent_id: str) -> None:
        """Remove an agent's inbox file."""
        self._inbox_path(agent_id).unlink(missing_ok=True)
        self._lock_path(agent_id).unlink(missing_ok=True)

    def cleanup_all(self) -> None:
        """Remove all inbox files."""
        if not self._base_dir.exists():
            return
        for f in self._base_dir.iterdir():
            f.unlink(missing_ok=True)


def create_message(
    from_agent: str,
    to_agent: str,
    content: str,
    summary: str = "",
    message_type: str = "text",
    metadata: dict[str, Any] | None = None,
) -> MailboxMessage:
    return MailboxMessage(
        id=uuid.uuid4().hex[:12],
        from_agent=from_agent,
        to_agent=to_agent,
        content=content,
        summary=summary,
        message_type=message_type,
        timestamp=time.time(),
        metadata=metadata or {},
    )
