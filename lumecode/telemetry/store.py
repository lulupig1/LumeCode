"""会话级 JSONL trace 存储。

存储布局与现有 ``.lumecode/sessions/`` 保持一致：

    .lumecode/traces/<session_id>.jsonl

每行是一条已经结束的 span 记录。写入是 append-only 的，读取方可以容忍
损坏的行，因此进程崩溃最多丢掉正在写入的最后一行，不会破坏整个文件。
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lumecode.telemetry.models import TELEMETRY_SCHEMA_VERSION, Span

log = logging.getLogger(__name__)

TRACES_DIR = ".lumecode/traces"
DEFAULT_RETENTION_DAYS = 30
_MAX_SESSION_ID_LEN = 80

# session_id 可能来自用户输入或外部系统，做保守清洗，避免路径穿越。
_SAFE_SESSION_RE = re.compile(r"[^A-Za-z0-9._-]+")

# 同一进程内多个 recorder 可能写同一个会话文件；跨进程依赖 O_APPEND 的原子性。
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(path: str) -> threading.Lock:
    with _LOCKS_GUARD:
        lock = _LOCKS.get(path)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[path] = lock
        return lock


def sanitize_session_id(session_id: str) -> str:
    """把任意 session id 转成安全的文件名片段。"""
    cleaned = _SAFE_SESSION_RE.sub("_", session_id).strip("._-")
    if not cleaned:
        cleaned = "unknown"
    return cleaned[:_MAX_SESSION_ID_LEN]


@dataclass(frozen=True)
class TraceFileInfo:
    """一个会话 trace 文件的元信息。"""

    session_id: str
    path: str
    size_bytes: int
    mtime: float

    @property
    def age_days(self) -> int:
        return max(int((time.time() - self.mtime) // 86_400), 0)


class TraceStore:
    """把 span 追加写入会话 JSONL 文件。"""

    def __init__(self, base_dir: str | os.PathLike[str] = TRACES_DIR) -> None:
        self.base_dir = Path(base_dir)

    def path_for(self, session_id: str) -> Path:
        return self.base_dir / f"{sanitize_session_id(session_id)}.jsonl"

    def append_span(self, span: Span) -> bool:
        """追加一条 span。失败时返回 False，绝不向调用方抛异常。"""
        record: dict[str, Any] = {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "type": "span",
            **span.to_dict(),
        }
        return self.append_record(span.session_id, record)

    def append_record(self, session_id: str, record: dict[str, Any]) -> bool:
        path = self.path_for(session_id)
        try:
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        except (TypeError, ValueError) as e:
            # 调用方可能绕过 TelemetryRecorder 直接传任意对象。
            log.debug("trace record is not JSON serializable: %s", e)
            return False
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with _lock_for(str(path)), path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
            return True
        except OSError as e:
            # telemetry 是 best-effort：写盘失败只记 debug，不影响主流程。
            log.debug("trace append failed for %s: %s", path, e)
            return False

    def read(self, session_id: str) -> list[dict[str, Any]]:
        """读取一个会话的所有记录，跳过损坏的行。"""
        path = self.path_for(session_id)
        records: list[dict[str, Any]] = []
        try:
            with path.open(encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(value, dict):
                        records.append(value)
        except OSError as e:
            log.debug("trace read failed for %s: %s", path, e)
        return records

    def list_sessions(self) -> list[TraceFileInfo]:
        """按修改时间倒序返回已有的 trace 文件。"""
        try:
            entries = list(self.base_dir.glob("*.jsonl"))
        except OSError:
            return []

        result: list[TraceFileInfo] = []
        for path in entries:
            try:
                stat = path.stat()
            except OSError:
                continue
            if not path.is_file():
                continue
            result.append(
                TraceFileInfo(
                    session_id=path.stem,
                    path=str(path),
                    size_bytes=stat.st_size,
                    mtime=stat.st_mtime,
                )
            )
        result.sort(key=lambda item: item.mtime, reverse=True)
        return result

    def cleanup(self, max_age_days: int = DEFAULT_RETENTION_DAYS) -> int:
        """删除超过保留期的 trace 文件，返回删除数量。"""
        if max_age_days <= 0:
            return 0
        cutoff = time.time() - max_age_days * 86_400
        removed = 0
        for info in self.list_sessions():
            if info.mtime >= cutoff:
                continue
            try:
                Path(info.path).unlink()
                removed += 1
            except OSError as e:
                log.debug("trace cleanup failed for %s: %s", info.path, e)
        return removed
