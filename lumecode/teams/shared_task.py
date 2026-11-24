from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from lumecode.fileutil import atomic_write_text, file_lock


@dataclass
class SharedTask:
    id: str
    title: str
    description: str = ""
    status: str = "pending"  # pending | in_progress | completed | blocked
    assignee: str = ""
    blocks: list[str] = field(default_factory=list)
    blocked_by: list[str] = field(default_factory=list)
    created_by: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SharedTask:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class SharedTaskStore:
    """团队共享任务板，落盘为单个 JSON 文件。

    tmux / iTerm2 队友是**独立进程**，会同时读写同一个 tasks.json。
    因此所有写操作都在跨进程文件锁内执行「重新加载 → 修改 → 原子保存」，
    避免丢更新和 next_id 冲突。旧实现是无锁的 load-modify-save。
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._next_id = 1
        self._tasks: dict[str, SharedTask] = {}
        self._load()

    @property
    def _lock_path(self) -> Path:
        return self._path.with_name(self._path.name + ".lock")

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """在文件锁内重新加载最新状态，退出时由调用方保存。"""
        with file_lock(self._lock_path):
            self._load()
            yield

    def _load(self) -> None:
        self._tasks = {}
        self._next_id = 1
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # 文件损坏或被并发删除时退化为空任务板，而不是让整个工具崩溃。
            return
        self._next_id = data.get("next_id", 1)
        for t in data.get("tasks", []):
            task = SharedTask.from_dict(t)
            self._tasks[task.id] = task

    def _save(self) -> None:
        data = {
            "next_id": self._next_id,
            "tasks": [t.to_dict() for t in self._tasks.values()],
        }
        atomic_write_text(self._path, json.dumps(data, indent=2, ensure_ascii=False))

    def create(
        self,
        title: str,
        description: str = "",
        assignee: str = "",
        blocks: list[str] | None = None,
        blocked_by: list[str] | None = None,
        created_by: str = "",
    ) -> SharedTask:
        with self._locked():
            task_id = str(self._next_id)
            self._next_id += 1
            task = SharedTask(
                id=task_id,
                title=title,
                description=description,
                assignee=assignee,
                blocks=blocks or [],
                blocked_by=blocked_by or [],
                created_by=created_by,
            )
            self._tasks[task_id] = task
            self._save()
            return task

    def get(self, task_id: str) -> SharedTask | None:
        self._load()
        return self._tasks.get(task_id)

    def list_tasks(
        self,
        status: str | None = None,
        assignee: str | None = None,
    ) -> list[SharedTask]:
        self._load()
        result = list(self._tasks.values())
        if status:
            result = [t for t in result if t.status == status]
        if assignee:
            result = [t for t in result if t.assignee == assignee]
        return result

    def update(
        self,
        task_id: str,
        status: str | None = None,
        assignee: str | None = None,
        description: str | None = None,
        add_blocks: list[str] | None = None,
        add_blocked_by: list[str] | None = None,
    ) -> SharedTask | None:
        with self._locked():
            task = self._tasks.get(task_id)
            if task is None:
                return None
            if status is not None:
                task.status = status
            if assignee is not None:
                task.assignee = assignee
            if description is not None:
                task.description = description
            if add_blocks:
                for bid in add_blocks:
                    if bid not in task.blocks:
                        task.blocks.append(bid)
            if add_blocked_by:
                for bid in add_blocked_by:
                    if bid not in task.blocked_by:
                        task.blocked_by.append(bid)
            self._save()
            return task

    def init_empty(self) -> None:
        with self._locked():
            self._tasks.clear()
            self._next_id = 1
            self._save()
