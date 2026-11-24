"""后台任务的持有与回收。

``asyncio.ensure_future`` / ``asyncio.create_task`` 的返回值**必须被持有引用**：
事件循环只保留弱引用，任务对象可能在执行途中被垃圾回收，导致
"任务莫名其妙没跑完"。这里维护一个模块级的强引用集合，任务完成后自动移除。

顺带记录未处理的后台异常——否则它们会被 asyncio 静默丢弃，
只留下一条 "Task exception was never retrieved" 警告。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

log = logging.getLogger(__name__)

_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()


def _on_task_done(task: asyncio.Task[Any]) -> None:
    _BACKGROUND_TASKS.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log.warning("background task failed: %r", exc)


def spawn_background(coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
    """调度一个后台任务并持有引用，防止被 GC 提前回收。

    用于"发射后不管"的场景（记忆提取、整合、通知派发等）。
    需要等待结果的任务请直接 ``await``，不要用这个。
    """
    task = asyncio.ensure_future(coro)
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_on_task_done)
    return task


def pending_background_tasks() -> set[asyncio.Task[Any]]:
    """返回仍在运行的后台任务（供关闭流程与测试使用）。"""
    return {t for t in _BACKGROUND_TASKS if not t.done()}
